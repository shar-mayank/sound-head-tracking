#!/usr/bin/env python3
"""
head_track_audio.py — macOS head-tracking audio balance control.

Uses the webcam + MediaPipe Face Mesh to detect head yaw, then maps
that angle to the macOS system stereo balance via CoreAudio so the
sound follows your head orientation.

Requires macOS 13+ and Python 3.10+.
"""

import atexit
import os
import signal
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Suppress noisy MediaPipe / TensorFlow Lite / absl logging before imports.
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# CoreAudio via pyobjc-framework-CoreAudio
# ---------------------------------------------------------------------------
from CoreAudio import (
    AudioObjectGetPropertyData,
    AudioObjectHasProperty,
    AudioObjectSetPropertyData,
)

# FourCC property selectors packed as UInt32.
_FOURCC = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731

kAudioHardwarePropertyDefaultOutputDevice = _FOURCC("dOut")
kAudioDevicePropertyStereoPan = _FOURCC("span")
kAudioDevicePropertyVolumeScalar = _FOURCC("volm")
kAudioObjectPropertyScopeGlobal = _FOURCC("glob")
kAudioObjectPropertyScopeOutput = _FOURCC("outp")
kAudioObjectPropertyElementMain = 0
kAudioObjectSystemObject = 1

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_property_float(obj_id: int, address: tuple) -> float:
    status, _size, data = AudioObjectGetPropertyData(obj_id, address, 0, b"", 4, None)
    if status != 0:
        raise RuntimeError(f"AudioObjectGetPropertyData failed (status {status})")
    return struct.unpack("f", bytes(data))[0]


def _set_property_float(obj_id: int, address: tuple, value: float) -> None:
    status = AudioObjectSetPropertyData(obj_id, address, 0, b"", 4, struct.pack("f", value))
    if isinstance(status, tuple):
        status = status[0]
    if status != 0:
        raise RuntimeError(f"AudioObjectSetPropertyData failed (status {status})")


def _get_property_uint32(obj_id: int, address: tuple) -> int:
    status, _size, data = AudioObjectGetPropertyData(obj_id, address, 0, b"", 4, None)
    if status != 0:
        raise RuntimeError(f"AudioObjectGetPropertyData failed (status {status})")
    return struct.unpack("I", bytes(data))[0]


# ---------------------------------------------------------------------------
# CoreAudio: default output device
# ---------------------------------------------------------------------------

def get_default_output_device() -> int:
    """Return the AudioObjectID of the default output device."""
    address = (kAudioHardwarePropertyDefaultOutputDevice,
               kAudioObjectPropertyScopeGlobal,
               kAudioObjectPropertyElementMain)
    return _get_property_uint32(kAudioObjectSystemObject, address)


# ---------------------------------------------------------------------------
# CoreAudio: balance via StereoPan or per-channel volume fallback
#
# Strategy:
#   1. If the device supports kAudioDevicePropertyStereoPan, use it directly
#      (Float32 in 0.0 … 1.0, where 0.0 = left, 0.5 = centre, 1.0 = right).
#   2. Otherwise fall back to per-channel VolumeScalar on elements 1 (left)
#      and 2 (right).  We record the original volumes on startup and scale
#      them: the "quieter" side is reduced proportionally to the balance
#      value while the "louder" side stays at the original level.
# ---------------------------------------------------------------------------

class _BalanceController:
    """Abstraction over StereoPan vs per-channel volume."""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self._use_stereo_pan = False
        self._orig_left = 1.0
        self._orig_right = 1.0
        # For PerChannelVolume: track the "user-intended" base volume so we
        # don't feed our own balance-adjusted values back into the next frame.
        self._last_base = None
        # The last balance value we applied (so we can back-calculate the
        # user's intended volume when they change it mid-run).
        self._last_balance = 0.0

        pan_addr = (kAudioDevicePropertyStereoPan,
                    kAudioObjectPropertyScopeOutput,
                    kAudioObjectPropertyElementMain)
        if AudioObjectHasProperty(device_id, pan_addr):
            self._use_stereo_pan = True
            self._orig_pan = _get_property_float(device_id, pan_addr)
            return

        # Per-channel volume fallback (element 1 = left, element 2 = right).
        self._left_addr = (kAudioDevicePropertyVolumeScalar,
                           kAudioObjectPropertyScopeOutput, 1)
        self._right_addr = (kAudioDevicePropertyVolumeScalar,
                            kAudioObjectPropertyScopeOutput, 2)
        if not (AudioObjectHasProperty(device_id, self._left_addr) and
                AudioObjectHasProperty(device_id, self._right_addr)):
            raise RuntimeError(
                "Output device supports neither StereoPan nor per-channel "
                "VolumeScalar — cannot control balance.")
        self._orig_left = _get_property_float(device_id, self._left_addr)
        self._orig_right = _get_property_float(device_id, self._right_addr)
        # Normalize near-equal L/R volumes to avoid displaying spurious
        # asymmetry caused by CoreAudio floating-point imprecision.
        if abs(self._orig_left - self._orig_right) < 0.02:
            base = round(max(self._orig_left, self._orig_right), 2)
            self._orig_left = base
            self._orig_right = base
            _set_property_float(device_id, self._left_addr, base)
            _set_property_float(device_id, self._right_addr, base)
        self._last_base = max(self._orig_left, self._orig_right)

    # -- public API --

    @property
    def method(self) -> str:
        return "StereoPan" if self._use_stereo_pan else "PerChannelVolume"

    @property
    def original_description(self) -> str:
        if self._use_stereo_pan:
            # Convert from 0..1 StereoPan to -1..+1 display range.
            return f"{self._orig_pan * 2.0 - 1.0:+.2f}"
        return f"L={self._orig_left:.2f} R={self._orig_right:.2f}"

    def set_balance(self, value: float) -> None:
        """Apply *value* (-1.0 … 1.0) as a stereo balance.

        StereoPan: convert balance to 0..1 pan range and set it (ignoring
        the device's original pan — we own it while the script runs and
        restore on exit).

        PerChannelVolume: detect user volume changes by comparing the
        current louder channel against what we last wrote, then apply
        the balance ratio to the user's intended base volume.
        """
        value = max(-1.0, min(1.0, value))
        if self._use_stereo_pan:
            addr = (kAudioDevicePropertyStereoPan,
                    kAudioObjectPropertyScopeOutput,
                    kAudioObjectPropertyElementMain)
            # StereoPan range is 0.0 (left) .. 1.0 (right); convert from
            # balance range -1.0 .. +1.0.
            pan = (value + 1.0) / 2.0
            _set_property_float(self.device_id, addr, pan)
        else:
            # Read current volumes to detect user volume changes.
            cur_left = _get_property_float(self.device_id, self._left_addr)
            cur_right = _get_property_float(self.device_id, self._right_addr)
            cur_max = max(cur_left, cur_right)

            # Figure out what we *expected* the louder channel to be based
            # on our last write.  If the actual value differs significantly,
            # the user changed the volume externally — update our base.
            if self._last_base is not None:
                expected_loud = self._last_base  # we always set the loud side to base
                if abs(cur_max - expected_loud) > 0.005:
                    # User changed volume — adopt the new level.
                    self._last_base = cur_max
            base = self._last_base if self._last_base is not None else cur_max

            if value <= 0:
                left_vol = base
                right_vol = base * (1.0 + value)   # value is negative
            else:
                left_vol = base * (1.0 - value)
                right_vol = base
            _set_property_float(self.device_id, self._left_addr,
                                max(0.0, min(1.0, left_vol)))
            _set_property_float(self.device_id, self._right_addr,
                                max(0.0, min(1.0, right_vol)))
            self._last_balance = value

    def restore(self) -> None:
        """Restore original balance / volumes."""
        try:
            if self._use_stereo_pan:
                addr = (kAudioDevicePropertyStereoPan,
                        kAudioObjectPropertyScopeOutput,
                        kAudioObjectPropertyElementMain)
                _set_property_float(self.device_id, addr, self._orig_pan)
            else:
                _set_property_float(self.device_id, self._left_addr, self._orig_left)
                _set_property_float(self.device_id, self._right_addr, self._orig_right)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Yaw extraction via landmark geometry
# ---------------------------------------------------------------------------
# Instead of solvePnP (which is sensitive to model/camera assumptions and
# produces wild values), we use a direct geometric ratio:
#
#   nose_tip (1) vs midpoint of left_ear (234) and right_ear (454).
#
# When looking straight ahead the nose is centred between the ears.
# When turning right the nose moves toward the right ear, and vice versa.
# The ratio (nose_x - mid_x) / half_ear_distance gives a stable [-1, 1]
# signal that we scale to approximate degrees.
#
# This is immune to the sign-flip and extreme-value problems of Euler
# angle decomposition.

# Landmark indices.
_NOSE_TIP = 1
_LEFT_EAR = 234   # left tragion
_RIGHT_EAR = 454  # right tragion

# Empirical scale: a ratio of 1.0 corresponds to roughly this many degrees
# of yaw.  Kept conservative so small head movements stay near zero.
_RATIO_TO_DEG = 40.0


def estimate_yaw(landmarks, frame_w: int, frame_h: int) -> float | None:
    """Return the head yaw angle in degrees, or None if landmarks are bad.

    Positive = head turned right, negative = head turned left.
    """
    nose = landmarks[_NOSE_TIP]
    left_ear = landmarks[_LEFT_EAR]
    right_ear = landmarks[_RIGHT_EAR]

    # Sanity: all three landmarks should be roughly in-frame.
    for lm in (nose, left_ear, right_ear):
        if not (-0.1 <= lm.x <= 1.1 and -0.1 <= lm.y <= 1.1):
            return None

    ear_mid_x = (left_ear.x + right_ear.x) / 2.0
    ear_span = abs(right_ear.x - left_ear.x)

    # If ears are nearly overlapping the face is too far or sideways — bail.
    if ear_span < 0.02:
        return None

    # How far the nose is from the ear midpoint, normalised by ear span.
    ratio = (nose.x - ear_mid_x) / (ear_span / 2.0)

    # Clamp to avoid crazy values from bad detections.
    ratio = max(-1.5, min(1.5, ratio))

    return ratio * _RATIO_TO_DEG


# ---------------------------------------------------------------------------
# Yaw → balance mapping
#
# Design goal: the audio should feel like it's coming from the laptop.
# Small head turns (typical during normal use) should produce very little
# shift.  Only a deliberate, large turn should move the balance noticeably.
#
#   - Wide dead zone (±8°) so looking roughly at the screen = centred.
#   - Saturation at ±45° — you have to really turn your head.
#   - Max balance ±0.35 — even at full turn both ears still get most of
#     the audio; it's a subtle spatial cue, not a hard pan.
#   - Quadratic curve so the first few degrees beyond the dead zone
#     barely register, and the effect ramps up gradually.
# ---------------------------------------------------------------------------

DEAD_ZONE_DEG = 8.0
MAX_YAW_DEG = 45.0
MAX_BALANCE = 0.38


def yaw_to_balance(yaw_deg: float) -> float:
    """Map a yaw angle (degrees) to a balance value with dead zone and clamp.

    Uses a quadratic curve so small movements are nearly silent and the
    effect builds gradually with larger turns.
    """
    if abs(yaw_deg) <= DEAD_ZONE_DEG:
        return 0.0

    sign = 1.0 if yaw_deg > 0 else -1.0
    effective = abs(yaw_deg) - DEAD_ZONE_DEG
    span = MAX_YAW_DEG - DEAD_ZONE_DEG  # 37°
    ratio = min(effective / span, 1.0)
    # Quadratic curve: gentle at small angles, steeper at large ones.
    return sign * (ratio ** 2) * MAX_BALANCE


# ---------------------------------------------------------------------------
# Smoothing (double EMA + outlier gate)
# ---------------------------------------------------------------------------

class SmoothedYaw:
    """Two-stage smoother with outlier rejection for yaw angles.

    Stage 1: reject samples that jump more than *max_jump* degrees from the
             current smoothed value (likely a solvePnP glitch).
    Stage 2: double-EMA (two cascaded exponential moving averages) for a
             smoother, less laggy response than a single very-low-alpha EMA.
    """

    def __init__(self, alpha: float = 0.28, max_jump: float = 25.0):
        self.alpha = alpha
        self.max_jump = max_jump
        self._s1 = 0.0  # first EMA stage
        self._s2 = 0.0  # second EMA stage
        self._initialised = False

    @property
    def value(self) -> float:
        return self._s2

    def update(self, sample: float) -> float:
        if not self._initialised:
            self._s1 = sample
            self._s2 = sample
            self._initialised = True
            return sample

        # Outlier gate: clamp large jumps instead of ignoring them,
        # so fast head snaps still move toward the new position.
        diff = sample - self._s1
        if abs(diff) > self.max_jump:
            sample = self._s1 + self.max_jump * (1.0 if diff > 0 else -1.0)

        self._s1 += self.alpha * (sample - self._s1)
        self._s2 += self.alpha * (self._s1 - self._s2)
        return self._s2


class EMA:
    """Simple exponential moving average (used for balance output)."""

    def __init__(self, alpha: float = 0.15, initial: float = 0.0):
        self.alpha = alpha
        self.value = initial

    def update(self, sample: float) -> float:
        self.value += self.alpha * (sample - self.value)
        return self.value


# ---------------------------------------------------------------------------
# Terminal display helpers
# ---------------------------------------------------------------------------

def balance_bar(balance: float, width: int = 10) -> str:
    """Render a simple ASCII balance bar centred at 0."""
    # Map balance (-0.6 … +0.6) to position (0 … width)
    pos = int((balance / MAX_BALANCE + 1.0) / 2.0 * width)
    pos = max(0, min(width, pos))
    bar = list("·" * (width + 1))
    bar[pos] = "="
    # Fill from centre to pos
    centre = width // 2
    lo, hi = min(centre, pos), max(centre, pos)
    for i in range(lo, hi + 1):
        bar[i] = "="
    return "[" + "".join(bar) + "]"


# ---------------------------------------------------------------------------
# MediaPipe model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = "face_landmarker.task"


def _ensure_model() -> str:
    """Download the FaceLandmarker model if it doesn't exist. Return path."""
    import os
    import urllib.request

    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    print(f"  Downloading model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("  Done.")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Download model if needed (before suppressing stderr) ---
    model_path = _ensure_model()

    # Suppress stderr to hide C++ log spam (MediaPipe/TF/absl) during init.
    _init_devnull = os.open(os.devnull, os.O_WRONLY)
    _init_stderr = os.dup(2)
    os.dup2(_init_devnull, 2)

    # --- Parallel startup: webcam + MediaPipe model load ---
    def _open_cam():
        c = cv2.VideoCapture(0)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return c

    def _load_landmarker():
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path,
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return FaceLandmarker.create_from_options(options)

    with ThreadPoolExecutor(max_workers=2) as pool:
        cam_future = pool.submit(_open_cam)
        lm_future = pool.submit(_load_landmarker)
        cap = cam_future.result()
        landmarker = lm_future.result()

    # Restore stderr for user-facing output.
    os.dup2(_init_stderr, 2)
    os.close(_init_stderr)
    os.close(_init_devnull)

    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).", file=sys.stderr)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Startup: CoreAudio ---
    device_id = get_default_output_device()
    bal_ctrl = _BalanceController(device_id)

    def restore_balance(*_args) -> None:
        """Restore original volumes on exit."""
        bal_ctrl.restore()

    atexit.register(restore_balance)
    _running = True

    def _stop(*_args):
        nonlocal _running
        _running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    yaw_smoother = SmoothedYaw(alpha=0.18, max_jump=25.0)
    balance_smoother = EMA(alpha=0.18, initial=0.0)
    frame_interval = 1.0 / 60.0  # cap at 60 fps
    frame_count = 0
    last_yaw = None          # last successfully estimated yaw
    no_face_since = None     # monotonic time when face was last lost
    WARMUP_FRAMES = 10       # ignore first N frames (detection is noisy)

    print("=" * 56)
    print("  Head-Tracking Audio Balance  (macOS CoreAudio)")
    print("=" * 56)
    print(f"  Webcam        : index 0  ({frame_w}x{frame_h})")
    print(f"  Output device : id {device_id}  ({bal_ctrl.method})")
    print(f"  Original bal. : {bal_ctrl.original_description}")
    print(f"  Dead zone     : +/-{DEAD_ZONE_DEG:.0f} deg")
    print(f"  Max balance   : +/-{MAX_BALANCE}")
    print(f"  Max yaw       : +/-{MAX_YAW_DEG:.0f} deg")
    print("-" * 56)
    print("  Press Ctrl+C to quit (balance restored).")
    print("-" * 56)
    print()

    # Redirect stderr to /dev/null to silence C++ clearcut/absl log spam
    # that env vars cannot suppress.  Our output goes to stdout.
    _devnull = open(os.devnull, "w")
    _orig_stderr = os.dup(2)
    os.dup2(_devnull.fileno(), 2)

    try:
        while _running:
            loop_start = time.monotonic()

            # Detect output device changes (e.g. headphone hot-swap).
            try:
                new_device_id = get_default_output_device()
            except RuntimeError:
                new_device_id = device_id
            if new_device_id != device_id:
                # Restore old device, switch to new one.
                bal_ctrl.restore()
                device_id = new_device_id
                try:
                    bal_ctrl = _BalanceController(device_id)
                except RuntimeError as e:
                    print(f"\n  WARNING: new device unsupported ({e}), waiting...")
                    time.sleep(1)
                    continue
                # Reset smoothers so we don't carry stale balance into new device.
                balance_smoother = EMA(alpha=0.18, initial=0.0)
                print(
                    f"\r  Switched to device id {device_id} ({bal_ctrl.method})"
                    f"                        ",
                    flush=True,
                )

            ret, frame = cap.read()
            if not ret:
                continue

            # Flip horizontally — Mac webcams mirror the image, which
            # inverts left/right and makes yaw point the wrong way.
            frame = cv2.flip(frame, 1)

            frame_count += 1

            # Skip warmup frames — early detections are often jittery.
            if frame_count <= WARMUP_FRAMES:
                continue

            # Run MediaPipe on every frame for snappier tracking.
            yaw_deg = None
            face_found = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )
            timestamp_ms = int(time.monotonic() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                lm = results.face_landmarks[0]
                yaw_deg = estimate_yaw(lm, frame_w, frame_h)

            face_found = yaw_deg is not None

            # Decide the balance target.
            if face_found:
                # Good detection — smooth the raw yaw first.
                smooth_yaw = yaw_smoother.update(yaw_deg)
                last_yaw = smooth_yaw
                no_face_since = None
                target = yaw_to_balance(smooth_yaw)
            else:
                # No usable face — drift to centre.
                if no_face_since is None:
                    no_face_since = time.monotonic()
                elapsed_no_face = time.monotonic() - no_face_since
                fade = max(0.0, 1.0 - elapsed_no_face)
                target = yaw_to_balance(last_yaw) * fade if last_yaw is not None else 0.0

            balance = balance_smoother.update(target)

            # Negate: head turns left → sound shifts right (toward the
            # laptop), so the audio feels anchored to the screen.
            balance = -balance

            # Apply to system audio.
            try:
                bal_ctrl.set_balance(balance)
            except RuntimeError:
                pass  # non-fatal; keep running

            # Display: show the smoothed yaw, not the raw noisy value.
            if last_yaw is not None:
                yaw_str = f"{yaw_smoother.value:+6.1f}"
            else:
                yaw_str = "  ---"
            bar = balance_bar(balance)
            print(
                f"\r  Yaw: {yaw_str}\u00b0  |  Balance: {balance:+.2f}  |  {bar}  ",
                end="",
                flush=True,
            )

            # Frame-rate cap.
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # Restore stderr so cleanup messages are visible.
        os.dup2(_orig_stderr, 2)
        os.close(_orig_stderr)
        _devnull.close()
        cap.release()
        landmarker.close()
        restore_balance()
        print("\n  Balance restored. Goodbye :)")


if __name__ == "__main__":
    main()
