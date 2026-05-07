# Head-Tracking Audio Balance for macOS

A terminal-based Python application that uses your webcam to detect head rotation
(yaw) via MediaPipe Face Mesh and maps it to the macOS system stereo balance
through CoreAudio in real time.

Turn your head right and the audio shifts right. Turn left and it shifts left.
A dead zone around centre and exponential smoothing prevent jitter.

## Requirements

- macOS 13 or later
- Python 3.10 or later
- A built-in or USB webcam
- [uv](https://docs.astral.sh/uv/) package manager

## Install

1. Clone the repository and enter the directory:

```
git clone <repo-url>
cd apple-sound-head-tracking
```

2. Create the virtual environment and install dependencies with uv:

```
uv sync
```

This reads `pyproject.toml` and installs `opencv-python`, `mediapipe`,
`numpy`, and `pyobjc-framework-CoreAudio` into a local `.venv`.

## Usage

Run the script through uv:

```
uv run python head_track_audio.py
```

Or activate the environment first:

```
source .venv/bin/activate
python head_track_audio.py
```

On first launch macOS will ask for camera permission. Grant it in
System Settings > Privacy & Security > Camera.

The terminal will show a single updating line with the current yaw angle,
balance value, and a visual bar:

```
Yaw:  -12.3 deg  |  Balance: -0.28  |  [=====.........]
```

Press Ctrl+C to quit. The system audio balance is always reset to centre
on exit.

## How it works

1. Captures frames from the webcam at up to 30 fps.
2. Runs MediaPipe Face Mesh (every other frame) to get 468 facial landmarks.
3. Feeds six key landmarks into `cv2.solvePnP` with a canonical 3-D face
   model to extract the head yaw angle.
4. Maps yaw to a balance value:
   - Dead zone of +/-5 degrees around centre (balance stays at 0.0).
   - Linear mapping from 5 to 30 degrees to a balance of 0.0 to 0.6.
   - Clamped at +/-0.6 so both ears always receive audio.
5. Applies an exponential moving average (alpha 0.15) for smooth transitions.
6. Writes the balance to the default output device via CoreAudio
   `kAudioDevicePropertyStereoPan`.

## Configuration

Edit the constants near the top of `head_track_audio.py`:

| Constant        | Default | Description                              |
|-----------------|---------|------------------------------------------|
| `DEAD_ZONE_DEG` | 5.0     | Degrees of yaw ignored around centre     |
| `MAX_YAW_DEG`   | 30.0    | Yaw angle at which balance saturates     |
| `MAX_BALANCE`   | 0.6     | Maximum stereo pan value (0.0 to 1.0)    |
| EMA `alpha`     | 0.15    | Smoothing factor (lower = smoother)      |

## Troubleshooting

**Camera not found**: Make sure no other application is using the webcam.
Try passing a different camera index in the `cv2.VideoCapture()` call.

**Permission denied for audio**: The script needs no special entitlements
for CoreAudio stereo pan. If balance does not change, check that your
output device supports `kAudioDevicePropertyStereoPan` (most built-in
speakers and headphone outputs do).

**MediaPipe import errors on Apple Silicon**: Ensure you are using
Python 3.10+ built for arm64. uv handles this automatically.

## License

MIT
