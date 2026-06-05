// YawSmoothing.swift — Smoothing filters and yaw-to-balance mapping.
//
// Direct port of the Python SmoothedYaw, EMA, yaw_to_balance, and
// _balance_indicator from head_track_audio.py / menubar_app.py.

import Foundation

// MARK: - Constants (must match head_track_audio.py)

let kDeadZoneDeg: Double = 8.0
let kMaxYawDeg:   Double = 45.0
let kMaxBalance:  Double = 0.38

/// Scale converting the eye/nose geometric ratio into degrees of yaw.
/// (Eyes sit closer together than ears, so this is smaller than the Python
/// ear-based script's 40°.)  A ratio of ~1.0 ≈ a strong deliberate turn.
let kRatioToDeg:  Double = 45.0

// MARK: - SmoothedYaw (double-EMA + outlier gate)

/// Two-stage smoother with outlier rejection for yaw angles.
///
/// Stage 1: clamp jumps larger than `maxJump` degrees.
/// Stage 2: double-EMA (two cascaded exponential moving averages).
final class SmoothedYaw {
    private let alpha: Double
    private let maxJump: Double
    private var s1: Double = 0
    private var s2: Double = 0
    private var initialised = false

    /// The current smoothed value.
    var value: Double { s2 }

    init(alpha: Double = 0.18, maxJump: Double = 25.0) {
        self.alpha   = alpha
        self.maxJump = maxJump
    }

    @discardableResult
    func update(_ sample: Double) -> Double {
        guard initialised else {
            s1 = sample; s2 = sample
            initialised = true
            return sample
        }

        var clamped = sample
        let diff = sample - s1
        if abs(diff) > maxJump {
            clamped = s1 + maxJump * (diff > 0 ? 1.0 : -1.0)
        }

        s1 += alpha * (clamped - s1)
        s2 += alpha * (s1 - s2)
        return s2
    }

    func reset() {
        s1 = 0; s2 = 0; initialised = false
    }
}

// MARK: - EMA (single exponential moving average)

final class EMA {
    private let alpha: Double
    private(set) var value: Double

    init(alpha: Double = 0.15, initial: Double = 0.0) {
        self.alpha = alpha
        self.value = initial
    }

    @discardableResult
    func update(_ sample: Double) -> Double {
        value += alpha * (sample - value)
        return value
    }

    func reset(to val: Double = 0.0) { value = val }
}

// MARK: - Yaw → Balance mapping

/// Map a yaw angle (degrees) to a balance value in `[-kMaxBalance, +kMaxBalance]`.
///
/// Uses a dead-zone, saturation, and quadratic curve — identical to the
/// Python `yaw_to_balance()`.
func yawToBalance(_ yawDeg: Double) -> Double {
    guard abs(yawDeg) > kDeadZoneDeg else { return 0.0 }

    let sign: Double = yawDeg > 0 ? 1.0 : -1.0
    let effective = abs(yawDeg) - kDeadZoneDeg
    let span      = kMaxYawDeg - kDeadZoneDeg   // 37°
    let ratio     = min(effective / span, 1.0)
    return sign * (ratio * ratio) * kMaxBalance
}

// MARK: - Balance indicator (menu bar ASCII bar)

/// Return a short string like `"◀  ●   "` showing the current balance.
func balanceIndicator(_ balance: Double) -> String {
    let clamped = max(-kMaxBalance, min(kMaxBalance, balance))
    var pos = Int((clamped / kMaxBalance + 1.0) / 2.0 * 4.0 + 0.5)
    pos = max(0, min(4, pos))

    var slots = [String](repeating: " ", count: 5)
    slots[pos] = "●"

    let left  = pos < 2 ? "◀" : " "
    let right = pos > 2 ? "▶" : " "
    return "\(left)\(slots.joined())\(right)"
}
