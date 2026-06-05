// AppDelegate.swift — Menu bar application (NSStatusItem).
//
// Port of the Python menubar_app.py.  Runs as an LSUIElement (no Dock
// icon).  The camera is opened in-process so macOS TCC correctly
// attributes it to this app's bundle identity.

import AppKit
import CoreAudio
import UserNotifications

// MARK: - Constants

private let kVersion   = "1.0.0"
private let kIssuesURL = "https://github.com/shar-mayank/sound-head-tracking/issues/new"
private let kSoundSettingsURL = "x-apple.systempreferences:com.apple.preference.sound"

/// Balance magnitude above which the menu bar shows a directional arrow
/// instead of the centred dot.
private let kArrowThreshold = 0.05

// MARK: - AppDelegate

final class AppDelegate: NSObject, NSApplicationDelegate, HeadTrackerDelegate {

    // ---- Menu bar ----
    private var statusItem: NSStatusItem!

    // ---- Menu items ----
    private var yawItem:        NSMenuItem!
    private var balanceItem:    NSMenuItem!
    private var versionItem:    NSMenuItem!
    private var deviceNameItem: NSMenuItem!
    private var deviceTypeItem: NSMenuItem!
    private var toggleItem:     NSMenuItem!

    // ---- Tracking state ----
    private var headTracker:      HeadTracker?
    private var balanceCtrl:      AudioBalanceController?
    private var trackingOn       = false
    private var yawSmoother      = SmoothedYaw(alpha: 0.5, maxJump: 40.0)
    private var balanceSmoother  = EMA(alpha: 0.5)
    private var lastYaw:         Double?
    private var yawCenter:       Double = 0       // neutral-yaw estimate
    private var yawCenterInit    = false
    // First-second neutral calibration (handles a large constant bias that the
    // near-centre adaptive drift can't climb to on its own).
    private var calibrating      = false
    private var calibSamples:    [Double] = []
    private var calibStart:      TimeInterval = 0
    private var noFaceSince:     TimeInterval?
    private var currentBalance:  Double = 0

    // ---- Face-lost notification throttle ----
    private var faceWasDetected   = false
    private var faceLostSince:    TimeInterval?
    private var lastFaceLostNote: TimeInterval = 0

    // ---- Timers ----
    private var displayTimer: Timer?

    // ==================================================================
    // MARK: - NSApplicationDelegate
    // ==================================================================

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)   // no Dock icon

        buildStatusItem()
        buildMenu()
        listenForDeviceChanges()

        // Request notification permission (non-blocking).
        UNUserNotificationCenter.current()
            .requestAuthorization(options: [.alert]) { _, _ in }

        // Restore last-known toggle state.
        if UserDefaults.standard.bool(forKey: "trackingEnabled") {
            enableTracking()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        disableTracking()
    }

    // ==================================================================
    // MARK: - Status item
    // ==================================================================

    private func buildStatusItem() {
        // IMPORTANT: keep the status item at `variableLength` (sized to its
        // content) at ALL times.  A fixed / oversized width makes the item's
        // frame extend rightward into the region the system reserves for the
        // green camera "in use" indicator, which then draws on top of our
        // icon.  A compact item always sits cleanly to the LEFT of that
        // indicator — exactly what Discord and Hand Mirror do.
        statusItem = NSStatusBar.system.statusItem(
            withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            button.image = NSImage(
                systemSymbolName: "headphones",
                accessibilityDescription: "Sound Head Tracking")
            button.image?.isTemplate = true        // adapts to light/dark bar
            button.imagePosition = .imageLeading    // icon, then optional arrow
        }
        updateStatusBar()
    }

    /// Refresh the menu bar appearance.  The icon is always the headphones
    /// symbol; when tracking is active a single compact arrow/dot shows the
    /// balance direction.  Never touches `statusItem.length` — it stays
    /// `variableLength` so the camera indicator cannot overlap it.
    private func updateStatusBar() {
        guard let button = statusItem.button else { return }
        guard trackingOn else { button.title = ""; return }

        // Arrow points where the HEAD is turned.  `currentBalance` is the
        // (negated) audio balance, so a positive balance means the head is
        // turned left and vice-versa — hence the inverted comparison here.
        let arrow: String
        if currentBalance > kArrowThreshold      { arrow = "◀" }
        else if currentBalance < -kArrowThreshold { arrow = "▶" }
        else                                      { arrow = "•" }
        button.title = " \(arrow)"
    }

    // ==================================================================
    // MARK: - Menu
    // ==================================================================

    private func buildMenu() {
        let menu = NSMenu()

        // Section 1: live stats (disabled = non-clickable grey text).
        yawItem     = menu.addItem(withTitle: "Yaw:          --",
                                   action: nil, keyEquivalent: "")
        balanceItem = menu.addItem(withTitle: "Balance:   --",
                                   action: nil, keyEquivalent: "")

        menu.addItem(.separator())

        // Section 2: device info.
        versionItem = menu.addItem(
            withTitle: "Version:      \(kVersion)", action: nil, keyEquivalent: "")
        let (name, type) = deviceInfo()
        deviceNameItem = menu.addItem(
            withTitle: "Device name:  \(name)", action: nil, keyEquivalent: "")
        deviceTypeItem = menu.addItem(
            withTitle: "Device type:  \(type)", action: nil, keyEquivalent: "")

        menu.addItem(.separator())

        // Section 3: toggle.
        toggleItem = menu.addItem(
            withTitle: "Turn on/off",
            action: #selector(onToggle), keyEquivalent: "")
        toggleItem.target = self

        // Recenter — set "straight ahead" to the current head position.
        let recenterItem = menu.addItem(
            withTitle: "Recenter (look straight first)",
            action: #selector(onRecenter), keyEquivalent: "r")
        recenterItem.target = self

        menu.addItem(.separator())

        // Section 4: open Sound Settings.
        let soundItem = menu.addItem(
            withTitle: "Open Sound Settings",
            action: #selector(onOpenSoundSettings), keyEquivalent: "")
        soundItem.target = self

        menu.addItem(.separator())

        // Section 5: report issue.
        let reportItem = menu.addItem(
            withTitle: "Report Issue",
            action: #selector(onReportIssue), keyEquivalent: "")
        reportItem.target = self

        menu.addItem(.separator())

        // Section 6: quit.
        let quitItem = menu.addItem(
            withTitle: "\u{23FB}  Quit",
            action: #selector(onQuit), keyEquivalent: "q")
        quitItem.target = self

        statusItem.menu = menu
    }

    // ==================================================================
    // MARK: - Toggle tracking
    // ==================================================================

    @objc private func onToggle() {
        if trackingOn { disableTracking() } else { enableTracking() }
    }

    /// Set the current head position as "straight ahead" (instant neutral).
    @objc private func onRecenter() {
        guard trackingOn else { return }
        yawCenter     = yawSmoother.value
        yawCenterInit = true
        calibrating   = false
        balanceSmoother.reset()
    }

    private func enableTracking() {
        trackingOn = true
        toggleItem?.state = .on
        updateStatusBar()
        UserDefaults.standard.set(true, forKey: "trackingEnabled")

        // Reset smoothers.
        yawSmoother.reset()
        balanceSmoother.reset()
        lastYaw        = nil
        yawCenter      = 0
        yawCenterInit  = false
        calibrating    = true
        calibSamples   = []
        calibStart     = ProcessInfo.processInfo.systemUptime
        noFaceSince    = nil
        currentBalance = 0
        faceWasDetected = false
        faceLostSince   = nil

        // Audio balance controller.
        if let devID = try? AudioBalanceController.defaultOutputDevice() {
            balanceCtrl = try? AudioBalanceController(deviceID: devID)
        }

        // Head tracker (camera + Vision).
        let tracker = HeadTracker()
        tracker.delegate = self
        tracker.start()
        headTracker = tracker

        // Display refresh timer (10 Hz, same as Python version).
        // Install in `.common` mode so it keeps firing while the menu is open
        // (the default mode is suspended during menu/event tracking).
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.refreshDisplay()
        }
        RunLoop.main.add(timer, forMode: .common)
        displayTimer = timer
    }

    private func disableTracking() {
        trackingOn = false
        toggleItem?.state = .off
        updateStatusBar()
        UserDefaults.standard.set(false, forKey: "trackingEnabled")

        headTracker?.stop();  headTracker = nil
        displayTimer?.invalidate(); displayTimer = nil

        balanceCtrl?.restore(); balanceCtrl = nil

        yawItem?.title     = "Yaw:          --"
        balanceItem?.title = "Balance:   --"
    }

    // ==================================================================
    // MARK: - HeadTrackerDelegate
    // ==================================================================

    func headTracker(_ tracker: HeadTracker, didUpdate yaw: Double, face: Bool) {
        guard trackingOn else { return }

        let target: Double
        if face {
            let smooth = yawSmoother.update(yaw)

            if calibrating {
                // First ~1s after enabling: assume the user is facing the
                // screen and average the readings to establish the neutral
                // yaw.  This removes a constant camera/posture bias of ANY
                // size (the near-centre drift below can only correct small
                // residuals).  Hold balance centred while calibrating.
                calibSamples.append(smooth)
                if ProcessInfo.processInfo.systemUptime - calibStart >= 1.0,
                   calibSamples.count >= 5 {
                    yawCenter     = calibSamples.reduce(0, +) / Double(calibSamples.count)
                    yawCenterInit = true
                    calibrating   = false
                }
                lastYaw     = 0
                noFaceSince = nil
                target      = 0
            } else {
                // Slow near-centre drift to track small residual bias.
                if !yawCenterInit {
                    yawCenter     = smooth
                    yawCenterInit = true
                } else if abs(smooth - yawCenter) < kDeadZoneDeg {
                    yawCenter += 0.02 * (smooth - yawCenter)
                }

                let centered = smooth - yawCenter
                lastYaw     = centered
                noFaceSince = nil
                target      = yawToBalance(centered)
            }
        } else {
            if noFaceSince == nil {
                noFaceSince = ProcessInfo.processInfo.systemUptime
            }
            let elapsed = ProcessInfo.processInfo.systemUptime - (noFaceSince ?? 0)
            let fade    = max(0, 1.0 - elapsed)
            target      = lastYaw != nil ? yawToBalance(lastYaw!) * fade : 0
        }

        let balance = balanceSmoother.update(target)
        // Negate: head turns left → sound shifts right (toward the screen).
        currentBalance = -balance
        balanceCtrl?.setBalance(currentBalance)

        handleFaceLost(face: face)
    }

    func headTracker(_ tracker: HeadTracker, didFail error: HeadTracker.TrackerError) {
        disableTracking()

        let alert = NSAlert()
        alert.messageText = "Camera Error"
        switch error {
        case .cameraUnavailable:
            alert.informativeText = "Could not open the webcam."
        case .permissionDenied:
            alert.informativeText =
                "Camera access is required. Please grant access in " +
                "System Settings → Privacy → Camera."
        }
        alert.runModal()
    }

    // ==================================================================
    // MARK: - Display refresh (10 Hz timer)
    // ==================================================================

    private func refreshDisplay() {
        guard trackingOn else { return }

        // Watchdog: if the camera has stalled (no frames for >2s), the yaw
        // would otherwise get "stuck" at its last value until the user toggles
        // off/on.  Rebuild the capture session automatically instead.
        if let ht = headTracker, ht.secondsSinceLastFrame > 2.0 {
            ht.restart()
        }

        // Check for device hot-swap.
        if let newID = try? AudioBalanceController.defaultOutputDevice(),
           newID != balanceCtrl?.deviceID {
            balanceCtrl?.restore()
            balanceCtrl = try? AudioBalanceController(deviceID: newID)
            balanceSmoother.reset()
            let (name, type) = deviceInfo()
            deviceNameItem?.title = "Device name:  \(name)"
            deviceTypeItem?.title = "Device type:  \(type)"
        }

        // Yaw label (calibrated: ~0° at rest, symmetric at the extremes).
        if let y = lastYaw {
            yawItem?.title = String(format: "Yaw:       %+6.1f°", y)
        } else {
            yawItem?.title = "Yaw:          --"
        }

        // Balance label.
        balanceItem?.title = String(format: "Balance:  %+5.2f", currentBalance)

        // Menu bar icon (compact, never fixed-width — see buildStatusItem).
        updateStatusBar()
    }

    // ==================================================================
    // MARK: - Face-lost notification
    // ==================================================================

    private func handleFaceLost(face: Bool) {
        let now = ProcessInfo.processInfo.systemUptime

        if face {
            faceWasDetected = true
            faceLostSince   = nil
            return
        }

        if faceWasDetected, faceLostSince == nil {
            faceLostSince = now
        }

        guard let lost = faceLostSince,
              (now - lost) > 1.5,
              (now - lastFaceLostNote) > 30.0
        else { return }

        lastFaceLostNote = now
        faceWasDetected  = false

        let content = UNMutableNotificationContent()
        content.title = "Sound Head Tracking"
        content.body  = "Face not detected — balance reset to centre"
        let req = UNNotificationRequest(identifier: "face-lost",
                                        content: content, trigger: nil)
        UNUserNotificationCenter.current().add(req)
    }

    // ==================================================================
    // MARK: - Device-change listener
    // ==================================================================

    private func listenForDeviceChanges() {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope:    kAudioObjectPropertyScopeGlobal,
            mElement:  kAudioObjectPropertyElementMain)

        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &addr, .main
        ) { [weak self] _, _ in
            guard let self else { return }
            let (name, type) = self.deviceInfo()
            self.deviceNameItem?.title = "Device name:  \(name)"
            self.deviceTypeItem?.title = "Device type:  \(type)"
        }
    }

    // ==================================================================
    // MARK: - Utilities
    // ==================================================================

    private func deviceInfo() -> (name: String, type: String) {
        guard let id = try? AudioBalanceController.defaultOutputDevice() else {
            return ("Unknown", "Unknown")
        }
        return (AudioBalanceController.deviceName(id) ?? "Unknown",
                AudioBalanceController.deviceType(id))
    }

    @objc private func onOpenSoundSettings() {
        if let url = URL(string: kSoundSettingsURL) {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func onReportIssue() {
        if let url = URL(string: kIssuesURL) {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func onQuit() {
        disableTracking()
        NSApp.terminate(nil)
    }
}
