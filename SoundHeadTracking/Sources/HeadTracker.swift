// HeadTracker.swift — Webcam capture + Apple Vision face-landmark yaw.
//
// Replaces the Python MediaPipe + OpenCV pipeline with native macOS
// frameworks: AVFoundation for the camera, Vision for face landmarks.
//
// Yaw is computed GEOMETRICALLY from the eyes + nose (not Vision's coarse
// built-in `.yaw`, and not the ears).  Eyes stay visible across the whole
// comfortable turn range, so the signal is symmetric left/right — unlike
// ear-based geometry, where the far ear is occluded and its guessed
// position makes one direction collapse to near-zero.

import AVFoundation
import CoreMedia
import Vision

// MARK: - Delegate protocol

protocol HeadTrackerDelegate: AnyObject {
    /// Called on the **main thread** after each processed frame.
    func headTracker(_ tracker: HeadTracker, didUpdate yaw: Double, face: Bool)
    /// Called on the **main thread** when a fatal error occurs.
    func headTracker(_ tracker: HeadTracker, didFail error: HeadTracker.TrackerError)
}

// MARK: - HeadTracker

final class HeadTracker: NSObject {

    enum TrackerError {
        case cameraUnavailable
        case permissionDenied
    }

    /// `+1` if turning the head right should yield positive yaw; flip to `-1`
    /// if the left/right sense is ever inverted on a particular machine.
    private let kYawSign: Double = 1.0

    weak var delegate: HeadTrackerDelegate?

    private var session: AVCaptureSession?
    private let queue  = DispatchQueue(label: "com.sharmayank.headtracker",
                                       qos: .userInteractive)
    private var running = false

    // Reusable Vision request (serial queue → safe to share).
    private let request = VNDetectFaceLandmarksRequest()

    // Watchdog: monotonic timestamp of the last delivered frame.
    private let stateLock = NSLock()
    private var lastFrameStamp: TimeInterval = 0

    /// Seconds since the last camera frame arrived (0 if none yet).
    var secondsSinceLastFrame: TimeInterval {
        stateLock.lock(); defer { stateLock.unlock() }
        return lastFrameStamp == 0
            ? 0
            : ProcessInfo.processInfo.systemUptime - lastFrameStamp
    }

    private func markFrame() {
        stateLock.lock()
        lastFrameStamp = ProcessInfo.processInfo.systemUptime
        stateLock.unlock()
    }

    // MARK: Start / Stop / Restart

    func start() {
        guard !running else { return }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupAndRun()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
                guard let self else { return }
                if ok { self.setupAndRun() }
                else  { self.fail(.permissionDenied) }
            }
        default:
            fail(.permissionDenied)
        }
    }

    func stop() {
        running = false
        NotificationCenter.default.removeObserver(self)
        queue.async { [weak self] in
            self?.session?.stopRunning()
            self?.session = nil
        }
    }

    /// Tear down and rebuild the capture session.  Called by the app's
    /// watchdog when frame delivery stalls, or on a capture runtime error —
    /// this is what recovers the "yaw stuck until toggle off/on" state.
    func restart() {
        guard running else { return }
        markFrame()   // reset watchdog so it doesn't re-fire during rebuild
        queue.async { [weak self] in
            guard let self, self.running else { return }
            self.session?.stopRunning()
            self.session = nil
            self.setupSession()
        }
    }

    // MARK: Private — session setup

    private func setupAndRun() {
        queue.async { [weak self] in self?.setupSession() }
    }

    /// Builds and starts the capture session.  Always runs on `queue`.
    private func setupSession() {
        let sess = AVCaptureSession()
        sess.sessionPreset = .vga640x480

        guard let camera =
                AVCaptureDevice.default(.builtInWideAngleCamera,
                                        for: .video, position: .front)
                ?? AVCaptureDevice.default(.builtInWideAngleCamera,
                                           for: .video, position: .unspecified)
                ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera),
              sess.canAddInput(input)
        else { fail(.cameraUnavailable); return }

        sess.addInput(input)

        // Fresh output each time (an output can only belong to one session).
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard sess.canAddOutput(output)
        else { fail(.cameraUnavailable); return }
        sess.addOutput(output)

        // NOTE: the capture connection is intentionally NOT mirrored.  We work
        // in the raw camera coordinate space and account for it in the yaw
        // geometry, avoiding platform-specific `isVideoMirrored` quirks.

        // Self-healing: observe runtime errors / interruptions.
        let nc = NotificationCenter.default
        nc.removeObserver(self)
        nc.addObserver(self, selector: #selector(sessionRuntimeError(_:)),
                       name: .AVCaptureSessionRuntimeError, object: sess)
        nc.addObserver(self, selector: #selector(sessionInterruptionEnded(_:)),
                       name: .AVCaptureSessionInterruptionEnded, object: sess)

        session = sess
        running = true
        markFrame()            // seed watchdog
        sess.startRunning()
    }

    @objc private func sessionRuntimeError(_ note: Notification) {
        restart()
    }

    @objc private func sessionInterruptionEnded(_ note: Notification) {
        queue.async { [weak self] in
            guard let self, self.running,
                  let s = self.session, !s.isRunning else { return }
            s.startRunning()
        }
    }

    private func fail(_ error: TrackerError) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.headTracker(self, didFail: error)
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension HeadTracker: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard running,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }

        markFrame()

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        try? handler.perform([request])

        guard let face = request.results?.first,
              let yaw = Self.geometricYaw(face, sign: kYawSign)
        else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.delegate?.headTracker(self, didUpdate: 0, face: false)
            }
            return
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.headTracker(self, didUpdate: yaw, face: true)
        }
    }

    /// Geometric head yaw (degrees) from eye + nose landmarks.
    ///
    /// The horizontal offset of the nose from the eye midpoint, normalised by
    /// the eye half-span, gives a scale-invariant signal symmetric in both
    /// directions.  Positive (× `sign`) = user turned to THEIR right.
    static func geometricYaw(_ face: VNFaceObservation, sign: Double) -> Double? {
        guard let lm = face.landmarks,
              let leftEye  = lm.leftEye,
              let rightEye = lm.rightEye,
              let nose     = lm.nose
        else { return nil }

        func centroidX(_ region: VNFaceLandmarkRegion2D) -> Double {
            let pts = region.normalizedPoints
            guard !pts.isEmpty else { return .nan }
            return pts.reduce(0.0) { $0 + Double($1.x) } / Double(pts.count)
        }

        let lx = centroidX(leftEye)
        let rx = centroidX(rightEye)
        let nx = centroidX(nose)
        guard lx.isFinite, rx.isFinite, nx.isFinite else { return nil }

        let eyeMid   = (lx + rx) / 2.0
        let halfSpan = abs(rx - lx) / 2.0
        guard halfSpan > 0.001 else { return nil }

        // Raw (un-mirrored) camera: turning right moves the nose toward
        // image-left (smaller x), so (eyeMid − nose) is positive on a right
        // turn.  Normalised by eye half-span → scale-invariant ratio.
        let ratio   = (eyeMid - nx) / halfSpan
        let clamped = max(-2.0, min(2.0, ratio))
        return sign * clamped * kRatioToDeg
    }
}
