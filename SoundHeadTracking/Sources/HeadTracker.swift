// HeadTracker.swift — Webcam capture + Apple Vision face-landmark yaw.
//
// Yaw is computed GEOMETRICALLY from the eyes + nose, the same class of
// approach that works in the Python MediaPipe script (Vision has no ear
// landmarks, so we use eyes — which also never get occluded when turning).
//
// Key properties:
//   • Roll-invariant: the nose offset is projected onto the eye-line axis,
//     so tilting the head while facing the screen does NOT change the yaw.
//   • Quality-gated: low-quality frames are skipped (value held) instead of
//     collapsing to ~0.
//   • Frame-throttled + reduced camera fps to keep CPU low.
//   • Self-healing: a stalled capture session is rebuilt automatically.

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
    private let kYawSign: Double = -1.0

    /// Frames per second to *process* with Vision (camera is also capped here).
    private let targetFPS: Double = 12.0

    weak var delegate: HeadTrackerDelegate?

    private var session: AVCaptureSession?
    private let queue  = DispatchQueue(label: "com.sharmayank.headtracker",
                                       qos: .userInteractive)
    private var running = false

    // Vision request (configured once, reused on the serial queue).
    private let request: VNDetectFaceLandmarksRequest = {
        let r = VNDetectFaceLandmarksRequest()
        r.revision = VNDetectFaceLandmarksRequestRevision3
        r.constellation = .constellation76Points   // includes pupils
        return r
    }()

    // Vision-processing throttle (capture-queue only).
    private var lastProcess: TimeInterval = 0
    private var minInterval: TimeInterval { 1.0 / 13.0 }   // slightly > targetFPS

    // Watchdog: monotonic timestamp of the last delivered camera frame.
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

        // Cap camera frame rate (best effort) to cut capture + Vision cost.
        if let range = camera.activeFormat.videoSupportedFrameRateRanges.first,
           Double(range.minFrameRate) <= targetFPS,
           targetFPS <= Double(range.maxFrameRate),
           (try? camera.lockForConfiguration()) != nil {
            let d = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
            camera.activeVideoMinFrameDuration = d
            camera.activeVideoMaxFrameDuration = d
            camera.unlockForConfiguration()
        }

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

        markFrame()   // camera liveness (watchdog) — every received frame

        // Throttle expensive Vision work to ~targetFPS.
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastProcess >= minInterval else { return }
        lastProcess = now

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        try? handler.perform([request])

        // Genuinely no face → tell the app (it fades to centre).
        guard let face = request.results?.first else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.delegate?.headTracker(self, didUpdate: 0, face: false)
            }
            return
        }

        // Face present but low-quality landmarks → SKIP (hold last value)
        // rather than collapse the yaw toward zero.
        guard let yaw = Self.geometricYaw(face, sign: kYawSign) else { return }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.headTracker(self, didUpdate: yaw, face: true)
        }
    }

    /// Roll-invariant geometric head yaw (degrees) from eye + nose landmarks.
    /// Positive (× `sign`) = user turned to THEIR right.  Returns `nil` for
    /// unusable frames (missing/collapsed landmarks) so the caller can hold.
    static func geometricYaw(_ face: VNFaceObservation, sign: Double) -> Double? {
        guard let lm = face.landmarks else { return nil }

        // Use a CONSISTENT eye-centre source for both eyes — mixing a pupil
        // point on one eye with an eye-outline centroid on the other injects a
        // fake horizontal offset (spurious yaw).  Prefer both pupils; else
        // fall back to both eye outlines.
        let leftEye: CGPoint
        let rightEye: CGPoint
        if let lp = lm.leftPupil, let rp = lm.rightPupil,
           !lp.normalizedPoints.isEmpty, !rp.normalizedPoints.isEmpty {
            leftEye = centroid(lp); rightEye = centroid(rp)
        } else if let le = lm.leftEye, let re = lm.rightEye,
                  !le.normalizedPoints.isEmpty, !re.normalizedPoints.isEmpty {
            leftEye = centroid(le); rightEye = centroid(re)
        } else {
            return nil
        }

        guard let noseR = lm.nose, !noseR.normalizedPoints.isEmpty
        else { return nil }
        let nose = centroid(noseR)
        guard leftEye.x.isFinite, rightEye.x.isFinite, nose.x.isFinite
        else { return nil }

        // Eye-line vector (rotates with head tilt).
        let ex = rightEye.x - leftEye.x
        let ey = rightEye.y - leftEye.y
        let span = (ex * ex + ey * ey).squareRoot()
        guard span > 0.05 else { return nil }   // eyes collapsed → unreliable

        // Project the nose offset onto the eye-line axis → roll-invariant.
        let axisX = ex / span
        let axisY = ey / span
        let eyeMidX = (leftEye.x + rightEye.x) / 2
        let eyeMidY = (leftEye.y + rightEye.y) / 2
        let along = (nose.x - eyeMidX) * axisX + (nose.y - eyeMidY) * axisY

        let ratio   = Double(along) / Double(span / 2)
        let clamped = max(-2.0, min(2.0, ratio))
        return sign * clamped * kRatioToDeg
    }

    // MARK: Landmark helpers

    private static func centroid(_ region: VNFaceLandmarkRegion2D) -> CGPoint {
        let pts = region.normalizedPoints
        guard !pts.isEmpty else { return CGPoint(x: CGFloat.nan, y: CGFloat.nan) }
        var sx: CGFloat = 0, sy: CGFloat = 0
        for p in pts { sx += p.x; sy += p.y }
        return CGPoint(x: sx / CGFloat(pts.count), y: sy / CGFloat(pts.count))
    }
}
