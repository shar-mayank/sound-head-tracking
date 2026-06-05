// HeadTracker.swift — Webcam capture + Apple Vision face-yaw detection.
//
// Replaces the Python MediaPipe + OpenCV pipeline with native macOS
// frameworks: AVFoundation for the camera, Vision for face detection.
// VNFaceObservation.yaw gives the head yaw angle directly.

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

    weak var delegate: HeadTrackerDelegate?

    private var session: AVCaptureSession?
    private let output = AVCaptureVideoDataOutput()
    private let queue  = DispatchQueue(label: "com.sharmayank.headtracker",
                                       qos: .userInteractive)
    private var running = false

    // MARK: Start / Stop

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
        queue.async { [weak self] in
            self?.session?.stopRunning()
            self?.session = nil
        }
    }

    // MARK: Private — session setup

    private func setupAndRun() {
        let sess = AVCaptureSession()
        sess.sessionPreset = .medium        // ~480×360, enough for face detection

        // Find the camera (built-in or any available).
        guard let camera = AVCaptureDevice.default(
                    .builtInWideAngleCamera, for: .video, position: .unspecified)
                ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera),
              sess.canAddInput(input)
        else { fail(.cameraUnavailable); return }

        sess.addInput(input)

        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard sess.canAddOutput(output)
        else { fail(.cameraUnavailable); return }
        sess.addOutput(output)

        // Mirror the image (same as Python cv2.flip) so left/right match
        // the user's perspective.
        if let conn = output.connection(with: .video), conn.isVideoMirroringSupported {
            conn.isVideoMirrored = true
        }

        session = sess
        running = true

        queue.async { sess.startRunning() }
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

        // Use Vision to detect face rectangles (includes yaw property).
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        try? handler.perform([request])

        guard let results = request.results,
              let face = results.first,
              let yawNumber = face.yaw
        else {
            // No face detected.
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.delegate?.headTracker(self, didUpdate: 0, face: false)
            }
            return
        }

        // VNFaceObservation.yaw is in radians.
        // Positive = face turned to camera's left = user turned RIGHT.
        // This matches the Python convention (after its cv2.flip).
        let yawDeg = yawNumber.doubleValue * (180.0 / .pi)

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.headTracker(self, didUpdate: yawDeg, face: true)
        }
    }
}
