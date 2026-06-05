// AudioBalanceController.swift — CoreAudio stereo balance control.
//
// Direct port of the Python _BalanceController from head_track_audio.py.
// Strategy:
//   1. If the device supports kAudioDevicePropertyStereoPan, use it.
//   2. Otherwise fall back to per-channel VolumeScalar on elements 1/2.

import CoreAudio
import Foundation

// MARK: - Errors

enum AudioBalanceError: Error {
    case unsupportedDevice
    case propertyError(OSStatus)
}

// MARK: - AudioBalanceController

final class AudioBalanceController {

    let deviceID: AudioObjectID
    private let useStereoPan: Bool
    private var lastBase: Float?
    private var lastBalance: Float = 0

    var method: String { useStereoPan ? "Stereo Pan" : "Per-Channel Volume" }

    // -----------------------------------------------------------------
    // Init — probe device capabilities
    // -----------------------------------------------------------------

    init(deviceID: AudioObjectID) throws {
        self.deviceID = deviceID

        var panAddr = Self.address(kAudioDevicePropertyStereoPan,
                                   kAudioObjectPropertyScopeOutput)

        if AudioObjectHasProperty(deviceID, &panAddr) {
            useStereoPan = true
        } else {
            var lAddr = Self.address(kAudioDevicePropertyVolumeScalar,
                                     kAudioObjectPropertyScopeOutput, 1)
            var rAddr = Self.address(kAudioDevicePropertyVolumeScalar,
                                     kAudioObjectPropertyScopeOutput, 2)
            guard AudioObjectHasProperty(deviceID, &lAddr),
                  AudioObjectHasProperty(deviceID, &rAddr) else {
                throw AudioBalanceError.unsupportedDevice
            }
            useStereoPan = false
            let l = try Self.getFloat(deviceID, &lAddr)
            let r = try Self.getFloat(deviceID, &rAddr)
            lastBase = max(l, r)
        }
    }

    // -----------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------

    /// Apply `value` (–1 … +1) as stereo balance.
    func setBalance(_ value: Double) {
        let v = Float(max(-1, min(1, value)))

        if useStereoPan {
            var addr = Self.address(kAudioDevicePropertyStereoPan,
                                    kAudioObjectPropertyScopeOutput)
            let pan = (v + 1) / 2          // 0 = left, 0.5 = centre, 1 = right
            try? Self.setFloat(deviceID, &addr, pan)
        } else {
            var lAddr = Self.address(kAudioDevicePropertyVolumeScalar,
                                     kAudioObjectPropertyScopeOutput, 1)
            var rAddr = Self.address(kAudioDevicePropertyVolumeScalar,
                                     kAudioObjectPropertyScopeOutput, 2)
            let curL = (try? Self.getFloat(deviceID, &lAddr)) ?? 0
            let curR = (try? Self.getFloat(deviceID, &rAddr)) ?? 0
            let curMax = max(curL, curR)

            // Detect user volume changes.
            if let base = lastBase, abs(curMax - base) > 0.005 {
                lastBase = curMax
            }
            let base = lastBase ?? curMax

            let leftVol:  Float
            let rightVol: Float
            if v <= 0 {
                leftVol  = base
                rightVol = base * (1 + v)
            } else {
                leftVol  = base * (1 - v)
                rightVol = base
            }

            try? Self.setFloat(deviceID, &lAddr, max(0, min(1, leftVol)))
            try? Self.setFloat(deviceID, &rAddr, max(0, min(1, rightVol)))
            lastBalance = v
        }
    }

    /// Reset balance to centre without changing the user's current volume.
    func restore() {
        if useStereoPan {
            var addr = Self.address(kAudioDevicePropertyStereoPan,
                                    kAudioObjectPropertyScopeOutput)
            try? Self.setFloat(deviceID, &addr, 0.5)
        } else {
            var lAddr = Self.address(kAudioDevicePropertyVolumeScalar,
                                     kAudioObjectPropertyScopeOutput, 1)
            var rAddr = Self.address(kAudioDevicePropertyVolumeScalar,
                                     kAudioObjectPropertyScopeOutput, 2)
            let l = (try? Self.getFloat(deviceID, &lAddr)) ?? 0
            let r = (try? Self.getFloat(deviceID, &rAddr)) ?? 0
            let level = max(l, r)
            try? Self.setFloat(deviceID, &lAddr, level)
            try? Self.setFloat(deviceID, &rAddr, level)
        }
    }

    // -----------------------------------------------------------------
    // Static helpers
    // -----------------------------------------------------------------

    /// Return the AudioObjectID of the default output device.
    static func defaultOutputDevice() throws -> AudioObjectID {
        var devID: AudioObjectID = 0
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var addr = address(kAudioHardwarePropertyDefaultOutputDevice,
                           kAudioObjectPropertyScopeGlobal)
        let st = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &addr, 0, nil, &size, &devID)
        guard st == noErr else { throw AudioBalanceError.propertyError(st) }
        return devID
    }

    /// Human-readable name of an audio device, or `nil`.
    static func deviceName(_ deviceID: AudioObjectID) -> String? {
        var addr = address(kAudioObjectPropertyName,
                           kAudioObjectPropertyScopeGlobal)
        var size: UInt32 = 0
        var st = AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size)
        guard st == noErr, size > 0 else { return nil }

        // The property is a CFStringRef.  Use Unmanaged for safe ARC bridging.
        var rawPtr: Unmanaged<CFString>?
        st = withUnsafeMutablePointer(to: &rawPtr) { ptr in
            ptr.withMemoryRebound(to: UInt8.self, capacity: Int(size)) { buf in
                AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, buf)
            }
        }
        guard st == noErr, let cfString = rawPtr?.takeRetainedValue() else { return nil }
        return cfString as String
    }

    /// Whether the device uses StereoPan or Per-Channel Volume.
    static func deviceType(_ deviceID: AudioObjectID) -> String {
        var addr = address(kAudioDevicePropertyStereoPan,
                           kAudioObjectPropertyScopeOutput)
        return AudioObjectHasProperty(deviceID, &addr)
            ? "Stereo Pan" : "Per-Channel Volume"
    }

    // -----------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------

    private static func address(
        _ selector: AudioObjectPropertySelector,
        _ scope: AudioObjectPropertyScope,
        _ element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
    ) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector,
                                   mScope: scope,
                                   mElement: element)
    }

    private static func getFloat(
        _ id: AudioObjectID,
        _ addr: inout AudioObjectPropertyAddress
    ) throws -> Float {
        var val: Float32 = 0
        var size = UInt32(MemoryLayout<Float32>.size)
        let st = AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &val)
        guard st == noErr else { throw AudioBalanceError.propertyError(st) }
        return val
    }

    private static func setFloat(
        _ id: AudioObjectID,
        _ addr: inout AudioObjectPropertyAddress,
        _ value: Float
    ) throws {
        var val = value
        let st = AudioObjectSetPropertyData(
            id, &addr, 0, nil,
            UInt32(MemoryLayout<Float32>.size), &val)
        guard st == noErr else { throw AudioBalanceError.propertyError(st) }
    }
}
