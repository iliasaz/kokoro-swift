//
//  MLXSTFT.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//

import Foundation
import MLX
import MLXNN
import ComplexModule

/// A Short-Time Fourier Transform (STFT) module for audio processing.
///
/// This module provides methods for computing the STFT and inverse STFT (iSTFT),
/// allowing spectral analysis and waveform reconstruction from magnitude and phase components.
///
/// ### **Features:**
/// - Converts waveforms into **spectral representations** via STFT.
/// - Reconstructs audio signals from spectrograms using inverse STFT.
/// - Supports **Hann windowing** and configurable FFT parameters.
///
/// ### **Inputs:**
/// - `inputData`: Audio waveform `(batch, samples)`.
///
/// ### **Outputs:**
/// - **STFT Magnitude & Phase**: `(batch, freq_bins, time_steps)`.
/// - **Reconstructed waveform** from inverse STFT.
///
/// ### **Example Usage:**
/// ```swift
/// let stft = MLXSTFT(filterLength: 800, hopLength: 200, winLength: 800)
/// let (magnitude, phase) = stft.transform(audioData)
/// let reconstructedAudio = stft.inverse(magnitude: magnitude, phase: phase)
/// ```
class MLXSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int
    let window: String

    init(filterLength: Int = 800, hopLength: Int = 200, winLength: Int = 800, window: String = "hann") {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength
        self.window = window
    }

    /// Computes the Short-Time Fourier Transform (STFT) of an input waveform.
    ///
    /// - Parameters:
    ///   - x: Input waveform `(samples)`.
    ///   - nFFT: Number of FFT bins (default: `800`).
    ///   - hopLength: Hop size between frames (default: `nFFT / 4`).
    ///   - winLength: Window size (default: `nFFT`).
    ///   - window: Type of window function (`"hann"` supported).
    ///   - center: If `true`, pads signal before STFT.
    ///   - padMode: Padding mode (`"reflect"` or `"constant"`).
    /// - Returns: STFT spectrogram `(freq_bins, time_steps)`.
    func mlxSTFT(
        x: MLXArray,
        nFFT: Int = 800,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: String = "hann",
        center: Bool = true,
        padMode: String = "reflect"
    ) -> MLXArray {
        let hop = hopLength ?? (nFFT / 4)
        let win = winLength ?? nFFT

        var w: MLXArray
        if window.lowercased() == "hann" {
            w = npHanning(win)
        } else {
            fatalError("Only 'hann' window is supported.")
        }

        // Ensure window length matches nFFT
        if w.shape[0] < nFFT {
            let padSize = nFFT - w.shape[0]
            w = concatenated([w, MLX.zeros([padSize])], axis: 0)
        }

        var paddedX = x
        if center {
            paddedX = padSignal(x: x, padding: nFFT / 2, padMode: padMode)
        }

        let numFrames = 1 + (paddedX.shape[0] - nFFT) / hop
        guard numFrames > 0 else {
            fatalError("Input too short for nFFT=\(nFFT) with hopLength=\(hop) and center=\(center).")
        }

        // Create STFT frames
        let shape = [numFrames, nFFT]
        let strides = [hop, 1]
        let frames = MLX.asStrided(paddedX, shape, strides: strides)

        // Perform FFT and return magnitude & phase
        return rfft(frames * w).transposed(axes: [1, 0])
    }

    /// Computes the inverse Short-Time Fourier Transform (iSTFT) to reconstruct a waveform.
    ///
    /// - Parameters:
    ///   - x: STFT spectrogram `(freq_bins, time_steps)`.
    ///   - hopLength: Hop size between frames (default: `winLength / 4`).
    ///   - winLength: Window size (default: `(freq_bins - 1) * 2`).
    ///   - window: Type of window function (`"hann"` supported).
    ///   - center: If `true`, pads signal before inverse STFT.
    ///   - length: Desired length of the output signal.
    /// - Returns: Reconstructed waveform `(samples)`.
    func mlxISTFT(
        x: MLXArray,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: String = "hann",
        center: Bool = true,
        length: Int? = nil
    ) -> MLXArray {
        let hop = hopLength ?? (winLength! / 4)
        let win = winLength ?? ((x.shape[1] - 1) * 2)

        var w: MLXArray
        if window.lowercased() == "hann" {
            w = npHanning(win) // Hann window function
        } else {
            fatalError("Only 'hann' window is supported.")
        }

        if w.shape[0] < win {
            let padSize = win - w.shape[0]
            w = concatenated([w, MLX.zeros([padSize])], axis: 0)
        }

        let xT = x.transposed(axes: [1, 0])
        let totalSamples = (xT.shape[0] - 1) * hop + win
        var reconstructed = MLX.zeros([totalSamples])
        var windowSum = MLX.zeros([totalSamples])

        for i in 0..<xT.shape[0] {
            let frameTime = irfft(xT[i])
            let start = i * hop
            let end = start + win

            reconstructed[start..<end] += frameTime * w
            windowSum[start..<end] += w * w
        }

        reconstructed = MLX.where(windowSum .!= 0, reconstructed / windowSum, reconstructed)

        if center && length == nil {
            reconstructed = reconstructed[(win / 2)..<(reconstructed.shape[0] - win / 2)]
        }

        if let length = length {
            reconstructed = reconstructed[0..<length]
        }

        return reconstructed
    }

    /// Pads the input waveform using reflection or constant padding.
    ///
    /// - Parameters:
    ///   - x: Input waveform `(samples)`.
    ///   - padding: Number of samples to pad.
    ///   - padMode: Padding mode (`"reflect"` or `"constant"`).
    /// - Returns: Padded waveform `(samples + 2 * padding)`.
    private func padSignal(x: MLXArray, padding: Int, padMode: String) -> MLXArray {
        switch padMode.lowercased() {
            case "constant":
                return padded(x, widths: [IntOrPair((padding, padding))])
            case "reflect":
                let prefix = x[1 ..< (padding + 1)][.stride(by: -1)]
                let suffix = x[-(padding + 1) ..< -1][.stride(by: -1)]
                return concatenated([prefix, x, suffix])
            default:
                fatalError("Invalid pad mode: \(padMode)")
        }
    }


    /// Generates a Hann (Hanning) window of a given size.
    ///
    /// - Parameter size: The number of points in the window.
    /// - Returns: An `MLXArray` containing the Hann window values.
    private func npHanning(_ size: Int) -> MLXArray {
        let window = (0..<size).map { 0.5 * (1 - cos(2 * .pi * Double($0) / Double(size - 1))) }
        return MLXArray(converting: window)
    }

    /// Computes the Short-Time Fourier Transform (STFT) of an input waveform.
    ///
    /// - Parameter inputData: Input audio waveform `(batch, samples)`.
    /// - Returns: A tuple containing:
    ///   - `magnitude`: STFT magnitude `(batch, freq_bins, time_steps)`.
    ///   - `phase`: STFT phase `(batch, freq_bins, time_steps)`.
    func transform(_ inputData: MLXArray) -> (MLXArray, MLXArray) {
        var audioData = inputData
        if audioData.ndim == 1 {
            audioData = audioData.expandedDimensions(axis: 0) // Ensure batch dimension
        }

        var magnitudes: [MLXArray] = []
        var phases: [MLXArray] = []

        for batchIdx in 0..<audioData.shape[0] {
            let audioSlice = audioData[batchIdx]

            // Compute STFT using MLX API
            let stftResult = mlxSTFT(
                x: audioSlice,
                nFFT: filterLength,
                hopLength: hopLength,
                winLength: winLength,
                window: window
            )

            // Compute magnitude and phase
            let magnitude = MLX.abs(stftResult)
//            let phase = MLX.atan(stftResult)
            let phase = MLX.atan2(stftResult.imaginaryPart(), stftResult.realPart())

            magnitudes.append(magnitude)
            phases.append(phase)
        }

        return (MLX.stacked(magnitudes, axis: 0), MLX.stacked(phases, axis: 0))
    }

    /// Computes the inverse Short-Time Fourier Transform (iSTFT) to reconstruct a waveform.
    ///
    /// - Parameters:
    ///   - magnitude: STFT magnitude `(batch, freq_bins, time_steps)`.
    ///   - phase: STFT phase `(batch, freq_bins, time_steps)`.
    /// - Returns: Reconstructed waveform `(batch, samples)`.
    func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var reconstructedAudio: [MLXArray] = []

        for batchIdx in 0..<magnitude.shape[0] {
            let magnitudeSlice = magnitude[batchIdx]
            let phaseSlice = phase[batchIdx]

            // Unwrap phase for reconstruction
            let phaseContinuous = unwrap(phaseSlice, axis: 1) // Equivalent of `np.unwrap()`
            // Reconstruct complex STFT
            let stftComplex = magnitudeSlice * MLX.exp(Complex(imaginary: 1) * phaseContinuous)

            // Compute inverse STFT
            let audio = mlxISTFT(
                x: stftComplex,
                hopLength: hopLength,
                winLength: winLength,
                window: window
            )

            reconstructedAudio.append(audio)
        }

        return MLX.stacked(reconstructedAudio, axis: 0).expandedDimensions(axis: -2)
    }

    /// Unwraps a signal by correcting phase jumps larger than a given discontinuity threshold.
    ///
    /// This function mimics NumPy’s unwrap by taking the period-complement of large phase jumps.
    /// It adjusts adjacent differences that exceed `max(discont, period/2)` by adding or subtracting
    /// multiples of the period.
    ///
    /// - Parameters:
    ///   - p: Input MLX array of type Double.
    ///   - discont: Maximum discontinuity between values. If nil, defaults to period/2.
    ///   - axis: Axis along which to unwrap (default is the last axis).
    ///   - period: The period over which the input wraps (default is 2π).
    /// - Returns: An MLX array with the unwrapped signal.
    private func unwrap(_ phase: MLXArray,
                        discont: Float? = nil,
                        axis: Int = -1,
                        period: Float = 2 * Float.pi) -> MLXArray {
        // Compute differences along the specified axis using our custom diff.
        let dd = phase.diff(axis: axis)
        // Set the effective discontinuity (default to period/2 if nil).
        let effectiveDiscont = discont ?? (period / 2)

        // Define the bounds for the interval.
        let intervalHigh = period / 2
        let intervalLow  = -intervalHigh

        // Map the differences into the interval [intervalLow, intervalHigh].
        var ddmod = ((dd - intervalLow) % period) + intervalLow

        // Correct boundary cases:
        // For elements where ddmod equals intervalLow and the original phaseDiff is positive,
        // set ddmod to intervalHigh.
        ddmod = which( ((ddmod .== intervalLow) .&& (dd .> 0.0)), intervalHigh, ddmod)

        // Compute the phase correction as the difference between the modded difference and the original.
        var phCorrect = ddmod - dd
        phCorrect = which( (abs(dd) .< effectiveDiscont), 0.0, phCorrect )
        var up = MLXArray.init(data: phase.asData())
        up[1..., axis: axis] = phase[1..., axis: axis] + phCorrect.cumsum(axis: axis)
        return up
    }

    /// Performs STFT followed by inverse STFT for validation.
    ///
    /// - Parameter inputData: Input waveform `(batch, samples)`.
    /// - Returns: Reconstructed waveform `(batch, samples)`.
    func callAsFunction(_ inputData: MLXArray) -> MLXArray {
        let (magnitude, phase) = transform(inputData)
        return inverse(magnitude: magnitude, phase: phase)
    }
}

extension MLXArray {
    func diff(axis: Int) -> MLXArray {
        let axisDim = self.shape[axis]
        // Create slices for indices 1..<axisDim and 0..<(axisDim-1).
        let sliceStart = self[1..<axisDim, axis: axis]
        let sliceEnd   = self[0..<(axisDim - 1), axis: axis]
        return sliceStart - sliceEnd
    }
}
