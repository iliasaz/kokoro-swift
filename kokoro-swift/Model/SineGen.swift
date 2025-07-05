//
//  SineGen.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A sine wave generator for synthesizing fundamental and harmonic tones based on an input F0 signal.
///
/// This module generates sine waves corresponding to the fundamental frequency (F0) and its harmonics,
/// optionally adding noise based on voiced/unvoiced detection. The implementation follows:
/// - **Fundamental and Harmonics Generation**: Computes harmonics up to `harmonicNum`.
/// - **Sine Wave Computation**: Generates phase-continuous sine waves.
/// - **Noise Addition**: Adds noise to unvoiced segments based on a threshold.
///
/// This implementation follows the approach used in neural vocoders for speech synthesis.

class SineGen: Module {
    let sineAmp: Float
    let noiseStd: Float
    let harmonicNum: Int
    let dim: Int
    let samplingRate: Int
    let voicedThreshold: Float
    let flagForPulse: Bool
    let upsampleScale: Float
    var radInterpolate: Upsample
    var phaseInterpolate: Upsample

    init(
        sampRate: Int,
        upsampleScale: Float,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 0,
        flagForPulse: Bool = false
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.harmonicNum = harmonicNum
        self.dim = self.harmonicNum + 1
        self.samplingRate = sampRate
        self.voicedThreshold = voicedThreshold
        self.flagForPulse = flagForPulse
        self.upsampleScale = upsampleScale
        self.radInterpolate = Upsample(scaleFactor: .float(1.0 / upsampleScale), mode: .linear(alignCorners: false))
        self.phaseInterpolate = Upsample(scaleFactor: .float(upsampleScale), mode: .linear(alignCorners: false))
    }

    /// Converts F0 values to a voiced/unvoiced (UV) binary mask.
    ///
    /// - Parameter f0: Input F0 tensor of shape `(batch, length)`.
    /// - Returns: A binary tensor where 1 indicates voiced and 0 indicates unvoiced.
    func f02uv(_ f0: MLXArray) -> MLXArray {
        return MLX.where(f0 .> voicedThreshold, 1.0, 0.0) // Creates a binary mask
    }

    /// Converts F0 values to sine waves.
    ///
    /// - Parameter f0Values: Input F0 tensor of shape `(batch, length, dim)`.
    /// - Returns: A tensor containing sine waveforms.
    func f02sine(_ f0Values: MLXArray) -> MLXArray {
        var radValues = (f0Values / Float(samplingRate)) % 1

        // Generate random phase noise (no noise for fundamental component)
        let randIni = normal([f0Values.shape[0], f0Values.shape[2]])
        // Only zero out the first harmonic component
        // This preserves the random noise for higher harmonics while ensuring the fundamental frequency remains phase-aligned
        randIni[0..., 0] = MLX.zeros([randIni.shape[0]])
        radValues[0 ..< radValues.shape[0], 0, 0 ..< radValues.shape[2]] = radValues[0 ..< radValues.shape[0], 0, 0 ..< radValues.shape[2]] + randIni

        if !flagForPulse {
            radValues = interpolate(
              input: radValues.transposed(0, 2, 1),
              scaleFactor: [1 / Float(upsampleScale)],
              mode: "linear"
            ).transposed(0, 2, 1)

            var phase = MLX.cumsum(radValues, axis: 1) * 2 * Float.pi

            phase = interpolate(
              input: phase.transposed(0, 2, 1) * Float(upsampleScale),
              scaleFactor: [Float(upsampleScale)],
              mode: "linear"
            ).transposed(0, 2, 1)

            return MLX.sin(phase)

        } else {
            // Pulse-train generation: ensure phase continuity within voiced segments
            let uv = f02uv(f0Values)
            let uv1 = roll(uv, shift: -1, axes: [1])
            uv1[0..., -1, 0...] = MLX.ones([uv1.shape[0], uv1.shape[2]]) // Assigns 1.0 to the last time step for all batches
            let uLoc = (uv .< 1) * (uv1 .> 0)

            let tmpCumsum = MLX.cumsum(radValues, axis: 1)

            for idx in 0..<f0Values.shape[0] {
                let tempSum = tmpCumsum[idx, uLoc[idx, 0..., 0...], 0...]
                tempSum[1..., 0...] = tempSum[1..., 0...] - tempSum[0..<(tempSum.shape[0] - 1), 0...]
                tmpCumsum[idx, 0..., 0...] = MLX.zeros(tmpCumsum[idx, 0..., 0...].shape)
                tmpCumsum[idx, uLoc[idx, 0..., 0...], 0...] = tempSum
            }

            let iPhase = MLX.cumsum(radValues - tmpCumsum, axis: 1)
            return MLX.cos(iPhase * Float(2.0 * .pi))
        }
    }

    /// Generates sine waves from the input F0 sequence.
    ///
    /// - Parameter f0: Input F0 tensor of shape `(batch, length)`.
    /// - Returns: A tuple containing sine waveforms, voiced/unvoiced mask, and noise.
    func callAsFunction(f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // Fundamental frequency components with harmonics
        let fn = f0 * MLXArray(1...(harmonicNum + 1)).asType(.float32).reshaped([1, 1, harmonicNum + 1])
        // Generate sine waveforms
        let sineWaves = f02sine(fn) * sineAmp

        // Generate UV signal
        let uv = f02uv(f0)

        // Generate noise
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3
        let noise = noiseAmp * normal(sineWaves.shape)

        // Apply UV mask to sine waves and add noise
        let outputWaves = sineWaves * uv + noise
        return (outputWaves, uv, noise)
    }
}

