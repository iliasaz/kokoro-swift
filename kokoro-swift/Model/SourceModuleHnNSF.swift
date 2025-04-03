//
//  SourceModuleHnNSF.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A source excitation module for the Harmonic-plus-Noise Source-Filter (HN-NSF) model.
///
/// This module generates excitation signals for speech synthesis:
/// - **Harmonic excitation (`sineSource`)**: Modeled by a sine wave generator (`SineGen`).
/// - **Noise excitation (`noiseSource`)**: Gaussian noise controlled by the voiced/unvoiced (UV) decision.
///
/// **How it works:**
/// 1. **Sine wave generation**: Produces a fundamental + harmonics waveform using `SineGen`.
/// 2. **Harmonic excitation merging**: Uses a linear layer to sum harmonics into a single excitation.
/// 3. **Noise generation**: Adds Gaussian noise based on the UV signal.
///
/// ### **Inputs:**
/// - `f0Sampled`: Input fundamental frequency tensor `(batch, length, 1)`.
///
/// ### **Outputs:**
/// - `sineSource`: Harmonic excitation `(batch, length, 1)`.
/// - `noiseSource`: Noise excitation `(batch, length, 1)`.
/// - `uv`: Voiced/unvoiced decision mask `(batch, length, 1)`.

class SourceModuleHnNSF: Module {
    let sineAmp: Float
    let noiseStd: Float
    @ModuleInfo var sineGen: SineGen
    @ModuleInfo var linearLayer: Linear

    init(
        samplingRate: Int,
        upsampleScale: Float,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        addNoiseStd: Float = 0.003,
        voicedThreshold: Float = 0
    ) {
        self.sineAmp = sineAmp
        self.noiseStd = addNoiseStd

        // Sine wave generator
        self.sineGen = SineGen(
            sampRate: samplingRate,
            upsampleScale: upsampleScale,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: addNoiseStd,
            voicedThreshold: voicedThreshold
        )

        // Linear layer to merge harmonic components into a single excitation signal
        self.linearLayer = Linear(harmonicNum + 1, 1)

        super.init()
    }

    /// Generates harmonic and noise-based excitation signals from F0.
    ///
    /// - Parameter f0Sampled: Input F0 tensor `(batch, length, 1)`.
    /// - Returns: A tuple containing:
    ///   - `sineSource`: Harmonic excitation `(batch, length, 1)`.
    ///   - `noiseSource`: Noise excitation `(batch, length, 1)`.
    ///   - `uv`: Voiced/unvoiced decision `(batch, length, 1)`.
    func callAsFunction(_ f0Sampled: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // Step 1: Generate sine waveforms and voiced/unvoiced mask
        let (sineWaves, uv, _) = sineGen(f0: f0Sampled)

        // Step 2: Merge harmonics into a single excitation signal
        let sineMerge = MLX.tanh(linearLayer(sineWaves))

        // Step 3: Generate noise with the same shape as UV
        let noise = normal(uv.shape) * sineAmp / 3.0

        return (sineMerge, noise, uv)
    }
}
