//
//  Decoder.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/16/25.
//

import Foundation
import MLX
import MLXNN

/// The Decoder module is responsible for processing ASR embeddings, pitch (F0), and noise signals,
/// refining them through adaptive instance normalization blocks before passing them into the generator.
/// The final output is a synthesized waveform.
///
/// - The `encode` step applies an initial transformation to the input.
/// - The `decode` step refines the processed features.
/// - The `generator` synthesizes the final waveform using the learned representations.
class Decoder: Module {
    @ModuleInfo var encode: AdainResBlk1d
    @ModuleInfo var decodeLayers: [AdainResBlk1d]
    @ModuleInfo var f0Conv: ConvWeighted
    @ModuleInfo var nConv: ConvWeighted
    @ModuleInfo var asrResConv: ConvWeighted
    @ModuleInfo var generator: Generator

    init(
        dimIn: Int,
        styleDim: Int,
        dimOut: Int,
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genISTFTNFFT: Int,
        genISTFTHopSize: Int
    ) {
        // Initial encoding block
        self.encode = AdainResBlk1d(dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim)

        // Sequential decoding layers
        self.decodeLayers = [
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: true)
        ]

        // F0 and Noise convolutions for signal conditioning
        self.f0Conv = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1)
        self.nConv = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1)

        // ASR residual transformation
        self.asrResConv = ConvWeighted(inChannels: 512, outChannels: 64, kernelSize: 1, padding: 0)

        // Generator for final waveform synthesis
        self.generator = Generator(
            styleDim: styleDim,
            resblockKernelSizes: resblockKernelSizes,
            upsampleRates: upsampleRates,
            upsampleInitialChannel: upsampleInitialChannel,
            resblockDilationSizes: resblockDilationSizes,
            upsampleKernelSizes: upsampleKernelSizes,
            genISTFTNFFT: genISTFTNFFT,
            genISTFTHopSize: genISTFTHopSize
        )

        super.init()
    }

    func callAsFunction(asr: MLXArray, f0Curve: MLXArray, n: MLXArray, s: MLXArray) -> MLXArray {
        // Process F0 and Noise signals
        var f0 = f0Curve.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])
        f0 = f0Conv(x: f0, conv: conv1d).transposed(axes: [0, 2, 1])

        var n = n.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])
        n = nConv(x: n, conv: conv1d).transposed(axes: [0, 2, 1])

        // Concatenate ASR, F0, and Noise inputs
        var x = MLX.concatenated([asr, f0, n], axis: 1)
        x = encode(x, s: s)

        // Process ASR residuals
        var asrRes = asr.transposed(axes: [0, 2, 1])
        asrRes = asrResConv(x: asrRes, conv: conv1d).transposed(axes: [0, 2, 1])

        var residualConnection = true
        for block in decodeLayers {
            if residualConnection {
                x = MLX.concatenated([x, asrRes, f0, n], axis: 1)
            }
            x = block(x, s: s)

            // Disable residual connection if the current block applies upsampling
            if block.shouldUpsamle {
                residualConnection = false
            }
        }

        // Pass through the generator for final waveform output
        x = generator(x: x, style: s, f0: f0Curve)
        return x
    }

    /// Handles weight sanitization for compatibility with different model versions.
    func sanitize(key: String, weights: MLXArray) -> MLXArray {
        var sanitizedWeights: MLXArray?

        if key.contains("noise_convs") && key.hasSuffix(".weight") {
            sanitizedWeights = weights.transposed(axes: [0, 2, 1])
        } else if key.contains("weight_v") {
            if checkArrayShape(weights) {
                sanitizedWeights = weights
            } else {
                sanitizedWeights = weights.transposed(axes: [0, 2, 1])
            }
        } else {
            sanitizedWeights = weights
        }

        return sanitizedWeights ?? weights
    }

    private func checkArrayShape(_ arr: MLXArray) -> Bool {
        let shape = arr.shape

        // Ensure the array has exactly 3 dimensions
        guard shape.count == 3 else {
            return false
        }

        let outChannels = shape[0]
        let kH = shape[1]
        let kW = shape[2]

        // Check if outChannels is the largest and if kH and kW are the same
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }
}
