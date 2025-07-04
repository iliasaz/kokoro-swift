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
    @ModuleInfo(key: "encode") var encode: AdainResBlk1d
    @ModuleInfo(key: "decode") var decodeLayers: [AdainResBlk1d]
    @ModuleInfo(key: "F0_conv") var f0Conv: ConvWeighted
    @ModuleInfo(key: "N_conv") var nConv: ConvWeighted
    @ModuleInfo(key: "asr_res") var asrResConv: [ConvWeighted]
    @ModuleInfo(key: "generator") var generator: Generator

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
        self._encode.wrappedValue = AdainResBlk1d(dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim)

        // Sequential decoding layers
        self._decodeLayers.wrappedValue = [
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim),
            AdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: true)
        ]

        // F0 and Noise convolutions for signal conditioning
        self._f0Conv.wrappedValue = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1)
        self._nConv.wrappedValue = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1, groups: 1)

        // ASR residual transformation
        self._asrResConv.wrappedValue = [ConvWeighted(inChannels: 512, outChannels: 64, kernelSize: 1, padding: 0)]

        // Generator for final waveform synthesis
        self._generator.wrappedValue = Generator(
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
//        let loadedF0Curve: MLXArray
//        do {
//            print("loading F0Curve from /Users/ilia/Downloads/F0Curve.safetensors")
//            loadedF0Curve = try MLX.loadArrays(url: URL(fileURLWithPath: "/Users/ilia/Downloads/F0Curve.safetensors"))["F0Curve"]!
//        } catch {
//            fatalError("could not load F0Curve")
//        }
//        var f0 = loadedF0Curve.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])

        var f0 = f0Curve.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])

        f0 = f0Conv(x: f0, conv: conv1d).transposed(axes: [0, 2, 1])

//        let n0 = MLX.zeros(like: n)
//        var n = n0.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])
        var n = n.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])
        n = nConv(x: n, conv: conv1d).transposed(axes: [0, 2, 1])

        // Concatenate ASR, F0, and Noise inputs
        var x = MLX.concatenated([asr, f0, n], axis: 1)

        print("before encode: \(x.mean().item(Float.self)), \(x.max().item(Float.self))")
        x = encode(x, s: s)
        print("after encode mean/max:", x.mean().item(Float.self), x.max().item(Float.self))

        // Process ASR residuals
        var asrRes = asr.transposed(axes: [0, 2, 1])
        asrRes = asrResConv[0](x: asrRes, conv: conv1d).transposed(axes: [0, 2, 1])

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
        print("before generator mean/max:", x.mean(), x.max())
        // Pass through the generator for final waveform output
        let audio = generator(x: x, style: s, f0: f0Curve)
//        let audio = generator(x: x, style: s, f0: loadedF0Curve)

        return audio
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
