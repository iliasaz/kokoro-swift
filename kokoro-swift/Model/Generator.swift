//
//  Generator.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//
import Foundation
import MLX
import MLXNN

/// The Generator module for waveform synthesis.
/// This model uses a series of convolutional, upsampling, and adaptive normalization layers
/// to generate high-quality speech signals from learned representations.
class Generator: Module {
    @ModuleInfo var sourceModule: SourceModuleHnNSF
    @ModuleInfo var f0Upsample: Upsample
    @ModuleInfo var upsampleLayers: [ConvWeighted]
    @ModuleInfo var resBlocks: [AdaINResBlock1]
    @ModuleInfo var noiseConvs: [Conv1d]
    @ModuleInfo var noiseRes: [AdaINResBlock1]
    @ModuleInfo var convPost: ConvWeighted
    @ModuleInfo var reflectionPad: ReflectionPad1d
    @ModuleInfo var stft: MLXSTFT

    let numKernels: Int
    let numUpsamples: Int
    let postNFFT: Int

    init(
        styleDim: Int,
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genISTFTNFFT: Int,
        genISTFTHopSize: Int
    ) {
        self.numKernels = resblockKernelSizes.count
        self.numUpsamples = upsampleRates.count
        self.postNFFT = genISTFTNFFT

        // Source module for harmonic-plus-noise source modeling
        self.sourceModule = SourceModuleHnNSF(
            samplingRate: 24000,
            upsampleScale: Float(upsampleRates.reduce(1, *) * genISTFTHopSize),
            harmonicNum: 8,
            voicedThreshold: 10
        )

        // F0 Upsampling
        self.f0Upsample = Upsample(scaleFactor: .float(Float(upsampleRates.reduce(1, *) * genISTFTHopSize)))

        // Upsampling and ResBlocks
        var upsampleLayers: [ConvWeighted] = []
        var resBlocks: [AdaINResBlock1] = []
        var noiseConvs: [Conv1d] = []
        var noiseRes: [AdaINResBlock1] = []

        for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            let inChannels = upsampleInitialChannel / (1 << (i + 1))
            let outChannels = upsampleInitialChannel / (1 << i)

            upsampleLayers.append(
                ConvWeighted(inChannels: inChannels, outChannels: outChannels, kernelSize: k, stride: u, padding: (k - u) / 2, encode: true)
            )

            for (k, d) in zip(resblockKernelSizes, resblockDilationSizes) {
                resBlocks.append(AdaINResBlock1(channels: inChannels, kernelSize: k, dilation: d, styleDim: styleDim))
            }

            let cCur = upsampleInitialChannel / (1 << (i + 1))
            if i + 1 < numUpsamples {
                let strideF0 = upsampleRates[(i + 1)...].reduce(1, *)
                noiseConvs.append(
                    Conv1d(inputChannels: genISTFTNFFT + 2, outputChannels: cCur, kernelSize: strideF0 * 2, stride: strideF0, padding: (strideF0 + 1) / 2)
                )
                noiseRes.append(AdaINResBlock1(channels: cCur, kernelSize: 7, dilation: [1, 3, 5], styleDim: styleDim))
            } else {
                noiseConvs.append(Conv1d(inputChannels: genISTFTNFFT + 2, outputChannels: cCur, kernelSize: 1))
                noiseRes.append(AdaINResBlock1(channels: cCur, kernelSize: 11, dilation: [1, 3, 5], styleDim: styleDim))
            }
        }

        self.upsampleLayers = upsampleLayers
        self.resBlocks = resBlocks
        self.noiseConvs = noiseConvs
        self.noiseRes = noiseRes

        // Post-processing layers
        let lastChannel = upsampleInitialChannel / (1 << numUpsamples)
        self.convPost = ConvWeighted(inChannels: lastChannel, outChannels: genISTFTNFFT + 2, kernelSize: 7, stride: 1, padding: 3)
        self.reflectionPad = ReflectionPad1d(padding: (1, 0))

        // Short-time Fourier Transform module
        self.stft = MLXSTFT(
            filterLength: genISTFTNFFT,
            hopLength: genISTFTHopSize,
            winLength: genISTFTNFFT
        )

        super.init()
    }

    func callAsFunction(x: MLXArray, style: MLXArray, f0: MLXArray) -> MLXArray {
        // Step 1: Upsample f0
        var f0 = f0.expandedDimensions(axis: 1).transposed(axes: [0, 2, 1])
        f0 = f0Upsample(f0)

        // Step 2: Generate harmonic and noise sources
        var (harSource, _, _) = sourceModule(f0)
        harSource = harSource.transposed(axes: [0, 2, 1]).squeezed(axis: 1)
        let (harSpec, harPhase) = stft.transform(harSource)
        var har = MLX.concatenated([harSpec, harPhase], axis: 1).transposed(axes: [0, 2, 1])
        var x = x
        for i in 0..<numUpsamples {
            x = leakyRelu(x, negativeSlope: 0.1)

            var xSource = noiseConvs[i](har).transposed(axes: [0, 2, 1])
            xSource = noiseRes[i](x: xSource, s: style)

            x = x.transposed(axes: [0, 2, 1])
            x = upsampleLayers[i](x: x, conv: MLX.convTransposed1d).transposed(axes: [0, 2, 1])

            if i == numUpsamples - 1 {
                x = reflectionPad(x)
            }
            x += xSource

            var xs: MLXArray?
            for j in 0..<numKernels {
                let resBlockOut = resBlocks[i * numKernels + j](x: x, s: style)
                xs = xs.map { $0 + resBlockOut } ?? resBlockOut
            }
            x = xs! / Float(numKernels)
        }

        x = leakyRelu(x, negativeSlope: 0.01)

        x = x.transposed(axes: [0, 2, 1])
        x = convPost(x: x, conv: conv1d).transposed(axes: [0, 2, 1])

        // Final waveform synthesis
        let spec = MLX.exp(x[0..., 0..<postNFFT / 2 + 1, 0...])
        let phase = MLX.sin(x[0..., (postNFFT / 2 + 1)..., 0...])
        return stft.inverse(magnitude: spec, phase: phase)
    }
}
