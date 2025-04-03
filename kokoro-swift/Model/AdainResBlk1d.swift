//
//  AdainResBlk1d.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

class UpSample1d: Module {
    let shouldUpsample: Bool
    @ModuleInfo var interpolate: Upsample

    init(upsample: Bool) {
        self.shouldUpsample = upsample
        self.interpolate = Upsample(scaleFactor: 2.0, mode: .nearest) // Equivalent to PyTorch's nearest neighbor upsampling
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // If upsampling is not required, return input as is.
        guard shouldUpsample else { return x }

        // Apply nearest-neighbor upsampling using MLXNN's Upsample module
        return interpolate(x)
    }
}


class AdainResBlk1d: Module {
    let dimIn: Int
    let dimOut: Int
    let learnedShortcut: Bool
    let shouldUpsamle: Bool
    let activation: (MLXArray) -> MLXArray

    @ModuleInfo var upsample: UpSample1d
    @ModuleInfo var conv1: ConvWeighted
    @ModuleInfo var conv2: ConvWeighted
    @ModuleInfo var norm1: AdaIN1d
    @ModuleInfo var norm2: AdaIN1d
    @ModuleInfo var dropout: Dropout
    @ModuleInfo var pool: ConvWeighted?
    @ModuleInfo var conv1x1: ConvWeighted?

    init(
        dimIn: Int,
        dimOut: Int,
        styleDim: Int = 64,
        activation: @escaping (MLXArray) -> MLXArray = { MLXNN.leakyRelu($0, negativeSlope: 0.2) },
        upsample: Bool = false,
        dropoutP: Float = 0.0,
        bias: Bool = false
    ) {
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.learnedShortcut = dimIn != dimOut
        self.shouldUpsamle = upsample
        self.activation = activation

        self.upsample = UpSample1d(upsample: upsample)
        self.conv1 = ConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 3, stride: 1, padding: 1)
        self.conv2 = ConvWeighted(inChannels: dimOut, outChannels: dimOut, kernelSize: 3, stride: 1, padding: 1)
        self.norm1 = AdaIN1d(styleDim: styleDim, numFeatures: dimIn)
        self.norm2 = AdaIN1d(styleDim: styleDim, numFeatures: dimOut)
        self.dropout = Dropout(p: dropoutP)

        if upsample {
            self.pool = ConvWeighted(inChannels: 1, outChannels: dimIn, kernelSize: 3, stride: 2, padding: 1, groups: dimIn)
        } else {
            self.pool = nil
        }

        if learnedShortcut {
            self.conv1x1 = ConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 1, stride: 1, padding: 0, bias: false)
        } else {
            self.conv1x1 = nil
        }

        super.init()
    }

    /// Applies the shortcut connection (skip connection).
    private func shortcut(_ x: MLXArray) -> MLXArray {
        var x = x.transposed(axes: [0, 2, 1]) // Swap to (batch, seq_len, channels)
        x = upsample(x)
        x = x.transposed(axes: [0, 2, 1]) // Restore shape

        if let conv1x1 = conv1x1 {
            x = x.transposed(axes: [0, 2, 1])
            x = conv1x1(x: x, conv: MLX.conv1d)
            x = x.transposed(axes: [0, 2, 1])
        }
        return x
    }

    /// Computes the residual transformation.
    private func residual(_ x: MLXArray, s: MLXArray) -> MLXArray {
        var x = norm1(x: x, s: s)
        x = activation(x)

        if shouldUpsamle, let pool = pool {
            x = x.transposed(axes: [0, 2, 1])
            x = pool(x: x, conv: convTransposed1d)
            x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0))])
            x = x.transposed(axes: [0, 2, 1])
        }

        x = x.transposed(axes: [0, 2, 1])
        x = conv1(x: dropout(x), conv: MLX.conv1d)
        x = x.transposed(axes: [0, 2, 1])

        x = norm2(x: x, s: s)
        x = activation(x)

        x = x.transposed(axes: [0, 2, 1])
        x = conv2(x: x, conv: MLX.conv1d)
        x = x.transposed(axes: [0, 2, 1])

        return x
    }

    /// Forward pass of the residual block.
    func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        let out = residual(x, s: s)
        return (out + shortcut(x)) / sqrt(2.0)
    }
}
