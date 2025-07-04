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

    var upsample: UpSample1d
    @ModuleInfo(key: "conv1") var conv1: ConvWeighted
    @ModuleInfo(key: "conv2") var conv2: ConvWeighted
    @ModuleInfo(key: "norm1") var norm1: AdaIN1d
    @ModuleInfo(key: "norm2") var norm2: AdaIN1d
    @ModuleInfo(key: "pool") var pool: ConvWeighted?
    @ModuleInfo(key: "conv1x1") var conv1x1: ConvWeighted?

    init(
        dimIn: Int,
        dimOut: Int,
        styleDim: Int = 64,
        activation: @escaping (MLXArray) -> MLXArray = { MLXNN.leakyRelu($0, negativeSlope: 0.2) },
        upsample: Bool = false,
        bias: Bool = false
    ) {
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.learnedShortcut = dimIn != dimOut
        self.shouldUpsamle = upsample
        self.activation = activation

        self.upsample = UpSample1d(upsample: upsample)
        self._conv1.wrappedValue = ConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 3, stride: 1, padding: 1)
        self._conv2.wrappedValue = ConvWeighted(inChannels: dimOut, outChannels: dimOut, kernelSize: 3, stride: 1, padding: 1)
        self._norm1.wrappedValue = AdaIN1d(styleDim: styleDim, numFeatures: dimIn)
        self._norm2.wrappedValue = AdaIN1d(styleDim: styleDim, numFeatures: dimOut)

        if upsample {
            self._pool.wrappedValue = ConvWeighted(inChannels: 1, outChannels: dimIn, kernelSize: 3, stride: 2, padding: 1, groups: dimIn)
        } else {
            self._pool.wrappedValue = nil
        }

        if learnedShortcut {
            self._conv1x1.wrappedValue = ConvWeighted(inChannels: dimIn, outChannels: dimOut, kernelSize: 1, stride: 1, padding: 0, bias: false)
        } else {
            self._conv1x1.wrappedValue = nil
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
            x = pool(x: x, conv: convTransposed1d_wrapped)
            x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((1, 0)), IntOrPair((0, 0))])
            x = x.transposed(axes: [0, 2, 1])
        }

        x = x.transposed(axes: [0, 2, 1])
        x = conv1(x: x, conv: MLX.conv1d)
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
