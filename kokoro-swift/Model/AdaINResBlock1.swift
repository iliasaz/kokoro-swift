//
//  AdaINResBlock1.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// `AdaINResBlock1` implements an Adaptive Instance Normalization (AdaIN) residual block
/// designed for feature transformation in neural networks. This block is particularly
/// useful in style transfer and generative modeling, where feature statistics are adapted
/// based on external style inputs.
///
/// ### Architecture:
/// The block consists of three sequential sub-blocks, each applying:
/// 1. **Adaptive Instance Normalization (AdaIN1d)** - Normalizes input using a style vector.
/// 2. **Snake1D Activation** - A non-linear activation function based on periodic sine functions.
/// 3. **Weight-Normalized Convolution (`ConvWeighted`)** - Applies a learnable transformation.
///
/// Each sub-block has:
/// - `convs1`: A set of three weight-normalized convolutions.
/// - `convs2`: Another set of three weight-normalized convolutions.
/// - `adain1`: Three AdaIN modules applied before the first convolution.
/// - `adain2`: Three AdaIN modules applied before the second convolution.
/// - `alpha1`, `alpha2`: Learnable scaling parameters for the `Snake1D` activation function.
///
/// A **residual connection** is used at the end to stabilize training and improve information flow.
///
/// ### Reference:
/// This architecture is inspired by works in neural style transfer and adaptive feature modulation.
///
/// - Properties:
///   - `convs1`: First set of convolutions with weight normalization.
///   - `convs2`: Second set of convolutions.
///   - `adain1`: First set of AdaIN layers for adaptive normalization.
///   - `adain2`: Second set of AdaIN layers.
///   - `alpha1`, `alpha2`: Scaling factors used in Snake1D activation.
///
/// - Initialization:
///   - `init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5], styleDim: Int = 64)`
///     - Initializes the module with specified channels, kernel size, and dilation rates.
///     - Parameters:
///       - `channels`: Number of feature channels.
///       - `kernelSize`: The size of the convolutional kernel (default: `3`).
///       - `dilation`: A list of dilation values for the convolutions (default: `[1, 3, 5]`).
///       - `styleDim`: Dimensionality of the style vector for AdaIN (default: `64`).
///
/// - Methods:
///   - `callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray`
///     - Applies AdaIN normalization, Snake1D activation, and weight-normalized convolutions.
///     - Includes a residual connection to preserve information.
///     - Parameters:
///       - `x`: The input tensor of shape `[batch, channels, length]`.
///       - `s`: The style vector used for AdaIN modulation.
///     - Returns: The transformed tensor after feature modulation and convolution.

class AdaINResBlock1: Module {
    @ModuleInfo var convs1: [ConvWeighted]
    @ModuleInfo var convs2: [ConvWeighted]
    @ModuleInfo var adain1: [AdaIN1d]
    @ModuleInfo var adain2: [AdaIN1d]
    @ParameterInfo var alpha1: [MLXArray]
    @ParameterInfo var alpha2: [MLXArray]

    init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5], styleDim: Int = 64) {
        var convs1Local = [ConvWeighted]()
        var convs2Local = [ConvWeighted]()
        var adain1Local = [AdaIN1d]()
        var adain2Local = [AdaIN1d]()
        var alpha1Local = [MLXArray]()
        var alpha2Local = [MLXArray]()

        for i in 0..<3 {
            let padding1 = getPadding(kernelSize: kernelSize, dilation: dilation[i])
            let padding2 = getPadding(kernelSize: kernelSize, dilation: 1)

            convs1Local.append(ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize, stride: 1, padding: padding1, dilation: dilation[i]))
            convs2Local.append(ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize, stride: 1, padding: padding2, dilation: 1))

            adain1Local.append(AdaIN1d(styleDim: styleDim, numFeatures: channels))
            adain2Local.append(AdaIN1d(styleDim: styleDim, numFeatures: channels))

            alpha1Local.append(MLX.ones([1, channels, 1]))
            alpha2Local.append(MLX.ones([1, channels, 1]))
        }
        _convs1.wrappedValue = convs1Local
        _convs2.wrappedValue = convs2Local
        _adain1.wrappedValue = adain1Local
        _adain2.wrappedValue = adain2Local
        _alpha1.wrappedValue = alpha1Local
        _alpha2.wrappedValue = alpha2Local

        super.init()
    }

    func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
        var x = x

        for i in 0..<3 {
            var xt = adain1[i](x: x, s: s)  // Fixed argument labels
            xt = xt + (1 / alpha1[i]) * MLX.pow(MLX.sin(alpha1[i] * xt), 2)

            xt = xt.transposed(axes: [0, 2, 1])
            xt = convs1[i](x: xt, conv: MLX.conv1d)
            xt = xt.transposed(axes: [0, 2, 1])

            xt = adain2[i](x: xt, s: s)  // Fixed argument labels
            xt = xt + (1 / alpha2[i]) * MLX.pow(MLX.sin(alpha2[i] * xt), 2)

            xt = xt.transposed(axes: [0, 2, 1])
            xt = convs2[i](x: xt, conv: MLX.conv1d)
            xt = xt.transposed(axes: [0, 2, 1])

            x = xt + x  // Residual connection
        }

        return x
    }
}

public func getPadding(kernelSize: Int, dilation: Int) -> Int {
    return (kernelSize - 1) * dilation / 2
}
