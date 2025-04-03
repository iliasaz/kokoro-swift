//
//  ConvWeighted.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// `ConvWeighted` implements a 1D convolution layer with weight normalization.
///
/// Weight normalization is a reparameterization technique that decomposes each weight tensor
/// into a magnitude (`weightG`) and a direction (`weightV`), improving training stability.
/// This module follows the formulation from *Salimans & Kingma, 2016*.
///
/// - Properties:
///   - `stride`: The stride of the convolution.
///   - `padding`: The padding applied before convolution.
///   - `dilation`: The spacing between kernel elements.
///   - `groups`: Number of input feature groups for grouped convolution.
///   - `weightG`: A learnable scalar magnitude per output channel.
///   - `weightV`: A learnable weight direction vector.
///   - `bias`: An optional bias term applied to the output.
///
/// - Initialization:
///   - `init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 1, dilation: Int = 1, groups: Int = 1, bias: Bool = true, encode: Bool = false)`
///     - Configures the convolution layer with optional bias and weight normalization.
///     - Parameters:
///       - `inChannels`: Number of input feature channels.
///       - `outChannels`: Number of output feature channels.
///       - `kernelSize`: Size of the convolution kernel.
///       - `stride`: Step size for convolution (default: `1`).
///       - `padding`: Number of zero-padding elements added to input (default: `1`).
///       - `dilation`: Spacing between kernel elements (default: `1`).
///       - `groups`: Number of feature groups (default: `1`).
///       - `bias`: Whether to include a bias term (default: `true`).
///       - `encode`: If `true`, bias is shaped to match `inChannels` instead of `outChannels`.
///
/// - Methods:
///   - `callAsFunction(x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int) -> MLXArray) -> MLXArray`
///     - Applies weight normalization to the convolution weights and performs convolution.
///     - Automatically transposes weights if necessary based on input shape.
///     - Parameters:
///       - `x`: Input tensor of shape `[batch, inChannels, length]`.
///       - `conv`: Convolution function to apply.
///     - Returns: The convolved output tensor.
///
/// - Static Methods:
///   - `weightNorm(weightV: MLXArray, weightG: MLXArray, dim: Int? = nil) -> MLXArray`
///     - Normalizes `weightV` along the specified dimension and scales it by `weightG`.
///   - `computeNorm(x: MLXArray, p: Int, dim: [Int]? = nil, keepDim: Bool = false) -> MLXArray`
///     - Computes the L1 or L2 norm of `x` along the specified dimensions.

class ConvWeighted: Module {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int
    @ModuleInfo var weightG: MLXArray
    @ModuleInfo var weightV: MLXArray
    @ModuleInfo var bias: MLXArray?

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        encode: Bool = false
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        // Initialize weight magnitude (g) and direction (v) vectors
        self.weightG = MLX.ones([outChannels, 1, 1]) // Scalar magnitude per output channel
        self.weightV = MLX.ones([outChannels, kernelSize, inChannels]) // Direction vectors

        // Initialize bias if needed
        if bias {
            self.bias = MLX.zeros([encode ? inChannels : outChannels])
        } else {
            self.bias = nil
        }

        super.init()
    }

    func callAsFunction(x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
        let weight = Self.weightNorm(weightV: weightV, weightG: weightG, dim: 0)

        var biasReshaped: MLXArray? = nil
        if let bias = bias {
            biasReshaped = bias.reshaped([1, 1, -1])
        }

        func applyConv(_ x: MLXArray, _ weightToUse: MLXArray) -> MLXArray {
            var result = conv(x, weightToUse, stride, padding, dilation, groups, .default)
            if let biasReshaped = biasReshaped {
                result += biasReshaped
            }
            return result
        }

        if x.shape.last == weight.shape.last || groups > 1 {
            return applyConv(x, weight)
        } else {
            return applyConv(x, weight.transposed(axes: [2, 1, 0]))
        }
    }

    /// Applies weight normalization to the input tensor.
    /// Reparameterizes weights as w = g * (v / ||v||)
    static func weightNorm(weightV: MLXArray, weightG: MLXArray, dim: Int? = nil) -> MLXArray {
        let rank = weightV.ndim
        var axes: [Int]

        if let dim = dim {
            let adjustedDim = dim < -1 ? dim + rank : dim
            axes = Array(0..<rank).filter { $0 != adjustedDim }
        } else {
            axes = Array(0..<rank)
        }

        let normV = computeNorm(x: weightV, p: 2, dim: axes, keepDim: true)
        return (weightV / (normV + 1e-7)) * weightG
    }

    /// Computes the p-norm of a tensor along specified dimensions.
    static func computeNorm(x: MLXArray, p: Int, dim: [Int]? = nil, keepDim: Bool = false) -> MLXArray {
        guard p == 1 || p == 2 else {
            fatalError("Only p-norms with p of 1 or 2 are supported")
        }

        let dims = dim ?? Array(0..<x.ndim)

        if p == 1 {
            return x.abs().sum(axes: dims, keepDims: keepDim)
        } else {
            return (x * x).sum(axes: dims, keepDims: keepDim).sqrt()
        }
    }
}
