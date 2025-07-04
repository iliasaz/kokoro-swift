//
//  AdaLayerNorm.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN

/// `AdaLayerNorm` is an adaptive layer normalization module that modulates input activations
/// using a style vector. It computes scale (`gamma`) and shift (`beta`) parameters dynamically
/// based on the provided style input.
///
/// This module is useful in architectures where adaptive feature normalization is required,
/// such as style-based generative models or conditioning mechanisms in neural networks.
///
/// - Properties:
///   - `channels`: The number of channels in the input tensor.
///   - `eps`: A small value added to the variance to improve numerical stability.
///   - `fc`: A `Linear` layer that maps the style vector to scale (`gamma`) and shift (`beta`) parameters.
///
/// - Initialization:
///   - `init(styleDim: Int, channels: Int, eps: Float = 1e-5)`
///     - Initializes the module with the given style dimension and number of channels.
///     - Parameters:
///       - `styleDim`: The dimensionality of the style input vector.
///       - `channels`: The number of channels in the input tensor.
///       - `eps`: A small constant for numerical stability (default: `1e-5`).
///
/// - Methods:
///   - `callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray`
///     - Applies adaptive layer normalization to the input tensor `x` using the style vector `s`.
///     - Steps:
///       1. Computes the `gamma` (scaling) and `beta` (shifting) parameters via the fully-connected layer (`fc`).
///       2. Reshapes and splits the output into `gamma` and `beta`.
///       3. Computes mean and variance of `x` along the last axis.
///       4. Normalizes `x` using `(x - mean) / sqrt(variance + eps)`.
///       5. Scales and shifts the normalized tensor using `(1 + gamma) * normalizedX + beta`.
///     - Parameters:
///       - `x`: The input tensor of shape `[batch, channels, feature]`.
///       - `s`: The style vector of shape `[batch, styleDim]`.
///     - Returns: The adapted tensor after applying style-based normalization.

class AdaLayerNorm: Module {
    let channels: Int
    let eps: Float
    @ModuleInfo var fc: Linear

    init(styleDim: Int, channels: Int, eps: Float = 1e-5) {
        self.channels = channels
        self.eps = eps
        self.fc = Linear(styleDim, channels * 2, zeroInitialized: true)
        super.init()
    }

    func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
        // Compute style-based parameters via a fully-connected layer.
        let h = fc(s)
        // Reshape h to shape [batch, channels*2, 1]
        let hReshaped = h.reshaped([h.shape[0], h.shape[1], 1])
        // Split h along axis 1 into two parts: gamma and beta.
        let splits = hReshaped.split(axis: 1)
        var gamma = splits.0
        var beta = splits.1
        // Transpose gamma and beta to shape [1, batch, channels]
        gamma = gamma.transposed(axes: [2, 0, 1])
        beta = beta.transposed(axes: [2, 0, 1])

        // Compute instance normalization on x.
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normalizedX = (x - mean) / MLX.sqrt(variance + eps)
        // Apply adaptive scaling and shifting.
        return (1.0 + gamma) * normalizedX + beta
    }
}
