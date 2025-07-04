//
//  AdaIN1d.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// `AdaIN1d` implements Adaptive Instance Normalization (AdaIN) for 1D feature maps.
/// It is commonly used in style-based architectures where instance normalization is modulated
/// by an external style input.
///
/// This module normalizes the input using `InstanceNorm` and then applies adaptive scaling (`gamma`)
/// and shifting (`beta`) computed from a learned style vector.
///
/// - Properties:
///   - `norm`: An instance of `InstanceNorm`, which normalizes the input per feature.
///   - `fc`: A fully connected layer that maps the style vector to scaling (`gamma`) and shifting (`beta`) parameters.
///
/// - Initialization:
///   - `init(styleDim: Int, numFeatures: Int)`
///     - Initializes the normalization and style modulation components.
///     - Parameters:
///       - `styleDim`: The dimensionality of the style vector.
///       - `numFeatures`: The number of feature channels in the input tensor.
///
/// - Methods:
///   - `callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray`
///     - Applies instance normalization followed by adaptive modulation.
///     - Steps:
///       1. Passes the style vector `s` through a fully connected layer to obtain `gamma` and `beta`.
///       2. Expands the dimensions of `gamma` and `beta` for proper broadcasting.
///       3. Applies instance normalization to `x`.
///       4. Scales and shifts the normalized output using `(1 + gamma) * norm(x) + beta`.
///     - Parameters:
///       - `x`: The input tensor of shape `[batch, numFeatures, length]`.
///       - `s`: The style vector of shape `[batch, styleDim]`.
///     - Returns: The adapted tensor after applying style-based normalization.

class AdaIN1d: Module {
    var norm: InstanceNorm
    @ModuleInfo(key: "fc") var fc: Linear

    init(styleDim: Int, numFeatures: Int) {
        self.norm = InstanceNorm(dimensions: numFeatures, affine: false)
        self._fc.wrappedValue = Linear(styleDim, numFeatures * 2, zeroInitialized: true)
        super.init()
    }

    func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
        // Compute style-based parameters via fully connected layer
        var h = fc(s)
        h = h.expandedDimensions(axis: 2)  // Equivalent to view(..., 1) in PyTorch

        // Split into gamma and beta
        let (gamma, beta) = h.split(axis: 1)
        return (1 + gamma) * norm(x.swappedAxes(1, 2)).swappedAxes(1, 2) + beta
    }
}

