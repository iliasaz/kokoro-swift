//
//  InstanceNorm.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// `InstanceNorm` implements Instance Normalization as described in the paper:
/// *"Instance Normalization: The Missing Ingredient for Fast Stylization"*
/// (Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky, 2016).
///
/// Instance Normalization (IN) normalizes the input tensor by computing per-instance
/// mean and variance along spatial dimensions, effectively standardizing each feature
/// independently for every sample in the batch. Unlike Batch Normalization, IN does
/// not normalize across the batch, making it particularly effective for style transfer
/// and image generation tasks.
///
/// This module supports optional affine transformation and tracking of running statistics.
///
/// - Properties:
///   - `numFeatures`: Number of feature channels in the input tensor.
///   - `eps`: A small constant added to variance for numerical stability.
///   - `momentum`: Momentum factor for updating running statistics.
///   - `affine`: Whether to apply learnable scale (`weight`) and shift (`bias`).
///   - `trackRunningStats`: Whether to track running mean and variance during training.
///
/// - Initialization:
///   - `init(numFeatures: Int, eps: Float = 1e-5, momentum: Float = 0.1, affine: Bool = false, trackRunningStats: Bool = false)`
///     - Configures the normalization layer with optional learnable parameters.
///     - If `affine == true`, the layer learns `weight` and `bias` for each channel.
///     - If `trackRunningStats == true`, the layer tracks moving averages of mean and variance.
///
/// - Methods:
///   - `callAsFunction(_ input: MLXArray) -> MLXArray`
///     - Normalizes the input tensor per instance, computing mean and variance across spatial dimensions.
///     - If `affine == true`, applies a learnable scale and shift transformation.
///     - If `trackRunningStats == true`, updates running statistics during training.
///     - Parameters:
///       - `input`: The input tensor of shape `[batch, features, spatialDims...]`.
///     - Returns: The instance-normalized tensor.

class InstanceNorm: Module {
    let numFeatures: Int
    let eps: Float
    let momentum: Float
    let affine: Bool
    let trackRunningStats: Bool

    @ModuleInfo var weight: MLXArray?
    @ModuleInfo var bias: MLXArray?
    @ModuleInfo var runningMean: MLXArray?
    @ModuleInfo var runningVar: MLXArray?

    init(numFeatures: Int, eps: Float = 1e-5, momentum: Float = 0.1, affine: Bool = false, trackRunningStats: Bool = false) {
        self.numFeatures = numFeatures
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.trackRunningStats = trackRunningStats

        super.init()

        // Initialize parameters
        if self.affine {
            self.weight = MLX.ones([numFeatures])
            self.bias = MLX.zeros([numFeatures])
        }

        if self.trackRunningStats {
            self.runningMean = MLX.zeros([numFeatures])
            self.runningVar = MLX.ones([numFeatures])
        }
    }

    func getNoBatchDim() -> Int {
        return 2
    }

    func checkInputDim(_ input: MLXArray) throws {
        if input.ndim != 2 && input.ndim != 3 {
            throw NSError(domain: "InstanceNorm", code: 1, userInfo: [NSLocalizedDescriptionKey: "Expected 2D or 3D input, but got \(input.ndim)D input"])
        }
    }

    func handleNoBatchInput(_ input: MLXArray) -> MLXArray {
        let expanded = input.expandedDimensions(axis: 0)
        let result = applyInstanceNorm(expanded)
        return result.squeezed()
    }

    func applyInstanceNorm(_ input: MLXArray) -> MLXArray {
        let dims = Array(0..<input.ndim)
        let featureDim = dims[dims.count - getNoBatchDim()]

        // Compute statistics along all dimensions except batch and feature dims
        let reduceDims = dims.filter { $0 != 0 && $0 != featureDim }

        var mean: MLXArray
        var variance: MLXArray

        if self.training || !self.trackRunningStats {
            mean = input.mean(axes: reduceDims, keepDims: true)
            variance = input.variance(axes: reduceDims, keepDims: true)

            if self.trackRunningStats && self.training {
                let overallMean = mean.mean(axis: 0)
                let overallVar = variance.mean(axis: 0)

                self.runningMean = (1 - self.momentum) * self.runningMean! + self.momentum * overallMean
                self.runningVar = (1 - self.momentum) * self.runningVar! + self.momentum * overallVar
            }
        } else {
            var meanShape = Array(repeating: 1, count: input.ndim)
            meanShape[featureDim] = self.numFeatures
            var varShape = meanShape

            mean = self.runningMean!.reshaped(meanShape)
            variance = self.runningVar!.reshaped(varShape)
        }

        // Normalize
        var xNorm = (input - mean) / MLX.sqrt(variance + self.eps)

        // Apply affine transform if enabled
        if self.affine {
            var weightShape = Array(repeating: 1, count: input.ndim)
            weightShape[featureDim] = self.numFeatures
            var biasShape = weightShape

            let weightReshaped = self.weight!.reshaped(weightShape)
            let biasReshaped = self.bias!.reshaped(biasShape)

            xNorm = xNorm * weightReshaped + biasReshaped
        }

        return xNorm
    }

    func callAsFunction(_ input: MLXArray) throws -> MLXArray {
        try checkInputDim(input)

        let featureDim = input.ndim - getNoBatchDim()
        if input.shape[featureDim] != self.numFeatures {
            if self.affine {
                throw NSError(domain: "InstanceNorm", code: 2, userInfo: [NSLocalizedDescriptionKey: "Expected input's size at dim=\(featureDim) to match numFeatures (\(self.numFeatures)), but got: \(input.shape[featureDim])."])
            } else {
                print("Warning: Input's size at dim=\(featureDim) does not match numFeatures. You can silence this warning by not passing in numFeatures when affine=False.")
            }
        }

        if input.ndim == getNoBatchDim() {
            return handleNoBatchInput(input)
        }

        return applyInstanceNorm(input)
    }
}
