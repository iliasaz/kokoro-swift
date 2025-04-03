//
//  AlbertLayer.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertLayer` represents a single transformer block in the ALBERT model.
/// It consists of a self-attention mechanism followed by a feed-forward network.
///
/// - Properties:
///   - `attention`: An instance of `AlbertSelfAttention` for computing self-attention over the input.
///   - `fullLayerLayerNorm`: A `LayerNorm` module that normalizes the output with a residual connection.
///   - `ffn`: A `Linear` layer that projects the attention output to `intermediateSize`.
///   - `ffnOutput`: A `Linear` layer that projects the intermediate representation back to `hiddenSize`.
///   - `activation`: A `GELU` activation function for non-linearity.
///
/// - Initialization:
///   - `init(config: AlbertModelArgs)`
///     - Initializes the self-attention and feed-forward components based on the model configuration.
///
/// - Methods:
///   - `callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray`
///     - Applies self-attention followed by a feed-forward network.
///     - Steps:
///       1. Computes `attentionOutput` using self-attention.
///       2. Passes `attentionOutput` through the feed-forward network (`ffChunk`).
///       3. Applies a residual connection and layer normalization.
///     - Parameters:
///       - `hiddenStates`: The input tensor of shape `[batchSize, seqLength, hiddenSize]`.
///       - `attentionMask`: An optional mask to prevent attention to certain positions.
///     - Returns: A tensor of shape `[batchSize, seqLength, hiddenSize]` with transformed representations.
///
/// - Private Methods:
///   - `ffChunk(_ attentionOutput: MLXArray) -> MLXArray`
///     - Applies the feed-forward transformation with activation.
///     - Steps:
///       1. Projects `attentionOutput` to `intermediateSize` using `ffn`.
///       2. Applies the `GELU` activation.
///       3. Projects back to `hiddenSize` using `ffnOutput`.
///     - Returns: A tensor of shape `[batchSize, seqLength, hiddenSize]`.

class AlbertLayer: Module {
    @ModuleInfo var attention: AlbertSelfAttention
    @ModuleInfo var fullLayerLayerNorm: LayerNorm
    @ModuleInfo var ffn: Linear
    @ModuleInfo var ffnOutput: Linear
    @ModuleInfo var activation: GELU

    init(config: AlbertModelArgs) {
        super.init()
        self.attention = AlbertSelfAttention(config: config)
        self.fullLayerLayerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.ffn = Linear(config.hiddenSize, config.intermediateSize)
        self.ffnOutput = Linear(config.intermediateSize, config.hiddenSize)
        self.activation = GELU()
    }

    func callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let attentionOutput = attention(hiddenStates: hiddenStates, attentionMask: attentionMask)
        let ffnOutput = ffChunk(attentionOutput)
        let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
        return output
    }

    private func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
        var output = ffn(attentionOutput)
        output = activation(output)
        output = ffnOutput(output)
        return output
    }
}
