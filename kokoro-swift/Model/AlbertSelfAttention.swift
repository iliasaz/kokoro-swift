//
//  AlbertSelfAttention.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertSelfAttention` implements the self-attention mechanism used in the ALBERT model.
/// It computes attention scores over input embeddings and applies weighted summation to produce
/// contextualized representations of input tokens.
///
/// This implementation follows the ALBERT paper:
/// **"ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"**
/// (Lan et al., 2019) - [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
///
/// - Properties:
///   - `numAttentionHeads`: The number of attention heads.
///   - `attentionHeadSize`: The size of each attention head.
///   - `allHeadSize`: The total size of the multi-head attention layer (`numAttentionHeads * attentionHeadSize`).
///   - `query`: A `Linear` layer that projects input embeddings into the query space.
///   - `key`: A `Linear` layer that projects input embeddings into the key space.
///   - `value`: A `Linear` layer that projects input embeddings into the value space.
///   - `dense`: A `Linear` layer that combines outputs from all attention heads.
///   - `layerNorm`: A `LayerNorm` module that normalizes the final output to stabilize training.
///   - `dropout`: A `Dropout` layer that applies regularization to prevent overfitting.
///
/// - Initialization:
///   - `init(config: AlbertModelArgs)`
///     - Initializes the self-attention layer based on the given ALBERT model configuration.
///     - Parameters:
///       - `config`: An instance of `AlbertModelArgs` containing hyperparameters such as hidden size, attention heads, and dropout probability.
///
/// - Methods:
///   - `transposeForScores(_ x: MLXArray) -> MLXArray`
///     - Reshapes the input tensor to separate attention heads and transposes the dimensions
///       to allow efficient computation of attention scores.
///     - Parameters:
///       - `x`: The input tensor of shape `[batchSize, seqLength, allHeadSize]`.
///     - Returns: A transposed tensor of shape `[batchSize, numAttentionHeads, seqLength, attentionHeadSize]`.
///
///   - `call(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray`
///     - Computes self-attention for the given hidden states.
///     - Steps:
///       1. Projects `hiddenStates` into `query`, `key`, and `value` representations.
///       2. Transposes them to prepare for attention score computation.
///       3. Computes scaled dot-product attention scores.
///       4. Applies the attention mask (if provided).
///       5. Computes attention probabilities using softmax and applies dropout.
///       6. Computes the weighted sum of value vectors.
///       7. Transposes and reshapes the output to match the original hidden state shape.
///       8. Passes the output through a dense layer and applies layer normalization.
///     - Parameters:
///       - `hiddenStates`: The input tensor of shape `[batchSize, seqLength, hiddenSize]`.
///       - `attentionMask`: An optional mask to prevent attending to certain positions.
///     - Returns: A tensor of shape `[batchSize, seqLength, hiddenSize]` containing the attention-weighted representations.

class AlbertSelfAttention: Module {
    let numAttentionHeads: Int
    let attentionHeadSize: Int
    let allHeadSize: Int

    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
//    var dropout: Dropout

    init(config: AlbertModelArgs) {
        self.numAttentionHeads = config.numAttentionHeads
        self.attentionHeadSize = config.hiddenSize / config.numAttentionHeads
        self.allHeadSize = self.numAttentionHeads * self.attentionHeadSize

        self._query.wrappedValue = Linear(config.hiddenSize, allHeadSize, zeroInitialized: true)
        self._key.wrappedValue = Linear(config.hiddenSize, allHeadSize, zeroInitialized: true)
        self._value.wrappedValue = Linear(config.hiddenSize, allHeadSize, zeroInitialized: true)
        self._dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, zeroInitialized: true)

        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
//        self.dropout = Dropout(p: config.attentionProbsDropoutProb)
        super.init()
    }

    func transposeForScores(_ x: MLXArray) -> MLXArray {
        let newShape = [x.shape[0], x.shape[1], numAttentionHeads, attentionHeadSize]
        return x.reshaped(newShape).transposed(axes: [0, 2, 1, 3])
    }

    func callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let mixedQueryLayer = query(hiddenStates)
        let mixedKeyLayer = key(hiddenStates)
        let mixedValueLayer = value(hiddenStates)

        let queryLayer = transposeForScores(mixedQueryLayer)
        let keyLayer = transposeForScores(mixedKeyLayer)
        let valueLayer = transposeForScores(mixedValueLayer)

        var attentionScores = MLX.matmul(queryLayer, keyLayer.transposed(axes: [0, 1, 3, 2]))
        attentionScores /= sqrt(Double(attentionHeadSize))

        if let attentionMask = attentionMask {
            attentionScores += attentionMask
        }

        let attentionProbs = MLX.softmax(attentionScores, axis: -1)
//        attentionProbs = dropout(attentionProbs)

        var contextLayer = MLX.matmul(attentionProbs, valueLayer)
        contextLayer = contextLayer.transposed(axes: [0, 2, 1, 3])
        let newContextLayerShape = [contextLayer.shape[0], contextLayer.shape[1], allHeadSize]
        contextLayer = contextLayer.reshaped(newContextLayerShape)

        contextLayer = dense(contextLayer)
        contextLayer = layerNorm(contextLayer + hiddenStates)

        return contextLayer
    }
}
