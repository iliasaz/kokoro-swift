//
//  CustomAlbert.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN

/// `CustomAlbert` is a Swift implementation of the ALBERT (A Lite BERT) model,
/// designed for natural language processing tasks. It integrates embeddings,
/// an encoder, and a pooler to process input text data.
///
/// - Properties:
///   - `config`: An instance of `AlbertModelArgs` containing model configurations.
///   - `embeddings`: An `AlbertEmbeddings` module that converts input tokens into embeddings.
///   - `encoder`: An `AlbertEncoder` module that processes embeddings through transformer layers.
///   - `pooler`: A `Linear` layer that transforms the encoder's output for classification tasks.
///
/// - Initialization:
///   - `init(config: AlbertModelArgs)`
///     - Initializes the model's components based on the provided configuration.
///
/// - Methods:
///   - `callAsFunction(inputIds: MLXArray, tokenTypeIds: MLXArray? = nil, attentionMask: MLXArray? = nil) -> (MLXArray, MLXArray)`
///     - Processes input token IDs through the embeddings, encoder, and pooler.
///     - Parameters:
///       - `inputIds`: Token IDs representing the input text.
///       - `tokenTypeIds`: Optional segment IDs differentiating parts of the input.
///       - `attentionMask`: Optional mask indicating which tokens should be attended to.
///     - Returns: A tuple containing the sequence output and the pooled output.
///
///   - `sanitize(weights: [String: MLXArray]) -> [String: MLXArray]`
///     - Cleans the model's weights by removing unnecessary entries.
///     - Parameters:
///       - `weights`: A dictionary of model weight names and their corresponding arrays.
///     - Returns: A sanitized dictionary of weights without entries containing "position_ids".

class CustomAlbert: Module {
    let config: AlbertModelArgs
    @ModuleInfo(key: "embeddings") var embeddings: AlbertEmbeddings
    @ModuleInfo(key: "encoder") var encoder: AlbertEncoder
    @ModuleInfo(key: "pooler") var pooler: Linear

    init(config: AlbertModelArgs) {
        self.config = config
        self._embeddings.wrappedValue = AlbertEmbeddings(config: config)
        self._encoder.wrappedValue = AlbertEncoder(config: config)
        self._pooler.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, zeroInitialized: true)
        super.init()
    }

    func callAsFunction(inputIds: MLXArray, tokenTypeIds: MLXArray? = nil, attentionMask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let embeddingOutput = embeddings(inputIds: inputIds, tokenTypeIds: tokenTypeIds)

        var processedAttentionMask: MLXArray? = nil
        if let mask = attentionMask {
            processedAttentionMask = (1.0 - mask[0..., .newAxis, .newAxis, 0...]) * -10000.0
        }

        let sequenceOutput = encoder(hiddenStates: embeddingOutput, attentionMask: processedAttentionMask)
        let pooledOutput = MLX.tanh(pooler(sequenceOutput[0..., 0]))

        return (sequenceOutput, pooledOutput)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if !key.contains("position_ids") {
                sanitizedWeights[key] = value
            }
        }
        return sanitizedWeights
    }
}
