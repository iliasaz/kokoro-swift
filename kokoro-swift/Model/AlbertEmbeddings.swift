//
//  AlbertEmbeddings.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertEmbeddings` is a module that constructs input embeddings for the ALBERT model,
/// incorporating word, position, and token type embeddings with layer normalization and dropout.
///
/// This module follows the embedding design used in the ALBERT (A Lite BERT) model, where
/// factorized embedding parameterization is applied to reduce memory consumption while
/// maintaining performance. Unlike BERT, ALBERT separates the size of hidden layers from
/// the size of embeddings, allowing for more efficient parameter usage.
///
/// This implementation is based on the paper:
/// **"ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"**
/// (Lan et al., 2019) - [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
///
/// - Properties:
///   - `wordEmbeddings`: An `Embedding` layer that maps input token IDs to dense word embeddings.
///   - `positionEmbeddings`: An `Embedding` layer that encodes positional information for each token.
///   - `tokenTypeEmbeddings`: An `Embedding` layer that differentiates between segment types (e.g., sentence A vs. B).
///   - `layerNorm`: A `LayerNorm` module that normalizes embeddings to stabilize training.
///   - `A dropout layer has been removed.
///
/// - Initialization:
///   - `init(config: AlbertModelArgs)`
///     - Initializes the embedding layers based on the given model configuration.
///     - Parameters:
///       - `config`: An instance of `AlbertModelArgs` containing hyperparameters such as embedding size, vocabulary size, and dropout probability.
///
/// - Methods:
///   - `call(inputIds: MLXArray, tokenTypeIds: MLXArray? = nil, positionIds: MLXArray? = nil) -> MLXArray`
///     - Computes embeddings for the given input token IDs.
///     - If `positionIds` is not provided, it is automatically generated based on sequence length.
///     - If `tokenTypeIds` is not provided, it defaults to a zero tensor.
///     - Returns a tensor containing the combined embeddings after applying layer normalization and dropout.

class AlbertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(config: AlbertModelArgs) {

        self._wordEmbeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.embeddingSize)
        self._positionEmbeddings.wrappedValue = Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: config.embeddingSize)
        self._tokenTypeEmbeddings.wrappedValue = Embedding(embeddingCount: config.typeVocabSize, dimensions: config.embeddingSize)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: config.embeddingSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(inputIds: MLXArray, tokenTypeIds: MLXArray? = nil, positionIds: MLXArray? = nil) -> MLXArray{
        let seqLength = inputIds.shape[1]
        let positionIds = positionIds ?? MLXArray.init(0 ..< seqLength)[.newAxis, 0...]
        let tokenTypeIds = tokenTypeIds ?? MLXArray.zeros(like: inputIds)

        let wordsEmbeddings = wordEmbeddings(inputIds)
        let positionEmbeddings = positionEmbeddings(positionIds)
        let tokenTypeEmbeddings = tokenTypeEmbeddings(tokenTypeIds)

        var embeddings = wordsEmbeddings + positionEmbeddings + tokenTypeEmbeddings
        embeddings = layerNorm(embeddings)

        return embeddings
    }
}
