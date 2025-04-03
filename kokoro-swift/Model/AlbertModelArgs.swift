//
//  AlbertModelArgs.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Foundation

struct AlbertModelArgs {
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let modelType: String
    let embeddingSize: Int
    let innerGroupNum: Int
    let numHiddenGroups: Int
    let hiddenDropoutProb: Float
    let attentionProbsDropoutProb: Float
    let typeVocabSize: Int
    let initializerRange: Float
    let layerNormEps: Float
    let vocabSize: Int
    let dropout: Float

    init(
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        maxPositionEmbeddings: Int,
        modelType: String = "albert",
        embeddingSize: Int = 128,
        innerGroupNum: Int = 1,
        numHiddenGroups: Int = 1,
        hiddenDropoutProb: Float = 0.1,
        attentionProbsDropoutProb: Float = 0.1,
        typeVocabSize: Int = 2,
        initializerRange: Float = 0.02,
        layerNormEps: Float = 1e-12,
        vocabSize: Int = 30522,
        dropout: Float = 0.0
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.modelType = modelType
        self.embeddingSize = embeddingSize
        self.innerGroupNum = innerGroupNum
        self.numHiddenGroups = numHiddenGroups
        self.hiddenDropoutProb = hiddenDropoutProb
        self.attentionProbsDropoutProb = attentionProbsDropoutProb
        self.typeVocabSize = typeVocabSize
        self.initializerRange = initializerRange
        self.layerNormEps = layerNormEps
        self.vocabSize = vocabSize
        self.dropout = dropout
    }
}
