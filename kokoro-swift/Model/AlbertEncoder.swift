//
//  AlbertEncoder.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN

class AlbertEncoder: Module {
    let config: AlbertModelArgs
    @ModuleInfo(key: "embedding_hidden_mapping_in") var embeddingHiddenMappingIn: Linear
    @ModuleInfo(key: "albert_layer_groups") var albertLayerGroups: [AlbertLayerGroup]

    init(config: AlbertModelArgs) {
        self.config = config
        self._embeddingHiddenMappingIn.wrappedValue = Linear(config.embeddingSize, config.hiddenSize, zeroInitialized: true)
        self._albertLayerGroups.wrappedValue = (0..<config.numHiddenGroups).map { _ in AlbertLayerGroup(config: config) }
        super.init()
    }

    func callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var output = embeddingHiddenMappingIn(hiddenStates)

        for i in 0..<config.numHiddenLayers {
            // Number of layers in a hidden group
            let layersPerGroup = config.numHiddenLayers / config.numHiddenGroups

            // Index of the hidden group
            let groupIdx = i / layersPerGroup

            output = albertLayerGroups[groupIdx](hiddenStates: output, attentionMask: attentionMask)
        }

        return output
    }
}
