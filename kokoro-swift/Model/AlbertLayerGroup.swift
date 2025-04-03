//
//  AlbertLayerGroup.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertLayerGroup` is a container for multiple `AlbertLayer` instances.
/// It sequentially applies a group of transformer layers to the input hidden states.
///
/// - Properties:
///   - `albertLayers`: An array of `AlbertLayer` instances, determined by `innerGroupNum` in the model config.
///
/// - Initialization:
///   - `init(config: AlbertModelArgs)`
///     - Initializes the group with `innerGroupNum` layers of `AlbertLayer`.
///
/// - Methods:
///   - `callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray`
///     - Passes the `hiddenStates` through all layers in the group.
///     - If an `attentionMask` is provided, it is applied at each layer.
///     - Returns the transformed tensor after all layers have been applied.

class AlbertLayerGroup: Module {
    var albertLayers: [AlbertLayer]

    init(config: AlbertModelArgs) {
        self.albertLayers = (0..<config.innerGroupNum).map { _ in AlbertLayer(config: config) }
        super.init()
    }

    func callAsFunction(hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var output = hiddenStates
        for layer in albertLayers {
            output = layer(hiddenStates: output, attentionMask: attentionMask)
        }
        return output
    }
}

