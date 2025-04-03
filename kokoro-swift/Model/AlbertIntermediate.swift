//
//  AlbertIntermediate.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertIntermediate` is the feed-forward transformation layer in the ALBERT model.
/// It applies a linear transformation followed by a GELU activation function.
///
/// - Properties:
///   - `dense`: A `Linear` layer that projects the input from `hiddenSize` to `intermediateSize`.
///   - `intermediateActFn`: A `GELU` activation function that introduces non-linearity.
///
/// - Methods:
///   - `call(hiddenStates: MLXArray) -> MLXArray`
///     - Applies the feed-forward transformation to the input tensor.
///     - Steps:
///       1. Passes `hiddenStates` through the `dense` layer.
///       2. Applies the `GELU` activation function.
///     - Parameters:
///       - `hiddenStates`: The input tensor from the self-attention output.
///     - Returns: A tensor of shape `[batchSize, seqLength, intermediateSize]` containing transformed hidden states.

class AlbertIntermediate: Module {
    @ModuleInfo var dense: Linear
    @ModuleInfo var intermediateActFn: GELU

    init(config: AlbertModelArgs) {
        super.init()
        self.dense = Linear(config.hiddenSize, config.intermediateSize)
        self.intermediateActFn = GELU()
    }

    func callAsFunction(hiddenStates: MLXArray) -> MLXArray {
        var output = dense(hiddenStates)
        output = intermediateActFn(output)
        return output
    }
}
