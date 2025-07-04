//
//  AlbertOutput.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertOutput` is the final transformation layer in an ALBERT transformer block.
/// It applies a linear transformation, dropout, and layer normalization to the intermediate outputs.
///
/// - Properties:
///   - `dense`: A `Linear` layer that projects the input from `intermediateSize` back to `hiddenSize`.
///   - `layerNorm`: A `LayerNorm` module that normalizes the output with a residual connection.
///   - `dropout`: A `Dropout` layer that applies regularization to prevent overfitting.
///
/// - Methods:
///   - `call(hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray`
///     - Processes the output from the `AlbertIntermediate` layer.
///     - Steps:
///       1. Applies a `dense` transformation to the `hiddenStates`.
///       2. Applies dropout for regularization.
///       3. Adds a residual connection with `inputTensor` and normalizes the output.
///     - Parameters:
///       - `hiddenStates`: The transformed output from the `AlbertIntermediate` layer.
///       - `inputTensor`: The original hidden states (used for the residual connection).
///     - Returns: A tensor of shape `[batchSize, seqLength, hiddenSize]` containing the final processed output.

class AlbertOutput: Module {
    @ModuleInfo var dense: Linear
    @ModuleInfo var layerNorm: LayerNorm
//    @ModuleInfo var dropout: Dropout

    init(config: AlbertModelArgs) {
        super.init()
        self.dense = Linear(config.intermediateSize, config.hiddenSize, zeroInitialized: true)
        self.layerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
//        self.dropout = Dropout(p: config.hiddenDropoutProb)
    }

    func callAsFunction(hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
        var output = dense(hiddenStates)
//        output = dropout(output)
        output = layerNorm(output + inputTensor)
        return output
    }
}
