//
//  AlbertSelfOutput.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Foundation
import MLX
import MLXNN

/// `AlbertSelfOutput` processes the output of the self-attention mechanism in the ALBERT model.
/// It applies a linear transformation, dropout, and layer normalization to stabilize training.
///
/// - Properties:
///   - `dense`: A `Linear` layer that projects the attention output back to the hidden size.
///   - `layerNorm`: A `LayerNorm` module that normalizes the output with a residual connection.
///   - `dropout`: A `Dropout` layer that applies regularization to reduce overfitting.
///
/// - Methods:
///   - `call(hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray`
///     - Applies the transformation to the attention output.
///     - Steps:
///       1. Passes `hiddenStates` through the `dense` layer.
///       2. Applies dropout to prevent overfitting.
///       3. Adds a residual connection with `inputTensor` and normalizes the output.
///     - Parameters:
///       - `hiddenStates`: The output from the self-attention mechanism.
///       - `inputTensor`: The original input to the attention layer (used in residual connection).
///     - Returns: A tensor of shape `[batchSize, seqLength, hiddenSize]` containing the processed attention output.

class AlbertSelfOutput: Module {
    @ModuleInfo var dense: Linear
    @ModuleInfo var layerNorm: LayerNorm
    @ModuleInfo var dropout: Dropout

    init(config: AlbertModelArgs) {
        super.init()
        self.dense = Linear(config.hiddenSize, config.hiddenSize)
        self.layerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.dropout = Dropout(p: config.hiddenDropoutProb)
    }

    func callAsFunction(hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
        var output = dense(hiddenStates)
        output = dropout(output)
        output = layerNorm(output + inputTensor)
        return output
    }
}
