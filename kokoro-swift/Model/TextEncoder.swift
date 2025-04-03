//
//  TextEncoder.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// `TextEncoder` is a neural network module that encodes text input into a continuous representation.
/// It combines **embedding lookup**, **convolutional feature extraction**, and **bi-directional LSTM modeling**
/// to create an effective sequence encoding for downstream tasks such as text-to-speech and sequence modeling.
///
/// ### **Architecture Overview**
/// - **Embedding Layer**: Converts token indices into dense feature vectors.
/// - **Convolutional Blocks**: Extracts **local dependencies** via stacked Conv1D layers with weight normalization.
/// - **Bidirectional LSTM**: Captures **long-term dependencies** by processing sequences in both directions.
/// - **Masking & Padding**: Handles variable-length inputs by masking invalid regions.
///
/// ### **Processing Pipeline**
/// 1. **Embeds input** → Converts token indices into dense vectors.
/// 2. **Applies CNN layers** → Extracts local patterns with Conv1D + LayerNorm + Activation + Dropout.
/// 3. **Processes with Bi-LSTM** → Captures long-range dependencies.
/// 4. **Pads output** → Ensures uniform sequence lengths.
/// 5. **Reapplies mask** → Prevents computation on invalid positions.
///
/// ### **Reference**
/// This model structure is inspired by **Tacotron-style** text encoders used in speech synthesis.
///
/// - **Properties**:
///   - `embedding`: Converts input token indices to dense vectors.
///   - `cnn`: A stack of `ConvBlock` modules for local feature extraction.
///   - `lstm`: A `BidirectionalLSTM` for sequence modeling.
///
/// - **Initialization**:
///   - `init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int, actv: @escaping (MLXArray) -> MLXArray = MLXNN.leakyRelu($0, negativeSlope: 0.2))`
///     - Configures embedding, CNN layers, and LSTM for text encoding.
///     - Parameters:
///       - `channels`: Number of feature channels.
///       - `kernelSize`: Size of the convolutional kernel.
///       - `depth`: Number of convolutional layers.
///       - `nSymbols`: Vocabulary size.
///       - `actv`: Activation function (default: LeakyReLU).
///
/// - **Methods**:
///   - `callAsFunction(x: MLXArray, inputLengths: MLXArray, m: MLXArray) -> MLXArray`
///     - Encodes input sequences into feature representations.
///     - Parameters:
///       - `x`: Input tensor of shape `[batch, seq_len]` (token indices).
///       - `inputLengths`: Sequence lengths before padding.
///       - `m`: Mask tensor indicating valid positions.
///     - Returns: Encoded feature tensor of shape `[batch, channels, max_seq_len]`.

/// A convolutional block that consists of weight-normalized Conv1D, LayerNorm, an activation, and dropout.
class ConvBlock: Module {
    @ModuleInfo var conv: ConvWeighted
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var dropout: Dropout
    let activation: (MLXArray) -> MLXArray  // Activation function

    init(channels: Int, kernelSize: Int, padding: Int, activation: @escaping (MLXArray) -> MLXArray) {
        self.conv = ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize, stride: 1, padding: padding)
        self.norm = LayerNorm(dimensions: channels)
        self.dropout = Dropout(p: 0.2)
        self.activation = activation
        super.init()
    }

    /// Forward pass for ConvBlock
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x.transposed(axes: [0, 2, 1])  // Swap to (batch, seq_len, channels) for Conv1D
        x = conv(x: x, conv: MLX.conv1d)  // Apply weight-normalized convolution
        x = x.transposed(axes: [0, 2, 1])  // Swap back to (batch, channels, seq_len)
        x = norm(x)  // Apply layer normalization
        x = activation(x)  // Apply activation function
        return dropout(x)  // Apply dropout for regularization
    }
}

class TextEncoder: Module {
    @ModuleInfo var embedding: Embedding
    @ModuleInfo var cnn: [ConvBlock]
    @ModuleInfo var lstm: BidirectionalLSTM

    init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int, actv: @escaping (MLXArray) -> MLXArray = { MLXNN.leakyRelu($0, negativeSlope: 0.2) }) {
        super.init()

        self.embedding = Embedding(embeddingCount: nSymbols, dimensions: channels)
        let padding = (kernelSize - 1) / 2

        // Construct CNN blocks with weight-normalized convolutions
        self.cnn = (0..<depth).map { _ in
            ConvBlock(channels: channels, kernelSize: kernelSize, padding: padding, activation: actv)
        }

        self.lstm = BidirectionalLSTM(inputSize: channels, hiddenSize: channels / 2)
    }

    func callAsFunction(x: MLXArray, inputLengths: MLXArray, m: MLXArray) -> MLXArray {
        // Step 1: Embedding Lookup
        var x = embedding(x)  // Convert input IDs into dense embeddings (batch, seq_len, channels)
        x = x.transposed(axes: [0, 2, 1])  // Swap to (batch, channels, seq_len)

        // Step 2: Apply Mask (Zero out masked positions)
        let mExpanded = m.expandedDimensions(axis: 1)  // Expand mask for broadcasting
        x = MLX.where(mExpanded, 0.0, x)  // Zero out masked positions

        // Step 3: Pass Through Convolutional Blocks
        for convBlock in cnn {
            x = convBlock(x)  // Apply CNN transformation
            x = MLX.where(mExpanded, 0.0, x)  // Reapply mask to maintain zeroing
        }

        // Step 4: Prepare for LSTM (Swap sequence and channel dimensions)
        x = x.transposed(axes: [0, 2, 1])  // Convert to (batch, seq_len, channels) for LSTM

        // Step 5: Apply LSTM and Extract Hidden States
        x = lstm(x).0  // Extract hidden states (ignore cell state)

        // Step 6: Restore Shape After LSTM
        x = x.transposed(axes: [0, 2, 1])  // Convert back to (batch, channels, seq_len)

        // Step 7: Pad Output to Maximum Sequence Length
        let xPad = MLX.zeros([x.shape[0], x.shape[1], m.shape.last!])  // Initialize zero tensor
        let sliceRange = 0..<x.shape.last!  // Get valid sequence range
        xPad[0..., 0..., sliceRange] = x  // Copy valid `x` into padded tensor

        // Step 8: Apply Mask Again (Zero out invalid padded regions)
        return MLX.where(mExpanded, 0.0, xPad)
    }
}
