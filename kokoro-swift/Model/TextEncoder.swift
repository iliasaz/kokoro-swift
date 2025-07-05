//
//  TextEncoder.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN
import MLXFast

/// The original Kokoro implementation uses a customer LayerNorm layer, which is basically the same as the defualt one
/// except weight is called gamma, and bias is called beta.
/// Hence, the parameters in the safetensors file have keys (gamma, beta)
/// We could re-key the parameters, but this would require and extra processing time during the weight loading, which we want to avoid
///
class CustomLayerNorm: Module, UnaryLayer {

    let dimensions: Int
    let eps: Float

    @ParameterInfo var gamma: MLXArray?
    @ParameterInfo var beta: MLXArray?

    /// Applies layer normalization [1] on the inputs.
    ///
    /// See [LayerNorm python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LayerNorm.html) for more information.
    ///
    /// ### References
    /// 1. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
    ///
    /// - Parameters:
    ///   - dimensions: number of features in the input
    ///   - eps: value added to the denominator for numerical stability
    ///   - affine: if `true` adds a trainable `weight` and `bias`
    public init(dimensions: Int, eps: Float = 1e-5, affine: Bool = true) {
        self.dimensions = dimensions
        self.eps = eps

        if affine {
            self.gamma = MLXArray.ones([dimensions])
            self.beta = MLXArray.zeros([dimensions])
        } else {
            self.gamma = nil
            self.beta = nil
        }
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: gamma, bias: beta, eps: eps)
    }
}

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

class TextEncoder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "cnn") var cnn: [[Module]]
    @ModuleInfo(key: "lstm") var lstm: BidirectionalLSTM

    init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int, actv: Module = LeakyReLU(negativeSlope: 0.2)) {
        self._embedding.wrappedValue = Embedding(embeddingCount: nSymbols, dimensions: channels)
        let padding = (kernelSize - 1) / 2

        // Construct CNN blocks with weight-normalized convolutions
        var cnnLayers = [[Module]]()
        for _ in 0 ..< depth {
            cnnLayers.append([
                ConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize, stride: 1, padding: padding),
                CustomLayerNorm(dimensions: channels),
                actv
            ])
        }

        self._cnn.wrappedValue = cnnLayers
        self._lstm.wrappedValue = BidirectionalLSTM(inputSize: channels, hiddenSize: channels / 2)
        super.init()
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
            for layer in convBlock {
                if layer is ConvWeighted || layer is CustomLayerNorm {
                    x = MLX.swappedAxes(x, 2, 1)
                    if let conv = layer as? ConvWeighted {
                        x = conv(x: x, conv: MLX.conv1d)
                    } else if let norm = layer as? CustomLayerNorm {
                        x = norm(x)
                    }
                    x = MLX.swappedAxes(x, 2, 1)
                } else if let activation = layer as? LeakyReLU {
                    x = activation(x)
                } else {
                    fatalError("Unsupported layer type: \(type(of: layer))")
                }
                x = MLX.where(mExpanded, 0.0, x)  // Reapply mask to maintain zeroing
            }
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
