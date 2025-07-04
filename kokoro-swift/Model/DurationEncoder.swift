//
//  DurationEncoder.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// `DurationEncoder` is a neural network module that encodes duration-related representations.
/// It combines **bi-directional LSTM modeling** and **adaptive normalization (AdaLayerNorm)**
/// to learn temporal dependencies in text-based sequence data.
///
/// ### **Architecture Overview**
/// - **LSTM Layers with AdaLayerNorm**: A stack of LSTMs followed by adaptive normalization.
/// - **Style Conditioning**: Injects style information into each layer to control sequence representation.
/// - **Masking & Padding**: Handles variable-length sequences by applying masks and zero-padding.
///
/// ### **Processing Pipeline**
/// 1. **Prepare inputs** → Swap axes to match MLX expectations.
/// 2. **Broadcast style vectors** → Ensure `style` input aligns with `x`.
/// 3. **Concatenate `x` and `style`** → Injects style features into the sequence representation.
/// 4. **Apply Masking** → Zero out padded regions before computation.
/// 5. **Pass through `nLayers` of LSTMs + AdaLayerNorm** → Processes sequential dependencies.
/// 6. **Return encoded duration representation** → Shape `[batch, seq_len, channels]`.
///
/// ### **Reference**
/// This architecture follows approaches used in **prosody modeling** and **duration prediction**
/// in text-to-speech and sequence modeling.
///
/// - **Properties**:
///   - `lstmLayers`: A stack of `LSTMLayerWithAdaNorm`, each containing an LSTM and AdaLayerNorm.
///   - `dropout`: Dropout rate for regularization.
///   - `dModel`: Dimensionality of the LSTM hidden state.
///   - `styDim`: Dimensionality of the style vector.
///
/// - **Initialization**:
///   - `init(styDim: Int, dModel: Int, nLayers: Int, dropout: Float = 0.1)`
///     - Configures a stack of LSTM layers with adaptive normalization.
///     - Parameters:
///       - `styDim`: Style embedding dimensionality.
///       - `dModel`: Number of hidden units in the LSTM.
///       - `nLayers`: Number of stacked LSTM layers.
///       - `dropout`: Dropout rate for regularization (default: `0.1`).
///
/// - **Methods**:
///   - `callAsFunction(x: MLXArray, style: MLXArray, textLengths: MLXArray, mask: MLXArray) -> MLXArray`
///     - Encodes the input sequence into a duration-aware representation.
///     - Parameters:
///       - `x`: Input tensor of shape `[batch, seq_len, channels]`.
///       - `style`: Style vector for conditioning.
///       - `textLengths`: Lengths of input sequences before padding.
///       - `mask`: Mask tensor indicating valid positions.
///     - Returns: Encoded feature tensor of shape `[batch, seq_len, channels]`.


/// A single layer of LSTM followed by AdaLayerNorm.
//class LSTMLayerWithAdaNorm: Module {
//    @ModuleInfo var lstm: BidirectionalLSTM
//    @ModuleInfo var adaNorm: AdaLayerNorm
//
//    init(styDim: Int, dModel: Int) {
//        self.lstm = BidirectionalLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2)
//        self.adaNorm = AdaLayerNorm(styleDim: styDim, channels: dModel)
//        super.init()
//    }
//
//    /// Forward pass of LSTM layer followed by AdaLayerNorm.
//    func callAsFunction(x: MLXArray, style: MLXArray, mask: MLXArray) -> MLXArray {
//        var x = x
//        // Apply LSTM
//        x = x.transposed(axes: [0, 2, 1])[0] // Convert to (batch, seq_len, channels)
//        x = lstm(x).0 // Extract hidden states
//        x = x.transposed(axes: [0, 2, 1]) // Restore shape
//
//        // Pad x to match mask size
//        let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape.last!])
//        let sliceRange = 0..<x.shape.last!
//        xPad[0..., 0..., sliceRange] = x
//        x = xPad
//
//        // Apply AdaLayerNorm
//        x = adaNorm(x: x.transposed(axes: [0, 2, 1]), s: style).transposed(axes: [0, 2, 1]) // Normalize & restore shape
//
//        // Broadcast `style` to match `x`'s shape
//        let expandedStyle = MLX.broadcast(style, to: [x.shape.last!, x.shape[0], style.shape[1]])
////        print("x: \(x.shape), mask: \(mask.shape), style: \(style.shape), expandedStyle: \(expandedStyle.shape)")
//
//        // Concatenate with style tensor
//        x = MLX.concatenated([x, expandedStyle.transposed(1, 2, 0)], axis: 1)
//
//        // Apply mask
//        let maskExpanded = mask.expandedDimensions(axis: -1).transposed(axes: [0, 2, 1])
//        x = MLX.where(maskExpanded, MLXArray.zeros(like: x), x)
//
//        return x
//    }
//}

//class DurationEncoder: Module {
//    @ModuleInfo(key: "lstms") var lstmLayers: [LSTMLayerWithAdaNorm]
//    let dropout: Float
//    let dModel: Int
//    let styDim: Int
//
//    init(styDim: Int, dModel: Int, nLayers: Int, dropout: Float = 0.1) {
//        self.dropout = dropout
//        self.dModel = dModel
//        self.styDim = styDim
//        super.init()
//        self.lstmLayers = (0..<nLayers).map { _ in LSTMLayerWithAdaNorm(styDim: styDim, dModel: dModel) }
//    }
//
//    func callAsFunction(x: MLXArray, style: MLXArray, textLengths: MLXArray, mask: MLXArray) -> MLXArray {
//        print("x: \(x.shape)")
//        // Step 1: Prepare Inputs
//        var x = x.transposed(axes: [2, 0, 1]) // Convert to (seq_len, batch, channels)
//
//        // Broadcast `style` to match `x`'s shape
//        let expandedStyle = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape.last!])
//        // Concatenate `x` and `s` along feature dimension
//        x = MLX.concatenated([x, expandedStyle], axis: -1)
//        // Apply mask (ensure padded regions are zeroed out)
//        let maskExpanded = mask.expandedDimensions(axis: -1).transposed(axes: [1, 0, 2])
//        x = MLX.where(maskExpanded, 0.0, x)
//        // Step 2: Pass Through LSTM + AdaLayerNorm Layers
//        x = x.transposed(axes: [1, 2, 0]) // Convert to (batch, channels, seq_len)
//        for layer in lstmLayers {
//            x = layer(x: x, style: style, mask: mask) // Process through LSTM + AdaLayerNorm
//        }
//        x = x.transposed(axes: [0, 2, 1]) // Restore shape (batch, seq_len, channels)
//        print("x: \(x.shape)")
//
//        return x
//    }
//}

class DurationEncoder: Module {
    @ModuleInfo(key: "lstms") var lstmLayers: [Module]
//    let dropout: Float
    let dModel: Int
    let styDim: Int

    init(styDim: Int, dModel: Int, nLayers: Int, dropout: Float = 0.1) {
//        self.dropout = dropout
        self.dModel = dModel
        self.styDim = styDim
        var layers: [Module] = []
        for _ in 0 ..< nLayers {
            layers.append(contentsOf: [
                BidirectionalLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2),
                AdaLayerNorm(styleDim: styDim, channels: dModel)
            ])
        }
        self._lstmLayers.wrappedValue = layers
        super.init()
    }

    func callAsFunction(x: MLXArray, style: MLXArray, textLengths: MLXArray, mask: MLXArray) -> MLXArray {
        // Step 1: Prepare Inputs
        var x = x.transposed(axes: [2, 0, 1]) // Convert to (seq_len, batch, channels)

        // Broadcast `style` to match `x`'s shape
        let expandedStyle = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape.last!])
        // Concatenate `x` and `s` along feature dimension
        x = MLX.concatenated([x, expandedStyle], axis: -1)
        // Apply mask (ensure padded regions are zeroed out)
        let maskExpanded = mask.expandedDimensions(axis: -1).transposed(axes: [1, 0, 2])
        x = MLX.where(maskExpanded, 0.0, x)

        // Step 2: Pass Through LSTM + AdaLayerNorm Layers
        x = x.transposed(axes: [1, 2, 0]) // Convert to (batch, channels, seq_len)
        for layer in lstmLayers {
            if let lstm = layer as? BidirectionalLSTM {
                x = x.transposed(axes: [0, 2, 1])[0] // Convert to (batch, seq_len, channels)
                x = lstm(x).0 // Extract hidden states
                x = x.transposed(axes: [0, 2, 1]) // Restore shape

                // Pad x to match mask size
                let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape.last!])
                let sliceRange = 0..<x.shape.last!
                xPad[0..., 0..., sliceRange] = x
                x = xPad
            } else if let adaLayerNorm = layer as? AdaLayerNorm {
                // Apply AdaLayerNorm
                x = adaLayerNorm(x: x.transposed(axes: [0, 2, 1]), s: style).transposed(axes: [0, 2, 1]) // Normalize & restore shape

                // Broadcast `style` to match `x`'s shape
                let expandedStyle = MLX.broadcast(style, to: [x.shape.last!, x.shape[0], style.shape[1]])

                // Concatenate with style tensor
                x = MLX.concatenated([x, expandedStyle.transposed(1, 2, 0)], axis: 1)

                // Apply mask
                let maskExpanded = mask.expandedDimensions(axis: -1).transposed(axes: [0, 2, 1])
                x = MLX.where(maskExpanded, MLXArray.zeros(like: x), x)
            }

        }
        x = x.transposed(axes: [0, 2, 1]) // Restore shape (batch, seq_len, channels)

        return x
    }
}
