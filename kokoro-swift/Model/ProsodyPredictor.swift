//
//  ProsodyPredictor.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/12/25.
//

import Foundation
import MLX
import MLXNN

/// A modular branch used for both `F0` and `N` processing pipelines.
///
/// This class applies a sequence of `AdainResBlk1d` layers followed by a
/// convolutional projection to process input features for either `F0` (pitch)
/// or `N` (noise). The architecture is identical for both branches but serves
/// distinct purposes:
/// - `F0Branch`: Predicts the fundamental frequency (pitch) of speech.
/// - `NBranch`: Models noise components in speech, such as breathiness.
///
/// While structurally identical, separating these branches allows for
/// independent training, ensuring that each learns features specific to
/// either pitch or noise without interference.

class ProsodyBranch: Module {
    @ModuleInfo var layers: [AdainResBlk1d]
    @ModuleInfo var projection: Conv1d

    init(inChannels: Int, outChannels: Int, styleDim: Int, dropout: Float, upsample: Bool = false) {
        self.layers = [
            AdainResBlk1d(dimIn: inChannels, dimOut: inChannels, styleDim: styleDim, dropoutP: dropout),
            AdainResBlk1d(dimIn: inChannels, dimOut: outChannels, styleDim: styleDim, upsample: upsample, dropoutP: dropout),
            AdainResBlk1d(dimIn: outChannels, dimOut: outChannels, styleDim: styleDim, dropoutP: dropout)
        ]
        self.projection = Conv1d(inputChannels: outChannels, outputChannels: 1, kernelSize: 1, padding: 0)
        super.init()
    }

    /// Applies the AdainResBlk1d transformations and projects the output.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape (batch, sequence_length, channels).
    ///   - style: Style conditioning tensor.
    /// - Returns: Processed tensor after applying the layers and projection.
    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x, s: style)
        }
        x = x.transposed(axes: [0, 2, 1])
        x = projection(x)
        return x.transposed(axes: [0, 2, 1]).squeezed(axis: 1)
    }
}

/// The main module for prosody prediction, handling duration, pitch (`F0`), and noise (`N`).
///
/// `ProsodyPredictor` consists of:
/// - **Text Encoding** (`textEncoder`): Processes input text sequences.
/// - **Duration Modeling** (`lstm`, `durationProj`): Predicts phoneme durations.
/// - **Pitch & Noise Prediction** (`f0Branch`, `nBranch`): Predicts `F0` (pitch) and `N` (noise).
///
/// The `callAsFunction` method handles duration prediction, while `f0nTrain` is
/// used during training to predict `F0` and `N` separately.

class ProsodyPredictor: Module {
    @ModuleInfo var textEncoder: DurationEncoder
    @ModuleInfo var lstm: BidirectionalLSTM
    @ModuleInfo var durationProj: Linear
    @ModuleInfo var sharedLSTM: BidirectionalLSTM
    @ModuleInfo var f0Branch: ProsodyBranch
    @ModuleInfo var nBranch: ProsodyBranch
    @ModuleInfo var dropout: Dropout

    init(styleDim: Int, dHid: Int, nLayers: Int, maxDur: Int = 50, dropout: Float = 0.1) {
        self.textEncoder = DurationEncoder(styDim: styleDim, dModel: dHid, nLayers: nLayers, dropout: dropout)
        self.lstm = BidirectionalLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)
        self.durationProj = Linear(dHid, maxDur, bias: true)
        self.sharedLSTM = BidirectionalLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)

        // Create F0 and N branches using ProsodyBranch
        self.f0Branch = ProsodyBranch(inChannels: dHid, outChannels: dHid / 2, styleDim: styleDim, dropout: dropout, upsample: true)
        self.nBranch = ProsodyBranch(inChannels: dHid, outChannels: dHid / 2, styleDim: styleDim, dropout: dropout, upsample: true)
        self.dropout = Dropout(p: 0.5)
        super.init()
    }

    /// Predicts phoneme durations and energy alignment for a given text input.
    ///
    /// - Parameters:
    ///   - texts: Input text sequence tensor.
    ///   - style: Style conditioning tensor.
    ///   - textLengths: Lengths of text sequences.
    ///   - alignment: Alignment matrix for attention modeling.
    ///   - m: Masking tensor for padding handling.
    /// - Returns: Tuple containing predicted phoneme durations and energy alignment.
    func callAsFunction(texts: MLXArray, style: MLXArray, textLengths: MLXArray, alignment: MLXArray, m: MLXArray) -> (MLXArray, MLXArray) {
        // Step 1: Encode Text with Style
        var d = textEncoder(x: texts, style: style, textLengths: textLengths, mask: m)

        // Step 2: Process with LSTM
        d = lstm(d).0 // Extract hidden states
        d = dropout(d) // Apply dropout during inference

        // Step 3: Predict Duration
        let duration = durationProj(d).squeezed(axis: -1)

        // Step 4: Compute Energy (en) from alignment matrix
        let en = MLX.matmul(d.transposed(axes: [0, 2, 1]), alignment)

        return (duration, en)
    }

    /// Trains the model to predict `F0` (pitch) and `N` (noise) components.
    ///
    /// - Parameters:
    ///   - x: Input feature tensor of shape (batch, sequence_length, hidden_dim).
    ///   - s: Style conditioning tensor.
    /// - Returns: Tuple containing `F0` and `N` predictions.
    func f0nTrain(x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
        var x = x.transposed(axes: [0, 2, 1]) // Swap to (batch, seq_len, channels)
        x = sharedLSTM(x).0 // Extract hidden states

        // Process through F0 and N branches
        let f0 = f0Branch(x, style: s)
        let n = nBranch(x, style: s)

        return (f0, n)
    }
}
