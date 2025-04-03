//
//  LSTM.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN

/// A bidirectional LSTM module that processes input sequences in both forward and backward directions.
///
/// This module uses two LSTM submodules (forwardLSTM and backwardLSTM) to compute hidden states:
/// - The forwardLSTM processes the input sequence as-is.
/// - The backwardLSTM processes a reversed copy of the input sequence (using stride indexing) and its output is reversed back.
///
/// The final output consists of:
/// 1. A concatenated tensor of forward and backward hidden states along the feature dimension.
/// 2. A tuple containing the final hidden and cell states for both directions:
///    - For the forward direction, the final state is taken from the last timestep.
///    - For the backward direction, the final state is taken from the first timestep.
///
/// Input:
/// - An MLXArray with shape `[N, L, D]` where:
///   - `N` is the batch size,
///   - `L` is the sequence length, and
///   - `D` is the feature dimension.
///
/// Usage:
/// ```swift
/// let (output, ((forwardHidden, forwardCell), (backwardHidden, backwardCell))) = bidirectionalLSTM(input)
/// ```
///
/// This structure is compatible with downstream modules that expect both the concatenated sequence output and the final states.

class BidirectionalLSTM: Module {
    @ModuleInfo var forwardLSTM: LSTM
    @ModuleInfo var backwardLSTM: LSTM

    init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.forwardLSTM = LSTM(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
        self.backwardLSTM = LSTM(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
        // Forward pass
        let (forwardOutput, forwardCellStates) = forwardLSTM(x)
        let forwardHiddenStates = forwardOutput

        // Reverse the input for backward pass
        let reversedX = x[0..., .stride(by: -1), 0...]
        let (backwardOutput, backwardCellStates) = backwardLSTM(reversedX)

        // Reverse the backward output to align with forward sequence
        let backwardHiddenStates = backwardOutput[0..., .stride(by: -1), 0...]

        // Concatenate forward and backward hidden states along the feature dimension
        let output = concatenated([forwardHiddenStates, backwardHiddenStates], axis: -1)

        // Extract the final hidden and cell states
        let forwardFinalHidden = forwardHiddenStates[-1]
        let forwardFinalCell = forwardCellStates[-1]
        let backwardFinalHidden = backwardHiddenStates[0]
        let backwardFinalCell = backwardCellStates[0]

        return (output, ((forwardFinalHidden, forwardFinalCell), (backwardFinalHidden, backwardFinalCell)))
    }
}
