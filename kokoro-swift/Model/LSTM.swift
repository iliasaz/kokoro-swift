//
//  LSTM.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/11/25.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

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

//class BidirectionalLSTM: Module {
//    @ModuleInfo(key: "forward") var forwardLSTM: LSTM
//    @ModuleInfo(key: "backward") var backwardLSTM: LSTM
//
//    init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
//        self._forwardLSTM.wrappedValue = LSTM(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
//        self._backwardLSTM.wrappedValue = LSTM(inputSize: inputSize, hiddenSize: hiddenSize, bias: bias)
//        super.init()
//    }
//
//    func callAsFunction(_ x: MLXArray) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
//        let input: MLXArray
//        if x.ndim == 2 {
//          input = x.expandedDimensions(axis: 0) // (1, seq_len, input_size)
//        } else {
//          input = x
//        }
//        // Forward pass
//        let (forwardOutput, forwardCellStates) = forwardLSTM(input)
//        let forwardHiddenStates = forwardOutput
//        // Reverse the input for backward pass
//        let reversedX = input[0..., .stride(by: -1), 0...]
//        let (backwardOutput, backwardCellStates) = backwardLSTM(reversedX)
//
//        // Reverse the backward output to align with forward sequence
//        let backwardHiddenStates = backwardOutput[0..., .stride(by: -1), 0...]
//
//        // Concatenate forward and backward hidden states along the feature dimension
//        let output = concatenated([forwardHiddenStates, backwardHiddenStates], axis: -1)
//
//        // Extract the final hidden and cell states
//        let forwardFinalHidden = forwardHiddenStates[-1]
//        let forwardFinalCell = forwardCellStates[-1]
//        let backwardFinalHidden = backwardHiddenStates[0]
//        let backwardFinalCell = backwardCellStates[0]
//
//        return (output, ((forwardFinalHidden, forwardFinalCell), (backwardFinalHidden, backwardFinalCell)))
//    }
//}

class BidirectionalLSTM: Module {
    /// A Bidirectional LSTM  layer.
    ///
    /// The input has shape `NLD` or `LD` where:
    ///
    /// * `N` is the optional batch dimension
    /// * `L` is the sequence length
    /// * `D` is the input's feature dimension
    ///
    /// The hidden state `h` and cell `c` have shape `NH` or `H`, depending on
    /// whether the input is batched or not. Returns the hidden state and cell state at each
    /// time step, of shape `NLH` or `LH`.
    ///
    /// - Parameters:
    ///   - inputSize: dimension of the input, `D`
    ///   - hiddenSize: dimension of the hidden state, `H`
    ///   - bias: if `true` use a bias


    /// forward direction
    @ParameterInfo(key: "weight_ih_l0") public var wxForward: MLXArray
    @ParameterInfo(key: "weight_hh_l0") public var whForward: MLXArray
    @ParameterInfo(key: "bias_ih_l0") public var biasIhForward: MLXArray?
    @ParameterInfo(key: "bias_hh_l0") public var biasHhForward: MLXArray?

    /// backward direction
    @ParameterInfo(key: "weight_ih_l0_reverse") public var wxBackward: MLXArray
    @ParameterInfo(key: "weight_hh_l0_reverse") public var whBackward: MLXArray
    @ParameterInfo(key: "bias_ih_l0_reverse") public var biasIhBackward: MLXArray?
    @ParameterInfo(key: "bias_hh_l0_reverse") public var biasHhBackward: MLXArray?

    private enum Direction {
        case forward, backward
    }

    public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        let scale = 1 / sqrt(Float(hiddenSize))
        self._wxForward.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, inputSize])
        self._whForward.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        self._wxBackward.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, inputSize])
        self._whBackward.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, hiddenSize])

        if bias {
            self._biasIhForward.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
            self._biasHhForward.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
            self._biasIhBackward.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
            self._biasHhBackward.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        } else {
            self._biasIhForward.wrappedValue = nil
            self._biasHhForward.wrappedValue = nil
            self._biasIhBackward.wrappedValue = nil
            self._biasHhBackward.wrappedValue = nil
        }
    }

    // one function for both forward and backward directions
    private func pass(
        _ direction: Direction,
        _ x: MLXArray,
        wx: MLXArray,
        wh: MLXArray,
        biasIh: MLXArray?,
        biasHh: MLXArray?,
        hidden: MLXArray? = nil,
        cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var x = x

        if let biasIh = biasIh, let biasHh = biasHh {
            x = MLX.addMM(biasIh + biasHh,x, wx.T)
        } else {
            x = MLX.matmul(x, wx.T)
        }

        var hidden: MLXArray! = hidden
        var cell: MLXArray! = cell
        var allHidden = [MLXArray]()
        var allCell = [MLXArray]()

        let indexRange: AnySequence<Int> = switch direction {
            case .forward:
                AnySequence(0 ..< x.dim(-2))
            case .backward:
                AnySequence(stride(from: x.dim(-2) - 1, through: 0, by: -1))
        }

        for index in indexRange {
            var ifgo = x[.ellipsis, index, 0...]
            if hidden != nil {
                ifgo = MLX.addMM(ifgo, hidden, wh.T)
            }

            // Split gates
            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            if cell != nil {
                cell = f * cell + i * g
            } else {
                cell = i * g
            }
            hidden = o * MLX.tanh(cell)

            switch direction {
                case .forward:
                    allCell.append(cell)
                    allHidden.append(hidden)
                case .backward:
                    // Insert at beginning to maintain original sequence order
                    allCell.insert(cell, at: 0)
                    allHidden.insert(hidden, at: 0)
            }
        }

        return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
    }

    func callAsFunction(
        _ x: MLXArray,
    ) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
        let input: MLXArray
        if x.ndim == 2 {
            input = x.expandedDimensions(axis: 0) // (1, seq_len, input_size)
        } else {
            input = x
        }

        let (forwardHidden, forwardCell) = pass(.forward, input, wx: wxForward, wh: whForward, biasIh: biasIhForward, biasHh: biasHhForward)
        let (backwardHidden, backwardCell) = pass(.backward, input, wx: wxBackward, wh: whBackward, biasIh: biasIhBackward, biasHh: biasHhBackward)
        let output = MLX.concatenated([forwardHidden, backwardHidden], axis: -1)

        return (
            output,
            (
                (forwardHidden[0..., -1, 0...], forwardCell[0..., -1, 0...]),
                (backwardHidden[0..., 0, 0...], backwardCell[0..., 0, 0...])
            )
        )
    }
}
