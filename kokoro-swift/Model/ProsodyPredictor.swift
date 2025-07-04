////
////  ProsodyPredictor.swift
////  kokoro-swift
////
////  Created by Ilia Sazonov on 3/12/25.
////
//
import Foundation
import MLX
import MLXNN

/// The main module for prosody prediction, handling duration, pitch (`F0`), and noise (`N`).
///
/// `ProsodyPredictor` consists of:
/// - **Text Encoding** (`textEncoder`): Processes input text sequences.
/// - **Duration Modeling** (`lstm`, `durationProj`): Predicts phoneme durations.
/// - **Pitch & Noise Prediction** (`F0`, `N`): Predicts `F0` (pitch) and `N` (noise).
///
/// The `callAsFunction` method handles duration prediction, while `f0nTrain` is
/// used during training to predict `F0` and `N` separately.

class DummyLinear: Module {
    @ModuleInfo(key: "linear_layer") var linear: Linear

    init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) {
        self._linear.wrappedValue = Linear(inputDimensions, outputDimensions, bias: bias, zeroInitialized: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear(x)
    }
}

class ProsodyPredictor: Module {
    @ModuleInfo(key: "text_encoder") var textEncoder: DurationEncoder
    @ModuleInfo(key: "lstm") var lstm: BidirectionalLSTM
    @ModuleInfo(key: "duration_proj") var durationProj: DummyLinear
    @ModuleInfo(key: "shared") var sharedLSTM: BidirectionalLSTM
    @ModuleInfo(key: "F0") var f0: [AdainResBlk1d]
    @ModuleInfo(key: "N") var n: [AdainResBlk1d]
    @ModuleInfo(key: "F0_proj") var f0Proj: Conv1d
    @ModuleInfo(key: "N_proj") var nProj: Conv1d

//    var dropout: Dropout

    init(styleDim: Int, dHid: Int, nLayers: Int, maxDur: Int = 50, dropout: Float = 0.1) {
        self._textEncoder.wrappedValue = DurationEncoder(styDim: styleDim, dModel: dHid, nLayers: nLayers, dropout: dropout)
        self._lstm.wrappedValue = BidirectionalLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)
        self._durationProj.wrappedValue = DummyLinear(dHid, maxDur, bias: true)
        self._sharedLSTM.wrappedValue = BidirectionalLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)

        self._f0.wrappedValue = [
            AdainResBlk1d(dimIn: dHid, dimOut: dHid, styleDim: styleDim),
            AdainResBlk1d(dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: true),
            AdainResBlk1d(dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim)
        ]

        self._n.wrappedValue = [
            AdainResBlk1d(dimIn: dHid, dimOut: dHid, styleDim: styleDim),
            AdainResBlk1d(dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: true),
            AdainResBlk1d(dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim)
        ]

        self._f0Proj.wrappedValue = Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0)
        self._nProj.wrappedValue = Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0)

//        self.dropout = Dropout(p: 0.5)
        super.init()
    }

    func callAsFunction(texts: MLXArray, style: MLXArray, textLengths: MLXArray, alignment: MLXArray, m: MLXArray) -> (MLXArray, MLXArray) {
        var d = textEncoder(x: texts, style: style, textLengths: textLengths, mask: m)
        d = lstm(d).0
//        d = dropout(d)
        let duration = durationProj(d).squeezed(axis: -1)
        let en = MLX.matmul(d.transposed(axes: [0, 2, 1]), alignment)
        return (duration, en)
    }

    func f0nTrain(x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
        var x = x.transposed(axes: [0, 2, 1])
        x = sharedLSTM(x).0
        x = x.transposed(axes: [0, 2, 1])

        var f0x = x
        for layer in f0 { f0x = layer(f0x, s: s) }
        f0x = f0x.transposed(axes: [0, 2, 1])
        f0x = f0Proj(f0x)
        f0x = f0x.transposed(axes: [0, 2, 1]).squeezed(axis: 1)

        var nx = x
        for layer in n { nx = layer(nx, s: s) }
        nx = nx.transposed(axes: [0, 2, 1])
        nx = nProj(nx)
        nx = nx.transposed(axes: [0, 2, 1]).squeezed(axis: 1)

        return (f0x, nx)
    }
}
