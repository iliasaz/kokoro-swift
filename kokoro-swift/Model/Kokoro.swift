//
//  Kokoro.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 4/26/25.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

final class Kokoro: Module {
    let config: KokoroConfig

    @ModuleInfo(key: "bert") var bert: CustomAlbert
    @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
    @ModuleInfo(key: "predictor") var predictor: ProsodyPredictor
    @ModuleInfo(key: "text_encoder") var textEncoder: TextEncoder
    @ModuleInfo(key: "decoder") var decoder: Decoder

    init(config: KokoroConfig) {
        self.config = config
        self._bert.wrappedValue = CustomAlbert(config: config.plbertArgs)
        self._bertEncoder.wrappedValue = Linear(config.plbertArgs.hiddenSize, config.hiddenDim, zeroInitialized: true)
        self._predictor.wrappedValue = ProsodyPredictor(
            styleDim: config.styleDim,
            dHid: config.hiddenDim,
            nLayers: config.nLayer,
            maxDur: config.maxDuration,
            dropout: config.dropout
        )
        self._textEncoder.wrappedValue = TextEncoder(
            channels: config.hiddenDim,
            kernelSize: config.textEncoderKernelSize,
            depth: config.nLayer,
            nSymbols: config.nToken
        )
        self._decoder.wrappedValue = Decoder(
            dimIn: config.hiddenDim,
            styleDim: config.styleDim,
            dimOut: config.nMels,
            resblockKernelSizes: config.istftNetConfig.resblockKernelSizes,
            upsampleRates: config.istftNetConfig.upsampleRates,
            upsampleInitialChannel: config.istftNetConfig.upsampleInitialChannel,
            resblockDilationSizes: config.istftNetConfig.resblockDilationSizes,
            upsampleKernelSizes: config.istftNetConfig.upsampleKernelSizes,
            genISTFTNFFT: config.istftNetConfig.genISTFTNFFT,
            genISTFTHopSize: config.istftNetConfig.genISTFTHopSize
        )
        super.init()
    }

    struct Output {
        let audio: MLXArray
        let predDurations: MLXArray
    }

    /// Core synthesis function.
    ///
    /// - Parameters:
    ///   - inputIDs: Tensor of phoneme token IDs. Shape: [1, seq_len]
    ///   - voice: Voice embedding tensor. Shape: [1, voice_dim]
    ///   - speed: Scalar speed factor.
    /// - Returns: Output containing generated audio and predicted durations.
    func callAsFunction(
        inputIDs: MLXArray,
        voice: MLXArray,
        speed: Float = 1.0
    ) -> Output {
        precondition(inputIDs.ndim == 2, "Expected inputIDs to be of shape [1, seq_len]")
        print("inputIDs shape: \(inputIDs.shape)")
        let batchSize = inputIDs.shape[0]
        let seqLen = inputIDs.shape[1]

        // Build attention mask
        let inputLengths = MLXArray([seqLen])
        var textMask = MLXArray(0 ..< seqLen)[.newAxis, .ellipsis]
        textMask = repeated(textMask, count: batchSize, axis: 0)
        textMask = (textMask + 1) .> inputLengths[0..., .newAxis]
        print("textMask shape: \(textMask.shape)")
        // BERT-style encoding
        let (bertSequenceOutput, _) = bert(inputIds: inputIDs, attentionMask: (.!textMask).asType(.int32))
        print("bert: attentionMask shape: \(textMask.shape), bertSequenceOutput shape: \(bertSequenceOutput.shape)")

        let bertEncoded = bertEncoder(bertSequenceOutput).transposed(0, 2, 1)
        print("bertEncoded shape: \(bertEncoded.shape)")

        print("voice shape: \(voice.shape)")
        let refS = voice[min(seqLen - 2 /* padding */ - 1, voice.shape[0] - 1), 0 ... 1, 0...]
        print("voice.shape[0]: \(voice.shape[0]) ")
        print("refS = voice[\(min(seqLen - 2 /* padding */ - 1, voice.shape[0] - 1)), 0 ... 1, 0...], voice.shape[0]: \(voice.shape[0])")
        print("refS shape: \(refS.shape)")

        let styleEmbedding = refS[0 ... 1, 128...]
        print("styleEmbedding shape: \(styleEmbedding.shape), mean: \(styleEmbedding.mean().item(Float.self)), max: \(styleEmbedding.max().item(Float.self)) ")

        let encoded = predictor.textEncoder(x: bertEncoded, style: styleEmbedding, textLengths: inputLengths, mask: textMask)
        print("encoded: \(encoded.shape)")
        let (x, _) = predictor.lstm(encoded)
        print("lstm x: \(x.shape)")
        var durationPred = predictor.durationProj(x)
        print("durationPred: \(durationPred.shape)")

        durationPred = MLX.sigmoid(durationPred).sum(axis: -1) / speed
        print("durationSigmoid: \(durationPred.shape)")

        let predDur = MLX.clip(MLX.round(durationPred), min: 1).asType(.int32)[0]
//        let predDur = MLXArray(converting: [13.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0, 13.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 6.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 26.0, 6.0])
        print("predDur: \(predDur.shape)")
        print("predDur: \(predDur.map({String($0.item(Float.self))}).joined(separator: ", "))")
        print("Sum of durations: \(predDur.sum().item(Int.self))")


        // Alignment expansion
        let indices = MLX.concatenated( predDur.enumerated().map { idx, dur in MLX.full([dur.item(Int.self)], values: idx) } )
        var predAlnTrg = MLXArray.zeros([seqLen, indices.shape[0]])
        predAlnTrg[indices, MLXArray(0 ..< indices.shape[0])] = MLXArray(1)
        predAlnTrg = predAlnTrg[.newAxis, .ellipsis]
        print("predAlnTrg: \(predAlnTrg.shape)")
        print("Frame count (T′): \(indices.shape[0])")
        // Target encodings
        let en = encoded.transposed(0, 2, 1).matmul(predAlnTrg)
        print("en shape:", en.shape)
        print("en min/max/mean:", en.min(), en.max(), en.mean())

        let (F0_pred, N_pred) = predictor.f0nTrain(x: en, s: styleEmbedding)
        print("F0_pred: \(F0_pred.shape), N_pred: \(N_pred.shape)")
        print("F0_pred min/max: \(F0_pred.min().item(Float.self)),\(F0_pred.max().item(Float.self)), N_pred min/max: \(N_pred.min().item(Float.self)), \(N_pred.max().item(Float.self))")

        let tEn = textEncoder(x: inputIDs, inputLengths: inputLengths, m: textMask)
        let asr = tEn.matmul(predAlnTrg)

        print("ASR shape: \(asr.shape), min/max/mean: \(asr.min().item(Float.self)), \(asr.max().item(Float.self)), \(asr.mean().item(Float.self))")

        let voiceS = refS[0 ... 1, 0 ... 127]
        let audio = decoder(asr: asr, f0Curve: F0_pred, n: N_pred, s: voiceS)

        print("audio shape: \(audio.shape), min, max, mean: \(audio.min().item(Float.self)), \(audio.max().item(Float.self)), \(audio.mean().item(Float.self))")
        return Output(audio: audio[0], predDurations: predDur)
    }


    /// Load model weights.
    ///
    /// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
    /// This function loads all `safetensor` files in the given `modelDirectory`,
    /// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
    /// updates the model with the weights.
    func loadWeights(
        modelDirectory: URL) throws {
        // load the weights
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
            }
        }

        // per-model cleanup
//        weights = model.sanitize(weights: weights)

        // quantize if needed
//        if let quantization {
//            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) {
//                path, module in
//                weights["\(path).scales"] != nil
//            }
//        }

        // apply the loaded weights
        let parameters = ModuleParameters.unflattened(weights)
        try self.update(parameters: parameters, verify: [.all])

        eval(self)
    }
}
