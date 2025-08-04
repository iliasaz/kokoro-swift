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
    ) -> MLXArray {
        precondition(inputIDs.ndim == 2, "Expected inputIDs to be of shape [1, seq_len]")
        let batchSize = inputIDs.shape[0]
        let seqLen = inputIDs.shape[1]

        // Build attention mask
        let inputLengths = MLXArray([seqLen])
        var textMask = MLXArray(0 ..< seqLen)[.newAxis, .ellipsis]
        textMask = repeated(textMask, count: batchSize, axis: 0)
        textMask = (textMask + 1) .> inputLengths[0..., .newAxis]
        // BERT-style encoding
        let (bertSequenceOutput, _) = bert(inputIds: inputIDs, attentionMask: (.!textMask).asType(.int32))
        let bertEncoded = bertEncoder(bertSequenceOutput).transposed(0, 2, 1)
        let refS = voice[min(seqLen - 2 /* padding */ - 1, voice.shape[0] - 1), 0 ... 1, 0...]
        let styleEmbedding = refS[0 ... 1, 128...]
        let encoded = predictor.textEncoder(x: bertEncoded, style: styleEmbedding, textLengths: inputLengths, mask: textMask)
        let (x, _) = predictor.lstm(encoded)
        var durationPred = predictor.durationProj(x)
        durationPred = MLX.sigmoid(durationPred).sum(axis: -1) / speed
        let predDur = MLX.clip(MLX.round(durationPred), min: 1).asType(.int32)[0]

        // Alignment expansion
        let indices = MLX.concatenated( predDur.enumerated().map { idx, dur in MLX.full([dur.item(Int.self)], values: idx) } )
        var predAlnTrg = MLXArray.zeros([seqLen, indices.shape[0]])
        predAlnTrg[indices, MLXArray(0 ..< indices.shape[0])] = MLXArray(1)
        predAlnTrg = predAlnTrg[.newAxis, .ellipsis]

        // Target encodings
        let en = encoded.transposed(0, 2, 1).matmul(predAlnTrg)
        let (F0_pred, N_pred) = predictor.f0nTrain(x: en, s: styleEmbedding)
        let tEn = textEncoder(x: inputIDs, inputLengths: inputLengths, m: textMask)
        let asr = tEn.matmul(predAlnTrg)
        let voiceS = refS[0 ... 1, 0 ... 127]

        asr.eval()
        F0_pred.eval()
        N_pred.eval()
        voiceS.eval()
        predDur.eval()

        autoreleasepool {
            _ = en
            _ = predAlnTrg
            _ = styleEmbedding
            _ = tEn
            _ = refS
            _ = indices
            _ = durationPred
            _ = x
            _ = encoded
            _ = styleEmbedding
            _ = bertEncoded
            _ = bertSequenceOutput
            _ = textMask
            _ = inputLengths
        }

        let audio = decoder(asr: asr, f0Curve: F0_pred, n: N_pred, s: voiceS)
        audio.eval()

        logger.debug("audio shape: \(audio.shape), min, max, mean: \(audio.min().item(Float.self)), \(audio.max().item(Float.self)), \(audio.mean().item(Float.self))")

        autoreleasepool {
          _ = asr
          _ = F0_pred
          _ = N_pred
          _ = voiceS
        }
        return audio[0]
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

        // apply the loaded weights
        let parameters = ModuleParameters.unflattened(weights)
        try self.update(parameters: parameters, verify: [.all])

        eval(self)
    }
}
