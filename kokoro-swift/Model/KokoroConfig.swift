//
//  KokoroConfig.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 6/19/25.
//


/// Configuration structure for the Kokoro model.
/// Parsed from config.json or constructed manually.
struct KokoroConfig: Codable {
    // Top-level fields
    let vocab: [String: Int]
    let nToken: Int
    let hiddenDim: Int
    let styleDim: Int
    let nLayer: Int
    let maxDuration: Int
    let dropout: Float
    let textEncoderKernelSize: Int
    let nMels: Int

    // Submodules
    let plbertArgs: AlbertModelArgs
    let istftNetConfig: ISTFTNetConfig

    init(
        vocab: [String: Int],
        nToken: Int,
        hiddenDim: Int,
        styleDim: Int,
        nLayer: Int,
        maxDuration: Int,
        dropout: Float,
        textEncoderKernelSize: Int,
        nMels: Int,
        plbertArgs: AlbertModelArgs,
        istftNetConfig: ISTFTNetConfig
    ) {
        self.vocab = vocab
        self.nToken = nToken
        self.hiddenDim = hiddenDim
        self.styleDim = styleDim
        self.nLayer = nLayer
        self.maxDuration = maxDuration
        self.dropout = dropout
        self.textEncoderKernelSize = textEncoderKernelSize
        self.nMels = nMels
        self.plbertArgs = plbertArgs
        self.istftNetConfig = istftNetConfig
    }
}

/// Configuration for the ISTFTNet Generator part of Kokoro Decoder.
/// This matches the `istftnet` dictionary inside config.json.
struct ISTFTNetConfig: Codable {
    let resblockKernelSizes: [Int]
    let upsampleRates: [Int]
    let upsampleInitialChannel: Int
    let resblockDilationSizes: [[Int]]
    let upsampleKernelSizes: [Int]
    let genISTFTNFFT: Int
    let genISTFTHopSize: Int

    init(
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genISTFTNFFT: Int,
        genISTFTHopSize: Int
    ) {
        self.resblockKernelSizes = resblockKernelSizes
        self.upsampleRates = upsampleRates
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblockDilationSizes = resblockDilationSizes
        self.upsampleKernelSizes = upsampleKernelSizes
        self.genISTFTNFFT = genISTFTNFFT
        self.genISTFTHopSize = genISTFTHopSize
    }
}

let kokoroDefaultVocab: [String: Int] = [
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11,
    "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "\u{0303}": 17, "ʣ": 18, "ʥ": 19,
    "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\u{AB67}": 23, "A": 24, "I": 25, "O": 31, "Q": 33,
    "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46,
    "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56,
    "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65,
    "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76,
    "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87,
    "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111,
    "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120,
    "ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132,
    "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143,
    "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169,
    "→": 171, "↗": 172, "↘": 173, "ᵻ": 177
]

extension KokoroConfig {
    static let defaultConfig = KokoroConfig(
        vocab: kokoroDefaultVocab,
        nToken: 178,
        hiddenDim: 512,
        styleDim: 128,
        nLayer: 3,
        maxDuration: 50,
        dropout: 0.2,
        textEncoderKernelSize: 5,
        nMels: 80,
        plbertArgs: AlbertModelArgs(
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            hiddenSize: 768,
            intermediateSize: 2048,
            maxPositionEmbeddings: 512,
            modelType: "albert",
            embeddingSize: 128,
            innerGroupNum: 1,
            numHiddenGroups: 1,
            hiddenDropoutProb: 0.1,
            attentionProbsDropoutProb: 0.1,
            typeVocabSize: 2,
            initializerRange: 0.02,
            layerNormEps: 1e-12,
            vocabSize: 178,
            dropout: 0.2
        ),
        istftNetConfig: ISTFTNetConfig(
            resblockKernelSizes: [3, 7, 11],
            upsampleRates: [10, 6],
            upsampleInitialChannel: 512,
            resblockDilationSizes: [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5]
            ],
            upsampleKernelSizes: [20, 12],
            genISTFTNFFT: 20,
            genISTFTHopSize: 5
        )
    )
}

