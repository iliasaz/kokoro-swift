//
//  KokoroTTS.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 4/27/25.
//

import Foundation
import MLX
import MLXNN
import AVFoundation

final class KokoroTTS {
    private let kokoro: Kokoro
    private(set) var g2ps: [LanguageDialect: G2P] = [:]
    private(set) var vocab: [String: Int]
    private(set) var voices: [KokoroVoice: MLXArray] = [:]
    private var audioEngine: AVAudioEngine!
    private var playerNode: AVAudioPlayerNode!
    private var audioFormat: AVAudioFormat!

    // Buffer tracking for reliable playback completion detection
    private var scheduledBufferCount = 0
    private var completedBufferCount = 0

    // Constants
    static let maxTokenCount = 510
    static let sampleRate = 24000

    init() {
        self.kokoro = Kokoro(config: KokoroConfig.defaultConfig)
        self.vocab = KokoroConfig.defaultConfig.vocab
        setupAudioSystem()
        do {
            try loadWeights()
        } catch {
            print("error loading weights: \(error.localizedDescription)")
        }
    }

    deinit {
         NotificationCenter.default.removeObserver(self)
         cleanupAudioSystem()
     }


    // MARK: - Voice Loading and Converting

    // loading from bundled safetensors
    func loadVoice(_ voice: KokoroVoice) throws -> MLXArray {
        let safeTensorPath = Bundle.main.path(forResource: voice.rawValue.fileName, ofType: "safetensors")!
        let loadedArrays = try MLX.loadArrays(url: URL(fileURLWithPath: safeTensorPath))
        guard let voiceArray = loadedArrays[voice.rawValue.name] else {
            throw VoiceError.voiceKeyMismatch(expected: voice.rawValue.name, foundInFile: loadedArrays.keys.joined(separator: ", "))
        }
        return voiceArray
    }

    // loads one or more available voices and initializes the corresponding G2Ps
    func loadVoices(_ voices: [KokoroVoice]) {
        self.voices.removeAll()
        self.g2ps.removeAll()
        var dialects: Set<LanguageDialect> = []
        for voice in voices {
            do {
                self.voices[voice] = try loadVoice(voice)
                dialects.insert(voice.rawValue.language)
            } catch {
                print("Error: could not load voice \(voice.rawValue.name)")
            }
        }
        for dialect in dialects {
//            self.g2ps[dialect] = G2P(british: dialect == .enGB, fallback: try? EspeakFallback(british: dialect == .enGB).phonemize)
            self.g2ps[dialect] = G2P(british: dialect == .enGB, fallback: nil)
        }
    }

    // Weights Loading
    func loadWeights(from filePath: String = "/Users/ilia/Downloads/kokoro-v1_0.safetensors") throws {
//        let filePath = Bundle.main.path(forResource: "kokoro-v1_0", ofType: "safetensors")!
        let rawWeights = try MLX.loadArrays(url: URL(fileURLWithPath: filePath))
//        print("raw weights: \(rawWeights.keys.sorted().joined(separator: "\n"))")

        let sanitizedWeights = sanitizeWeights(rawWeights)
        let weights = ModuleParameters.unflattened(sanitizedWeights)
//        print("weights: \(weights.flattened().map({$0.0}).sorted().joined(separator: "\n"))")
        try kokoro.update(parameters: weights, verify: .all)
        eval(kokoro)
    }


    // MARK: - Text to Audio

    // generate audio and play it
    func speak(text: String, voice: KokoroVoice, speed: Float = 1.0) {
        // Reset audio system to ensure clean state
        resetAudioSystem()

        do {
            let audioData = try generate(text: text, voice: voice, speed: speed)
            playAudioChunk(audioData)
        } catch {
            print("Error: \(error)")
        }
    }

    // generate audio data from a given text chunk using a given voice
    func generate(text: String, voice: KokoroVoice, speed: Float = 1.0) throws -> MLXArray {
        guard self.voices.keys.contains(voice) else {
            throw NSError(domain: "KokoroTTS", code: 1001, userInfo: [NSLocalizedDescriptionKey: "Voice \(voice.rawValue.name) not loaded"])
        }
        guard self.g2ps.keys.contains(voice.rawValue.language) else {
            throw NSError(domain: "KokoroTTS", code: 1002, userInfo: [NSLocalizedDescriptionKey: "G2P model not loaded for \(voice.rawValue.language)"])
        }
        let g2p = g2ps[voice.rawValue.language]!
        let (phonemes, _) = g2p(text: text)
        let inputIDs = encode(phonemes: phonemes)
        print("phonemes: \(phonemes), inputIDs shape: \(inputIDs.shape)")
        let output = kokoro(inputIDs: inputIDs, voice: self.voices[voice]!, speed: speed)
        return output.audio
    }

    private func playAudioChunk(_ audioBuffer: MLXArray) {
        // Skip empty chunks
        let audioShape = audioBuffer.shape
        print("audioShape: \(audioShape)")
        guard !isAudioEmpty(shape: audioShape) else {
            print("Skipping empty audio chunk")
            return
        }

        // Extract audio data
        let (frameCount, audioData) = extractAudioData(from: audioBuffer)
        print("audioData: \(audioData.count), frameCount: \(frameCount)")

        // Create PCM buffer
        guard let buffer = createAudioBuffer(frameCount: frameCount, audioData: audioData) else {
            print("Failed to create audio buffer")
            return
        }

        let waveFileUrl = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test.wav")
        do {
            print("saving wav file")
            try buffer.saveToWavFile(at: waveFileUrl)
            print("file saved: \(waveFileUrl.path())")
            return
        } catch {
            print("could not save the wav file: \(error.localizedDescription)")
            return
        }

        // Ensure audio engine is running
        if !audioEngine.isRunning {
            resetAudioSystem()
        }

        playerNode.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            print("scheduled")
            guard let self = self else { return }
        }

        // Start playback if needed
        if !playerNode.isPlaying {
            print("start playing")
            playerNode.play()

            // Simple retry if player didn't start
            if !playerNode.isPlaying {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    print("start playing again")
                    self.playerNode.play()
                }
            }
        }
    }

    private func extractAudioData(from audioBuffer: MLXArray) -> (frameCount: Int, audioData: [Float]) {
        let audioShape = audioBuffer.shape

        // Handle different tensor shapes
        if audioShape.count == 1 {
            // 1D array [samples]
            let frameCount = audioShape[0]
            audioBuffer.eval()
            return (frameCount, audioBuffer.asArray(Float.self))
        } else if audioShape.count == 2 {
            // 2D array [1, samples]
            let frameCount = audioShape[1]
            let firstBatch = audioBuffer[0]
            firstBatch.eval()
            return (frameCount, firstBatch.asArray(Float.self))
        }
        print("Unexpected audio shape")
        // Fallback for unexpected shape
        return (0, [])
    }

    private func createAudioBuffer(frameCount: Int, audioData: [Float]) -> AVAudioPCMBuffer? {
        // Create buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return nil
        }

        // Set frame length
        buffer.frameLength = buffer.frameCapacity

        // Copy data
        let channels = buffer.floatChannelData!
        let chunkSize = 32768 // 32K samples at a time

        for startIdx in stride(from: 0, to: min(frameCount, audioData.count), by: chunkSize) {
            autoreleasepool {
                let endIdx = min(startIdx + chunkSize, min(frameCount, audioData.count))

                // Copy with volume boost
                for i in startIdx..<endIdx {
                    if i < audioData.count && i < Int(buffer.frameCapacity) {
                        // Apply volume boost (25%) with clipping prevention
                        channels[0][i] = min(max(audioData[i] * 1.25, -0.98), 0.98)
                    }
                }
            }
        }
        return buffer
    }


    // MARK: - Audio System Setup

    private func setupAudioSystem() {
        print("Setting up audio system")

        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: Double(KokoroTTS.sampleRate), channels: 1)!

        // Use dedicated audio processing queue to avoid QoS inversions
        let audioQueue = DispatchQueue(label: "com.mlx.audio.processing", qos: .userInteractive)
        audioQueue.sync {
            // Use platform-agnostic AudioSessionManager
            AudioSessionManager.shared.setupAudioSession()

            audioEngine.attach(playerNode)
            audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)

            do {
                try audioEngine.start()
                print("Audio system started successfully")
            } catch {
                print("Failed to start audio engine: \(error)")
            }
        }
    }

    private func cleanupAudioSystem() {
        // Stop player node first, which is the likely source of QoS inversion
        if playerNode.isPlaying {
            playerNode.pause() // Use pause instead of stop to avoid blocking
        }

        // Then stop the audio engine
        if audioEngine.isRunning {
            audioEngine.pause() // Use pause instead of stop to avoid blocking
        }

        // Finally, deactivate the audio session
        AudioSessionManager.shared.deactivateAudioSession()
    }

    private func resetAudioSystem() {
        print("Resetting audio system")

        // Stop player node first to avoid QoS inversion
        if playerNode.isPlaying {
            playerNode.pause() // Use pause instead of stop to avoid blocking
        }

        // Then stop the audio engine
        if audioEngine.isRunning {
            audioEngine.pause() // Use pause instead of stop to avoid blocking
        }

        // Reset audio session using platform-agnostic manager
        AudioSessionManager.shared.resetAudioSession()

        // Reconnect components with proper error handling
        if playerNode.engine != nil {
            audioEngine.detach(playerNode)
        }
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)

        // Restart engine
        do {
            try audioEngine.start()
            print("Audio engine restarted")
        } catch {
            print("Failed to restart audio engine: \(error)")
        }
    }


    // MARK: - Helpers

    private func encode(phonemes: String) -> MLXArray {
        let inputIds = phonemes.compactMap { self.vocab[String($0)] }
//        print("inputIds: \(inputIds)")
let fixedInputIds = [65, 156, 138, 56, 61, 16, 83, 58, 157, 69, 56, 16, 83, 16, 62, 156, 43, 102, 55, 16, 102, 56, 16, 83, 16, 64, 156, 72, 54, 51, 16, 123, 156, 72, 58, 62, 16, 102, 56, 16, 55, 156, 102, 61, 62, 16, 72, 56, 46, 16, 55, 156, 102, 61, 62, 83, 123, 123, 51, 16, 81, 86, 123, 65, 157, 138, 68, 16, 83, 16, 54, 156, 102, 125, 83, 54, 16, 64, 156, 102, 54, 102, 46, 147, 16, 65, 157, 86, 123, 16, 81, 83, 16, 61, 62, 156, 69, 123, 68, 16, 65, 156, 102, 61, 58, 83, 123, 46, 16, 61, 156, 51, 53, 123, 177, 62, 61, 16, 62, 83, 16, 81, 76, 135, 68, 16, 50, 157, 63, 16, 46, 156, 86, 123, 46, 16, 62, 83, 16, 54, 156, 102, 61, 83, 56]
        let paddedInputIdsBase = [0] + fixedInputIds + [0] // Add BOS/EOS tokens
        return MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])
    }

    // loading from the original JSON format
    func loadVoiceFromJSON(_ voice: KokoroVoice) throws -> MLXArray {
        let shape = [510, 1, 256]
        print("voice filename: \(voice.rawValue.fileName)")
        let filePath = Bundle.main.path(forResource: voice.rawValue.fileName, ofType: "json")!
        let data = try Data(contentsOf: URL(fileURLWithPath: filePath))
        let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])

        var swiftArray = Array(repeating: Float(0.0), count: shape[0] * shape[1] * shape[2])
        var swiftArrayIndex = 0

        if let nestedArray = jsonObject as? [[[Any]]] {
            guard nestedArray.count == shape[0] else { throw VoiceError.invalidDimension(found: nestedArray.count, expected: shape[0]) }
            for i in 0 ..< nestedArray.count {
                guard nestedArray[i].count == shape[1] else { throw VoiceError.invalidDimension(found: nestedArray[i].count, expected: shape[1]) }
                for j in 0 ..< nestedArray[i].count {
                    guard nestedArray[i][j].count == shape[2] else { throw VoiceError.invalidDimension(found: nestedArray[i][j].count, expected: shape[2]) }
                    for k in 0 ..< nestedArray[i][j].count {
                        if let n = nestedArray[i][j][k] as? Double {
                            swiftArray[swiftArrayIndex] = Float(n)
                            swiftArrayIndex += 1
                        } else {
                            throw VoiceError.invalidElement("Cannoe load value ")
                        }
                    }
                }
            }
        } else {
            throw VoiceError.invalidDataShape("Unexpected shape of nestedArray")
        }

        guard swiftArrayIndex == shape[0] * shape[1] * shape[2] else {
            throw VoiceError.invalidDataShape("Mismatch in array size: \(swiftArrayIndex) vs \(shape[0] * shape[1] * shape[2])")
        }

        return MLXArray(swiftArray).reshaped(shape)
    }

    // converting to safetensors format
    func convertVoice(_ voice: KokoroVoice) throws {
        let voiceArray = try loadVoiceFromJSON(voice)
        print("Voice array shape: \(voiceArray.shape)")
        let downloadFolderUrl = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        let safeTensorUrl = downloadFolderUrl.appending(component: voice.rawValue.fileName.appending(".safetensors"), directoryHint: .notDirectory)
        print("saving to \(safeTensorUrl)")
        try MLX.save(arrays: [voice.rawValue.name: voiceArray], url: safeTensorUrl)
    }

    // utility to convert all voices in the original json format to safetensors
    func convertAllVoices() throws {
        for voice in KokoroVoice.allCases {
            print("Converting voice: \(voice.rawValue.name)")
            try convertVoice(voice)
            print("Successfully converted \(voice.rawValue.name)")
        }
        print("All voices converted successfully.")
    }

    // update weight keys as there are slight differences between the original kokoro weights and MLX implementations of common layers
    private func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]

        for (key, value) in weights.sorted(by: { $0.key < $1.key }) {
            if key.hasPrefix("bert") {
                if key.contains("position_ids") {
                    // drop position_ids as they are not used
                    continue
                } else {
                    sanitizedWeights[key] = value
                }
            } else if key.hasPrefix("bert_encoder") {
                sanitizedWeights[key] = value

            } else if key.hasPrefix("text_encoder") {
//                if key.hasSuffix(".gamma") || key.hasSuffix(".beta") {
//                    let baseKey = key.components(separatedBy: ".").dropLast().joined(separator: ".")
//                    let newKey = key.hasSuffix(".gamma") ? "\(baseKey).weight" : "\(baseKey).bias"
//                    sanitizedWeights[newKey] = value
//
//                } else
                if key.contains("weight_v") {
                    if checkArrayShape(value) {
                        sanitizedWeights[key] = value
                    } else {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    }

//                } else if key.hasSuffix(".weight_ih_l0_reverse") ||
//                            key.hasSuffix(".weight_hh_l0_reverse") ||
//                            key.hasSuffix(".bias_ih_l0_reverse") ||
//                            key.hasSuffix(".bias_hh_l0_reverse") ||
//                            key.hasSuffix(".weight_ih_l0") ||
//                            key.hasSuffix(".weight_hh_l0") ||
//                            key.hasSuffix(".bias_ih_l0") ||
//                            key.hasSuffix(".bias_hh_l0") {
//
//                    let lstmSanitized = sanitizeLSTMWeights(key: key, value: value)
//                    for (k, v) in lstmSanitized {
//                        sanitizedWeights[k] = v
//                    }
                } else {
                    sanitizedWeights[key] = value
                }

            } else if key.hasPrefix("predictor") {
                if key.contains("F0_proj.weight") || key.contains("N_proj.weight") {
                    sanitizedWeights[key] = value.transposed(0, 2, 1)

                } else if key.contains("weight_v") {
                    if checkArrayShape(value) {
                        sanitizedWeights[key] = value
                    } else {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    }

//                } else if key.hasSuffix(".weight_ih_l0_reverse") ||
//                            key.hasSuffix(".weight_hh_l0_reverse") ||
//                            key.hasSuffix(".bias_ih_l0_reverse") ||
//                            key.hasSuffix(".bias_hh_l0_reverse") ||
//                            key.hasSuffix(".weight_ih_l0") ||
//                            key.hasSuffix(".weight_hh_l0") ||
//                            key.hasSuffix(".bias_ih_l0") ||
//                            key.hasSuffix(".bias_hh_l0") {
//
//                    let lstmSanitized = sanitizeLSTMWeights(key: key, value: value)
//                    for (k, v) in lstmSanitized {
//                        sanitizedWeights[k] = v
//                    }
                } else {
                    sanitizedWeights[key] = value
                }

            } else if key.hasPrefix("decoder") {
                sanitizedWeights[key] = sanitizeDecoderWeights(key: key, value: value)
            }
        }

        return sanitizedWeights
    }

    private func sanitizeLSTMWeights(key: String, value: MLXArray) -> [String: MLXArray] {
        let baseKey = key.components(separatedBy: ".").dropLast().joined(separator: ".")
        let weightMap: [String: String] = [
            "weight_ih_l0_reverse": "Wx_backward",
            "weight_hh_l0_reverse": "Wh_backward",
            "bias_ih_l0_reverse": "bias_ih_backward",
            "bias_hh_l0_reverse": "bias_hh_backward",
            "weight_ih_l0": "Wx_forward",
            "weight_hh_l0": "Wh_forward",
            "bias_ih_l0": "bias_ih_forward",
            "bias_hh_l0": "bias_hh_forward",
        ]

        for (suffix, newSuffix) in weightMap {
            if key.hasSuffix(suffix) {
                return ["\(baseKey).\(newSuffix)": value]
            }
        }
        return [key: value]
    }

    private func sanitizeDecoderWeights(key: String, value: MLXArray) -> MLXArray {
        let returnValue: MLXArray
        if key.contains("noise_convs"), key.hasSuffix(".weight") {
            returnValue = value.transposed(0, 2, 1)
        } else if key.hasSuffix(".weight_v") || key.hasSuffix(".weight_g") {
            returnValue = checkArrayShape(value) ? value : value.transposed(0, 2, 1)
//            print("key: \(key), value shape: \(returnValue.shape)")
        } else {
            returnValue = value
        }
        return returnValue
    }

    private func checkArrayShape(_ arr: MLXArray) -> Bool {
        guard arr.shape.count != 3 else { return false }

        let outChannels = arr.shape[0]
        let kH = arr.shape[1]
        let kW = arr.shape[2]

        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }

    private func isAudioEmpty(shape: [Int]) -> Bool {
        if shape.count == 1 {
            return shape[0] <= 1
        } else if shape.count == 2 {
            return shape[1] <= 1
        }
        return true
    }

}

// MARK: - Voices
// Available languages
public enum LanguageDialect: String, CaseIterable {
    case none = ""
    case enUS = "en-us"
    case enGB = "en-gb"
}

public enum Gender {
    case female, male
}

public enum VoiceError: Error {
    case fileNotFound(String)
    case invalidDimension(found: Int, expected: Int)
    case invalidElement(String)
    case invalidDataShape(String)
    case voiceKeyMismatch(expected: String, foundInFile: String)
}

public typealias VoiceType = (language: LanguageDialect, gender: Gender, fileName: String, name: String)

// Available voices - english only for now
public enum KokoroVoice: RawRepresentable, CaseIterable, Equatable {
    case afAlloy, afAoede, afBella, afHeart, afJessica, afKore, afNicole, afNova, afRiver, afSarah, afSky
    case amAdam, amEcho, amEric, amFenrir, amLiam, amMichael, amOnyx, amPuck, amSanta
    case bfAlice, bfEmma, bfIsabella, bfLily
    case bmDaniel, bmFable, bmGeorge, bmLewis

    public var rawValue: VoiceType {
        switch self {
            case .afAlloy: return VoiceType(language: .enUS, gender: .female, fileName: "af_alloy", name: "afAlloy")
            case .afAoede: return VoiceType(language: .enUS, gender: .female, fileName: "af_aoede", name: "afAoede")
            case .afBella: return VoiceType(language: .enUS, gender: .female, fileName: "af_bella", name: "afBella")
            case .afHeart: return VoiceType(language: .enUS, gender: .female, fileName: "af_heart", name: "afHeart")
            case .afJessica: return VoiceType(language: .enUS, gender: .female, fileName: "af_jessica", name: "afJessica")
            case .afKore: return VoiceType(language: .enUS, gender: .female, fileName: "af_kore", name: "afKore")
            case .afNicole: return VoiceType(language: .enUS, gender: .female, fileName: "af_nicole", name: "afNicole")
            case .afNova: return VoiceType(language: .enUS, gender: .female, fileName: "af_nova", name: "afNova")
            case .afRiver: return VoiceType(language: .enUS, gender: .female, fileName: "af_river", name: "afRiver")
            case .afSarah: return VoiceType(language: .enUS, gender: .female, fileName: "af_sarah", name: "afSarah")
            case .afSky: return VoiceType(language: .enUS, gender: .female, fileName: "af_sky", name: "afSky")

            case .amAdam: return VoiceType(language: .enUS, gender: .male, fileName: "am_adam", name: "amAdam")
            case .amEcho: return VoiceType(language: .enUS, gender: .male, fileName: "am_echo", name: "amEcho")
            case .amEric: return VoiceType(language: .enUS, gender: .male, fileName: "am_eric", name: "amEric")
            case .amFenrir: return VoiceType(language: .enUS, gender: .male, fileName: "am_fenrir", name: "amFenrir")
            case .amLiam: return VoiceType(language: .enUS, gender: .male, fileName: "am_liam", name: "amLiam")
            case .amMichael: return VoiceType(language: .enUS, gender: .male, fileName: "am_michael", name: "amMichael")
            case .amOnyx: return VoiceType(language: .enUS, gender: .male, fileName: "am_onyx", name: "amOnyx")
            case .amPuck: return VoiceType(language: .enUS, gender: .male, fileName: "am_puck", name: "amPuck")
            case .amSanta: return VoiceType(language: .enUS, gender: .male, fileName: "am_santa", name: "amSanta")

            case .bfAlice: return VoiceType(language: .enGB, gender: .female, fileName: "bf_alice", name: "bfAlice")
            case .bfEmma: return VoiceType(language: .enGB, gender: .female, fileName: "bf_emma", name: "bfEmma")
            case .bfIsabella: return VoiceType(language: .enGB, gender: .female, fileName: "bf_isabella", name: "bfIsabella")
            case .bfLily: return VoiceType(language: .enGB, gender: .female, fileName: "bf_lily", name: "bfLily")

            case .bmDaniel: return VoiceType(language: .enGB, gender: .male, fileName: "bm_daniel", name: "bmDaniel")
            case .bmFable: return VoiceType(language: .enGB, gender: .male, fileName: "bm_fable", name: "bmFable")
            case .bmGeorge: return VoiceType(language: .enGB, gender: .male, fileName: "bm_george", name: "bmGeorge")
            case .bmLewis: return VoiceType(language: .enGB, gender: .male, fileName: "bm_lewis", name: "bmLewis")
        }
    }

    public init?(rawValue: VoiceType) {
        switch rawValue {
            case (language: .enUS, gender: .female, fileName: "af_alloy", name: "afAlloy"): self = .afAlloy
            case (language: .enUS, gender: .female, fileName: "af_aoede", name: "afAoede"): self = .afAoede
            case (language: .enUS, gender: .female, fileName: "af_bella", name: "afBella"): self = .afBella
            case (language: .enUS, gender: .female, fileName: "af_heart", name: "afHeart"): self = .afHeart
            case (language: .enUS, gender: .female, fileName: "af_jessica", name: "afJessica"): self = .afJessica
            case (language: .enUS, gender: .female, fileName: "af_kore", name: "afKore"): self = .afKore
            case (language: .enUS, gender: .female, fileName: "af_nicole", name: "afNicole"): self = .afNicole
            case (language: .enUS, gender: .female, fileName: "af_nova", name: "afNova"): self = .afNova
            case (language: .enUS, gender: .female, fileName: "af_river", name: "afRiver"): self = .afRiver
            case (language: .enUS, gender: .female, fileName: "af_sarah", name: "afSarah"): self = .afSarah
            case (language: .enUS, gender: .female, fileName: "af_sky", name: "afSky"): self = .afSky

            case (language: .enUS, gender: .male, fileName: "am_adam", name: "amAdam"): self = .amAdam
            case (language: .enUS, gender: .male, fileName: "am_echo", name: "amEcho"): self = .amEcho
            case (language: .enUS, gender: .male, fileName: "am_eric", name: "amEric"): self = .amEric
            case (language: .enUS, gender: .male, fileName: "am_fenrir", name: "amFenrir"): self = .amFenrir
            case (language: .enUS, gender: .male, fileName: "am_liam", name: "amLiam"): self = .amLiam
            case (language: .enUS, gender: .male, fileName: "am_michael", name: "amMichael"): self = .amMichael
            case (language: .enUS, gender: .male, fileName: "am_onyx", name: "amOnyx"): self = .amOnyx
            case (language: .enUS, gender: .male, fileName: "am_puck", name: "amPuck"): self = .amPuck
            case (language: .enUS, gender: .male, fileName: "am_santa", name: "amSanta"): self = .amSanta

            case (language: .enGB, gender: .female, fileName: "bf_alice", name: "bfAlice"): self = .bfAlice
            case (language: .enGB, gender: .female, fileName: "bf_emma", name: "bfEmma"): self = .bfEmma
            case (language: .enGB, gender: .female, fileName: "bf_isabella", name: "bfIsabella"): self = .bfIsabella
            case (language: .enGB, gender: .female, fileName: "bf_lily", name: "bfLily"): self = .bfLily

            case (language: .enGB, gender: .male, fileName: "bm_daniel", name: "bmDaniel"): self = .bmDaniel
            case (language: .enGB, gender: .male, fileName: "bm_fable", name: "bmFable"): self = .bmFable
            case (language: .enGB, gender: .male, fileName: "bm_george", name: "bmGeorge"): self = .bmGeorge
            case (language: .enGB, gender: .male, fileName: "bm_lewis", name: "bmLewis"): self = .bmLewis

            default: return nil
        }
    }
}


extension AVAudioPCMBuffer {
    func saveToWavFile(at url: URL) throws {
        let audioFile = try AVAudioFile(forWriting: url,
                                      settings: format.settings,
                                      commonFormat: .pcmFormatFloat32,
                                      interleaved: false)
        try audioFile.write(from: self)
    }
}
