//
//  MLXService.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Hub


/// A service class that manages machine learning models for text and vision-language tasks.
/// This class handles model loading, caching, and text generation using various LLM and VLM models.
//@Observable
//class MLXService {
//    /// Cache to store loaded model containers to avoid reloading.
//    private let modelCache = NSCache<NSString, ModelContainer>()
//
//    /// Tracks the current model download progress.
//    /// Access this property to monitor model download status.
//    @MainActor
//    private(set) var modelDownloadProgress: Progress?
//
//    /// Loads a model from the hub or retrieves it from cache.
//    /// - Parameter model: The model configuration to load
//    /// - Returns: A ModelContainer instance containing the loaded model
//    /// - Throws: Errors that might occur during model loading
//    private func load(model: TTSModel) async throws -> ModelContainer {
//        // Set GPU memory limit to prevent out of memory issues
//        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
//
//        // Return cached model if available to avoid reloading
//        if let container = modelCache.object(forKey: model.name as NSString) {
//            return container
//        } else {
//            // Load model and track download progress
//            let container = try await loadContainer(
//                hub: HubApi.default, configuration: model.configuration
//            ) { progress in
//                Task { @MainActor in
//                    self.modelDownloadProgress = progress
//                }
//            }
//
//            // Cache the loaded model for future use
//            modelCache.setObject(container, forKey: model.name as NSString)
//
//            return container
//        }
//    }
//
//    private func loadContainer(hub: HubApi = HubApi(),configuration: ModelConfiguration,progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }) async throws -> ModelContainer {
//        // download weights and config
//        let modelDirectory = try await downloadModel(hub: hub, configuration: configuration, progressHandler: progressHandler)
//        // load the generic config to unerstand which model and how to load the weights
//        let configurationURL = modelDirectory.appending(component: "config.json")
//        let modelConfig = try JSONDecoder().decode(KokoroConfig.self, from: Data(contentsOf: configurationURL))
//        let model = Kokoro(config: modelConfig)
//        // apply the weights to the bare model
//        try model.loadWeights(modelDirectory: modelDirectory)
//
//        return ModelContainer(context: context)
//    }
//
//    /// Generates text based on the provided messages using the specified model.
//    /// - Parameters:
//    ///   - messages: Array of chat messages including user, assistant, and system messages
//    ///   - model: The language model to use for generation
//    /// - Returns: An AsyncStream of generated text tokens
//    /// - Throws: Errors that might occur during generation
//    func generate(messages: [Message], model: LMModel) async throws -> AsyncStream<Generation> {
//        // Load or retrieve model from cache
//        let modelContainer = try await load(model: model)
//
//        // Map app-specific Message type to Chat.Message for model input
//        let chat = messages.map { message in
//            let role: Chat.Message.Role =
//                switch message.role {
//                case .assistant:
//                    .assistant
//                case .user:
//                    .user
//                case .system:
//                    .system
//                }
//
//            // Process any attached media for VLM models
//            let images: [UserInput.Image] = message.images.map { imageURL in .url(imageURL) }
//            let videos: [UserInput.Video] = message.videos.map { videoURL in .url(videoURL) }
//
//            return Chat.Message(
//                role: role, content: message.content, images: images, videos: videos)
//        }
//
//        // Prepare input for model processing
//        let userInput = UserInput(chat: chat)
//
//        // Generate response using the model
//        return try await modelContainer.perform { (context: ModelContext) in
//            let lmInput = try await context.processor.prepare(input: userInput)
//            // Set temperature for response randomness (0.7 provides good balance)
//            let parameters = GenerateParameters(temperature: 0.7)
//
//            return try MLXLMCommon.generate(
//                input: lmInput, parameters: parameters, context: context)
//        }
//    }
//}


/// Represents a language model configuration with its associated properties and type.
/// Can represent either a large language model (LLM) or a vision-language model (VLM).
struct TTSModel {
    /// Name of the model
    let name: String

    /// Configuration settings for model initialization
    let configuration: ModelConfiguration
}

// MARK: - Helpers
extension TTSModel: Identifiable, Hashable {
    var id: String {
        name
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}

extension LLMRegistry {
    public static let kokoro = ModelConfiguration(id: "prince-canuma/Kokoro-82M", defaultPrompt: "Check out my new app!")
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await TTSModelFactory.shared.loadContainer(
///     configuration: LLMRegistry.llama3_8B_4bit)
/// ```
//public class TTSModelFactory: ModelFactory {
//
//    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
//        self.typeRegistry = typeRegistry
//        self.modelRegistry = modelRegistry
//    }
//
//    /// Shared instance with default behavior.
//    public static let shared = TTSModelFactory(
//        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)
//
//    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
//    public let typeRegistry: ModelTypeRegistry
//
//    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
//    public let modelRegistry: AbstractModelRegistry
//
//    public func _load(
//        hub: HubApi, configuration: ModelConfiguration,
//        progressHandler: @Sendable @escaping (Progress) -> Void
//    ) async throws -> ModelContext {
//        // download weights and config
//        let modelDirectory = try await downloadModel(
//            hub: hub, configuration: configuration, progressHandler: progressHandler)
//
//        // load the generic config to unerstand which model and how to load the weights
//        let configurationURL = modelDirectory.appending(component: "config.json")
//        let baseConfig = try JSONDecoder().decode(
//            BaseConfiguration.self, from: Data(contentsOf: configurationURL))
//        let model = try typeRegistry.createModel(
//            configuration: configurationURL, modelType: baseConfig.modelType)
//
//        // apply the weights to the bare model
//        try loadWeights(
//            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization)
//
//        let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)
//
//        return .init(
//            configuration: configuration, model: model,
//            processor: LLMUserInputProcessor(
//                tokenizer: tokenizer, configuration: configuration,
//                messageGenerator: DefaultMessageGenerator()),
//            tokenizer: tokenizer)
//    }
//
//}

/// Extension providing a default HubApi instance for downloading model files
extension HubApi {
    /// Default HubApi instance configured to download models to the user's Downloads directory
    /// under a 'huggingface' subdirectory.
    static let `default` = HubApi(
        downloadBase: URL.downloadsDirectory.appending(path: "huggingface"))
}
