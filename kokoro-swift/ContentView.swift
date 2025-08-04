//
//  ContentView.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import SwiftUI

struct ContentView: View {
    @State private var text = """
    Once upon a time, in a valley wrapped in mist and mystery, there was a little village where the stars whispered secrets to those who dared to listen.
    """
    @State private var result: String = ""
    let kokoro = KokoroTTS()

    var body: some View {
        VStack {
            TextField("story", text: $text)
            Button("getTokens") {
                getTokens()
            }
            Text(result)

//            Button("convert All") {
//                convert()
//            }

            Button("load voices") {
                loadSafeTensorVoices()
            }

            Button("speak") {
                loadSafeTensorVoices()
                kokoro.speak(text: text, voice: .afAlloy)
            }
        }
        .padding()
        .task {
//            kokoro.loadVoices([.afAlloy])
//            kokoro.speak(text: text, voice: .afAlloy, speed: 0.9)
        }
    }

    func getTokens() {
//        kokoro.initEspeak()
        let voice: KokoroVoice = .afAlloy
        kokoro.loadVoices([voice])
        result = kokoro.getPhonemes(for: text, language: voice.rawValue.language)
    }

    func convert() {
        let tts = KokoroTTS()
        do {
            try tts.convertAllVoices()
        } catch {
            logger.error("error: \(error.localizedDescription)")
        }
    }

    func loadSafeTensorVoices() {
        kokoro.loadVoices([.afAlloy])
    }
}

#Preview {
    ContentView()
}
