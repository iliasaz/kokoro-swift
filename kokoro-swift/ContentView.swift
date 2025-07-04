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
            Button("do") {
                getTokens()
            }
            Text(result)

            Button("convert All") {
                convert()
            }

            Button("load voices") {
                loadSafeTensorVoices()
            }

            Button("speak") {
                kokoro.speak(text: text, voice: .afAlloy)
            }
        }
        .padding()
        .task {
            kokoro.loadVoices([.afAlloy])
            kokoro.speak(text: text, voice: .afAlloy)
        }
    }

    func getTokens() {
//        pos()
        result = ""
    }

    func convert() {
        let tts = KokoroTTS()
        do {
            try tts.convertAllVoices()
        } catch {
            print("error: \(error.localizedDescription)")
        }
    }

    func loadSafeTensorVoices() {
        kokoro.loadVoices([.afAlloy])
    }
}

#Preview {
    ContentView()
}
