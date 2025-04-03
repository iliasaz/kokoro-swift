//
//  ContentView.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/10/25.
//

import SwiftUI

struct ContentView: View {
    @State private var text = """
    Once upon a time, in a valley wrapped in mist and mystery, there was a little village where the stars whispered secrets to those who dared to listen. In this village lived a curious girl named Liora, who had a wild imagination and a knack for finding trouble—or maybe it just found her. Everyone said the woods beyond the hills were cursed, but Liora? She thought they looked like adventure.
    """
    @State private var result: String = ""

    var body: some View {
        VStack {
            TextField("story", text: $text)
            Button("do") {
                getTokens()
            }
            Text(result)
        }
        .padding()
    }

    func getTokens() {
//        pos()
        result = ""
    }
}

#Preview {
    ContentView()
}
