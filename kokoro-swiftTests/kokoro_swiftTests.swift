//
//  kokoro_swiftTests.swift
//  kokoro-swiftTests
//
//  Created by Ilia Sazonov on 3/10/25.
//

import Testing
@testable import kokoro_swift
import Foundation

struct kokoro_swiftTests {

    @Test func test_speak2file() async throws {
        let text = """
        Once upon a time, in a valley wrapped in mist and mystery, there was a little village where the stars whispered secrets to those who dared to listen.
        """
        let waveFileUrl = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test.wav")
        let kokoro = KokoroTTS()
        kokoro.loadVoices([.afAlloy])
        try kokoro.speak2file(text: text, voice: .afAlloy, fileURL: waveFileUrl)
        debugPrint("Test wave file created: \(waveFileUrl.absoluteString)")
    }
}
