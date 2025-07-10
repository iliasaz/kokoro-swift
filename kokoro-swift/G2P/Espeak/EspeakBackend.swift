//
//  EspeakBackend.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/30/25.
//

import Foundation
import libespeak_ng


// MARK: - Espeak Backend & Loader Stubs

/// A stub for the espeak-ng backend. In a full implementation, this class will call the espeak‑ng library.
class EspeakBackend {
    let language: String
    let preservePunctuation: Bool
    let withStress: Bool
    let tie: String
    let languageSwitch: String?

    let punctuator: Punctuation

    init(language: String, preservePunctuation: Bool, withStress: Bool, tie: String, languageSwitch: String? = nil) throws {
        self.language = language
        self.preservePunctuation = preservePunctuation
        self.withStress = withStress
        self.tie = tie
        self.languageSwitch = languageSwitch
        self.punctuator = Punctuation()
        let documentsDirectory = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let root = documentsDirectory
        try EspeakLib.ensureBundleInstalled(inRoot: root)
        espeak_ng_InitializePath(root.path)
        try espeak_ng_Initialize(nil).check()
        try espeak_ng_SetVoiceByName(ESPEAKNG_DEFAULT_VOICE).check()
        try espeak_ng_SetPhonemeEvents(1, 0).check()
    }

    deinit {
        let terminateOK = espeak_Terminate()
        print("ESpeakNGEngine termination OK: \(terminateOK == EE_OK)")
     }
}

import Foundation

// MARK: - Separator Type
typealias Separator = String
let defaultSeparator: Separator = " " // You can adjust this as needed

//struct Separator {
//    var phone: String
//    var word: String
//    var syllable: String? = nil  // optional — used only in some backends like Festival
//
//    static let defaultSeparator = Separator(phone: "", word: " ", syllable: nil)
//}

// MARK: - EspeakBackend Extension for Phonemize

extension EspeakBackend {

    // Define the regex once (static lazy to avoid recompilation)
    private static let espeakStressRegex: NSRegularExpression = {
        do {
            return try NSRegularExpression(pattern: "[ˈˌ'-]+", options: [])
        } catch {
            fatalError("Failed to compile stress regex: \(error)")
        }
    }()

    /// Phonemizes the given list of text lines.
    ///
    /// - Parameters:
    ///   - text: An array of strings; each string is a separate utterance.
    ///   - separator: A separator to be used between phonemes, syllables, etc.
    ///   - strip: If true, the output will not include trailing separators.
    /// - Returns: An array of phonemized strings.
    ///
    /// - Note: If `text` is empty or contains only empty utterances, those will be ignored.
    func phonemize(text: [String],
                   separator: Separator? = nil,
                   strip: Bool = false) -> [String] {

        // In Python, a string input would raise an error, but our parameter is already [String].
        let sep = separator ?? defaultSeparator

        // Preprocess the text and extract any punctuation marks.
        let (preprocessedText, punctuationMarks) = self._phonemizePreprocess(text: text)

        var phonemized: [String] = []
        phonemized = self._phonemizeAux(text: preprocessedText, offset: 0, separator: sep, strip: strip)

        return self._phonemizePostprocess(phonemized: phonemized, punctuationMarks: punctuationMarks, separator: sep, strip: strip)
    }

    // MARK: - Helper Methods (Stubs)
    /// Preprocesses the input text before phonemization.
    ///
    /// - Parameter text: An array of strings, each representing a separate utterance.
    /// - Returns: A tuple containing the processed text and a list of punctuation marks.
    private func _phonemizePreprocess(text: [String]) -> ([String], [MarkIndex]) {
        if self.preservePunctuation {
            // Preserve punctuation while returning both the processed text and the punctuation marks.
            return self.punctuator.preserve(text: text)
        }
        // Remove punctuation and return an empty punctuation list.
        return (self.punctuator.remove(text: text), [])
    }

    /// Postprocesses the raw phonemized output by restoring punctuation if required.
    ///
    /// - Parameters:
    ///   - phonemized: The array of phonemized strings.
    ///   - punctuationMarks: The punctuation marks extracted during preprocessing.
    ///   - separator: The separator used between phonemes.
    ///   - strip: A flag indicating whether trailing separators should be removed.
    /// - Returns: The final phonemized text with punctuation restored if needed.
    private func _phonemizePostprocess(phonemized: [String],
                                       punctuationMarks: [MarkIndex],
                                       separator: Separator,
                                       strip: Bool) -> [String] {
        if self.preservePunctuation {
            return Punctuation.restore(text: phonemized,
                                           marks: punctuationMarks,
                                           sep: separator,
                                           strip: strip)
        }
        return phonemized
    }

    /// Performs the core phonemization on the preprocessed text.
    ///
    /// - Parameters:
    ///   - text: The preprocessed text.
    ///   - offset: An integer offset (used for chunking in parallel processing).
    ///   - separator: The separator string.
    ///   - strip: Whether to strip trailing separators.
    /// - Returns: An array of phonemized strings.
    private func _phonemizeAux(text: [String], offset: Int, separator: Separator, strip: Bool) -> [String] {
        var output: [String] = []
        // TODO: process language switches
//        var langSwitches: [Int] = []

        for (index, line) in text.enumerated() {
            let lineNumber = index + 1

            // Phonemize using espeak binding
            let phonemeLine = self.espeakTextToPhonemes(line)

            // Postprocess (e.g. handle stress, separator, stripping)
            let (processed, hasSwitch) = self._postprocessLine(line: phonemeLine,
                                                               num: lineNumber,
                                                               separator: separator,
                                                               strip: strip)

            output.append(processed)

            // TODO: process language switches
//            if hasSwitch {
//                langSwitches.append(lineNumber + offset)
//            }
        }
        // TODO: process language switches
//        return (output, langSwitches)

        return output
    }

    private func espeakTextToPhonemes(_ line: String) -> String {
        let textMode: Int32 = 1 // UTF-8
        let tierCharacter = "͡"
        let phonemeMode: Int32 = Int32(0x02 | 0x01 << 7 | tierCharacter.unicodeScalars.first!.value << 8)
        var result = [String]()
        var phonemesLine: String = ""
        line.withCString { cString in
            var cStringPointer: UnsafePointer<CChar>? = cString
            let rawPtr = withUnsafeMutablePointer(to: &cStringPointer) {
                $0.withMemoryRebound(to: UnsafeRawPointer?.self, capacity: 1) { $0 }
            }
            let phonemesPtr = espeak_TextToPhonemes(rawPtr, textMode, phonemeMode)
            if let phonemesCStr = phonemesPtr {
                phonemesLine = String(cString: phonemesCStr)
            }
        }
        return " " + phonemesLine
    }

    private func _postprocessLine(line: String, num: Int, separator: Separator, strip: Bool) -> (String, Bool) {
        // 1. Clean the line: trim whitespace and replace newlines and double-spaces.
        var line = line.trimmingCharacters(in: .whitespacesAndNewlines)
        line = line.replacingOccurrences(of: "\n", with: " ")
        line = line.replacingOccurrences(of: "  ", with: " ")

        // 2. Fix extra underscores due to an espeak-ng bug.
        do {
            let underscoreRegex = try NSRegularExpression(pattern: "_+", options: [])
            let range = NSRange(location: 0, length: line.utf16.count)
            line = underscoreRegex.stringByReplacingMatches(in: line, options: [], range: range, withTemplate: "_")
        } catch {
            // Handle regex error if needed.
        }
        do {
            let underscoreSpaceRegex = try NSRegularExpression(pattern: "_ ", options: [])
            let range = NSRange(location: 0, length: line.utf16.count)
            line = underscoreSpaceRegex.stringByReplacingMatches(in: line, options: [], range: range, withTemplate: " ")
        } catch {
            // Handle regex error if needed.
        }

        // 3. Process language switches.
        // Assume _langSwitch.process(line:) returns a tuple (processedLine, hasSwitch).
        // TODO: imlpement language switch detection
//        let (processedLine, hasSwitch) = self._langSwitch.process(line: line)
//        if processedLine.isEmpty {
//            return ("", hasSwitch)
//        }

        let processedLine = line
        let hasSwitch = false

        // 4. Process each word in the line.
        var outLine = ""
        let words = processedLine.split(separator: " ").map { String($0) }
        for var word in words {
            // Process stress markers.
            word = self._processStress(word: word.trimmingCharacters(in: .whitespaces))
            // If not stripping and no tie is set, append an underscore.
            if !strip && self.tie == nil {
                word += "_"
            }
            // Process tie characters.
            word = self._processTie(word: word, separator: separator)
            outLine += word + defaultSeparator
        }

        // 5. If stripping is requested and separator.word is not empty, remove the last word separator.
        if strip && !defaultSeparator.isEmpty {
            outLine = String(outLine.dropLast(defaultSeparator.count))
        }

        return (outLine, hasSwitch)
    }

    /// Removes espeak stress markers unless `withStress` is enabled.
    /// Equivalent to: `re.sub(r"[ˈˌ'-]+", "", word)`
    private func _processStress(word: String) -> String {
        if self.withStress {
            return word
        }
        let range = NSRange(location: 0, length: word.utf16.count)
        return Self.espeakStressRegex.stringByReplacingMatches(in: word, options: [], range: range, withTemplate: "")
    }

    func _processTie(word: String, separator: Separator) -> String {
        // NOTE: We do not correct espeak's behavior with tie markers in language flags like (͡e͡n)
        if self.tie != "͡" {
            return word.replacingOccurrences(of: "͡", with: tie)
        }
        return word.replacingOccurrences(of: "_", with: defaultSeparator)
    }
}

