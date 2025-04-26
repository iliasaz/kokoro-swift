//
//  EspeakFallback.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/30/25.
//

import Foundation



/// A stub for setting the espeak-ng library and data paths.
class EspeakWrapper {
    static func setLibrary(path: String) {
        // Set the library path for espeak-ng.
    }

    static func setDataPath(path: String) {
        // Set the data path for espeak-ng.
    }
}

/// A stub loader for espeak-ng paths.
class EspeakngLoader {
    static func getLibraryPath() -> String {
        // Return the library path for espeak-ng.
        return "/path/to/espeak-ng/library"
    }

    static func getDataPath() -> String {
        // Return the data path for espeak-ng.
        return "/path/to/espeak-ng/data"
    }
}

// MARK: - EspeakFallback for English

/**
 EspeakFallback is used as a last resort for English.

 It uses the espeak-ng backend to generate a phoneme string for a given token,
 then applies a series of replacements to convert the espeak-ng output into the
 expected phonemic representation.
 */
class EspeakFallback {
    // Sorted mapping from espeak-ng output to the desired phoneme representation.
    static let E2M: [(String, String)] = {
        let mapping: [String: String] = [
            "ʔˌn\u{0329}": "tn",
            "ʔn\u{0329}": "tn",
            "ʔn": "tn",
            "ʔ": "t",
            "a^ɪ": "I",
            "a^ʊ": "W",
            "d^ʒ": "ʤ",
            "e^ɪ": "A",
            "e": "A",
            "t^ʃ": "ʧ",
            "ɔ^ɪ": "Y",
            "ə^l": "ᵊl",
            "ʲo": "jo",
            "ʲə": "jə",
            "ʲ": "",
            "ɚ": "əɹ",
            "r": "ɹ",
            "x": "k",
            "ç": "k",
            "ɐ": "ə",
            "ɬ": "l",
            "\u{0303}": ""
        ]
        // Sort keys by descending length.
        return mapping.sorted { $0.key.count > $1.key.count }
    }()

    let british: Bool
    var backend: EspeakBackend

    /**
     Initializes the EspeakFallback.

     - Parameter british: A Boolean indicating whether to use British English phoneme mappings.
     */
    init(british: Bool) throws {
        self.british = british

        // Set espeak-ng library and data paths.
        EspeakWrapper.setLibrary(path: EspeakngLoader.getLibraryPath())
        EspeakWrapper.setDataPath(path: EspeakngLoader.getDataPath())

        let language = british ? "en-gb" : "en-us"
        try self.backend = EspeakBackend(language: language, preservePunctuation: true, withStress: true, tie: "^")
    }

    /**
     Phonemizes a given token using the espeak-ng backend.

     - Parameter token: An MToken representing a word.
     - Returns: A tuple of (phoneme string, rating). Returns (nil, nil) if phonemization fails.
     */
    func phonemize(token: MToken) -> (String?, Int?) {
        let psArray = backend.phonemize(text: [token.text])
        guard let first = psArray.first, !first.isEmpty else {
            return (nil, nil)
        }
        var ps = first.trimmingCharacters(in: .whitespacesAndNewlines)
        // Apply replacements defined in E2M.
        for (old, new) in EspeakFallback.E2M {
            ps = ps.replacingOccurrences(of: old, with: new)
        }
        // Replace any non-whitespace character followed by U+0329 with "ᵊ" followed by that character.
        do {
            let regex = try NSRegularExpression(pattern: "(\\S)\\u{0329}", options: [])
            let range = NSRange(location: 0, length: ps.utf16.count)
            ps = regex.stringByReplacingMatches(in: ps, options: [], range: range, withTemplate: "ᵊ$1")
        } catch {
            // Ignore regex errors.
        }
        // Remove any remaining combining character U+0329.
        if let scalar = UnicodeScalar(809) {
            ps = ps.replacingOccurrences(of: String(scalar), with: "")
        }
        if self.british {
            ps = ps.replacingOccurrences(of: "e^ə", with: "ɛː")
            ps = ps.replacingOccurrences(of: "iə", with: "ɪə")
            ps = ps.replacingOccurrences(of: "ə^ʊ", with: "Q")
        } else {
            ps = ps.replacingOccurrences(of: "o^ʊ", with: "O")
            ps = ps.replacingOccurrences(of: "ɜːɹ", with: "ɜɹ")
            ps = ps.replacingOccurrences(of: "ɜː", with: "ɜɹ")
            ps = ps.replacingOccurrences(of: "ɪə", with: "iə")
            ps = ps.replacingOccurrences(of: "ː", with: "")
        }
        ps = ps.replacingOccurrences(of: "o", with: "ɔ") // For espeak < 1.52.
        ps = ps.replacingOccurrences(of: "^", with: "")
        return (ps, 2)
    }
}

// MARK: - EspeakG2P for Non-English/CJK Languages

/**
 EspeakG2P is used for most non-English/CJK languages.

 It employs a similar approach as EspeakFallback but with a different mapping
 (E2M) and different text preprocessing (e.g., converting angles to quotes, etc.).
 */
class EspeakG2P {
    static let E2M: [(String, String)] = {
        let mapping: [String: String] = [
            "a^ɪ": "I",
            "a^ʊ": "W",
            "d^z": "ʣ",
            "d^ʒ": "ʤ",
            "e^ɪ": "A",
            "o^ʊ": "O",
            "ə^ʊ": "Q",
            "s^s": "S",
            "t^s": "ʦ",
            "t^ʃ": "ʧ",
            "ɔ^ɪ": "Y"
        ]
        // Sort keys by descending length.
        return mapping.sorted { $0.key.count > $1.key.count }
    }()

    let language: String
    var backend: EspeakBackend

    /**
     Initializes the EspeakG2P.

     - Parameter language: A string representing the target language (e.g., "fr", "es").
     */
    init(language: String) throws {
        self.language = language
        try self.backend = EspeakBackend(language: language, preservePunctuation: true, withStress: true, tie: "^", languageSwitch: "remove-flags")
    }

    /**
     Phonemizes a given text using the espeak-ng backend.

     - Parameter text: The input text.
     - Returns: A tuple of (phoneme string, nil). Returns an empty string if phonemization fails.
     */
    func phonemize(text: String) -> (String, Any?) {
        // Convert angles to curly quotes.
        var modifiedText = text.replacingOccurrences(of: "«", with: String(UnicodeScalar(8220)!))
                                  .replacingOccurrences(of: "»", with: String(UnicodeScalar(8221)!))
        // Convert parentheses to angles.
        modifiedText = modifiedText.replacingOccurrences(of: "(", with: "«")
                                       .replacingOccurrences(of: ")", with: "»")
        let psArray = backend.phonemize(text: [modifiedText])
        guard let first = psArray.first else { return ("", nil) }
        var ps = first.trimmingCharacters(in: .whitespacesAndNewlines)
        for (old, new) in EspeakG2P.E2M {
            ps = ps.replacingOccurrences(of: old, with: new)
        }
        // Remove tie characters and hyphens.
        ps = ps.replacingOccurrences(of: "^", with: "").replacingOccurrences(of: "-", with: "")
        // Convert angles back to parentheses.
        ps = ps.replacingOccurrences(of: "«", with: "(").replacingOccurrences(of: "»", with: ")")
        return (ps, nil)
    }
}
