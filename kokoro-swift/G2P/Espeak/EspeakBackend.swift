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

    init(language: String, preservePunctuation: Bool, withStress: Bool, tie: String, languageSwitch: String? = nil) {
        self.language = language
        self.preservePunctuation = preservePunctuation
        self.withStress = withStress
        self.tie = tie
        self.languageSwitch = languageSwitch
        self.punctuator = Punctuation()
    }
}

import Foundation

// MARK: - Separator Type
typealias Separator = String
let default_separator: Separator = " " // You can adjust this as needed

// MARK: - EspeakBackend Extension for Phonemize

extension EspeakBackend {

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
        let sep = separator ?? default_separator

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
        // Stub: for now, simply join each line with the separator.
        // In a full implementation, call the underlying espeak-ng library.
        return text.map { line in
            // Here we could process each line further.
            return line.components(separatedBy: " ").joined(separator: separator)
        }
    }
}
