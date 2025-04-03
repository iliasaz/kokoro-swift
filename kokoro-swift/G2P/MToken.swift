//
//  MToken.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/28/25.
//

import Foundation

struct MToken {
    var text: String
    var tag: String
    var whitespace: String
    var isHead: Bool = true
    var alias: String? = nil
    var phonemes: String? = nil
    // Using Double to represent both Int and Float values
    var stress: Double? = nil
    var currency: String? = nil
    var numFlags: String = ""
    var prespace: Bool = false
    var rating: Int? = nil
    var startTs: Double? = nil
    var endTs: Double? = nil

    // Merges an array of tokens into one
    static func mergeTokens(_ tokens: [MToken], unk: String? = nil) -> MToken {
        // Gather non-nil stress values into a set
        let stresses = Set(tokens.compactMap { $0.stress })
        // Gather non-nil currencies
        let currencies = Set(tokens.compactMap { $0.currency })

        // Gather ratings. If any token has a nil rating, the merged rating will be nil.
        let ratings = tokens.map { $0.rating }
        let mergedRating: Int? = ratings.contains(where: { $0 == nil }) ? nil : ratings.compactMap { $0 }.min()

        // Merge phonemes based on unk
        var mergedPhonemes: String? = nil
        if let unk = unk {
            var phonemesStr = ""
            for t in tokens {
                if t.prespace && !phonemesStr.isEmpty,
                   let lastChar = phonemesStr.last,
                   !lastChar.isWhitespace,
                   t.phonemes != nil {
                    phonemesStr += " "
                }
                phonemesStr += t.phonemes ?? unk
            }
            mergedPhonemes = phonemesStr
        }

        // Build merged text by joining (text + whitespace) for all but the last token,
        // then appending the last token's text.
        let textPart = tokens.dropLast().map { $0.text + $0.whitespace }.joined()
        let mergedText = textPart + (tokens.last?.text ?? "")

        // Compute a score for each token based on its text:
        // add 1 for lowercase characters, 2 for uppercase (or non-lowercase) characters.
        func score(for token: MToken) -> Int {
            return token.text.reduce(0) { $0 + ($1.isLowercase ? 1 : 2) }
        }
        // Choose the token with the highest score to determine the merged tag.
        let bestToken = tokens.max { score(for: $0) < score(for: $1) }
        let mergedTag = bestToken?.tag ?? ""

        // Merged whitespace comes from the last token.
        let mergedWhitespace = tokens.last?.whitespace ?? ""
        // If there is exactly one unique stress value, use it; otherwise, nil.
        let mergedStress = (stresses.count == 1 ? stresses.first : nil)
        // Use the lexicographically maximum currency, if any.
        let mergedCurrency = currencies.isEmpty ? nil : currencies.max()

        // Merge numFlags by collecting all unique characters and sorting them.
        var flagSet = Set<Character>()
        for t in tokens {
            for c in t.numFlags {
                flagSet.insert(c)
            }
        }
        let mergedNumFlags = String(flagSet.sorted())

        // Use first token's prespace and startTs, last token's endTs.
        let mergedPrespace = tokens.first?.prespace ?? false
        let mergedStartTs = tokens.first?.startTs
        let mergedEndTs = tokens.last?.endTs

        return MToken(text: mergedText,
                      tag: mergedTag,
                      whitespace: mergedWhitespace,
                      isHead: tokens.first?.isHead ?? true,
                      alias: nil,
                      phonemes: mergedPhonemes,
                      stress: mergedStress,
                      currency: mergedCurrency,
                      numFlags: mergedNumFlags,
                      prespace: mergedPrespace,
                      rating: mergedRating,
                      startTs: mergedStartTs,
                      endTs: mergedEndTs)
    }

    // Checks if the token represents a "to" word.
    func isTo() -> Bool {
        return (text == "to" || text == "To") || (text == "TO" && (tag == "TO" || tag == "IN"))
    }

    // Returns debugging information as an array.
    func debugAll() -> [Any] {
        // If phonemes is nil, show ❓; if it's empty, show 🥷; otherwise show the phonemes.
        let ps: String
        if let phonemes = self.phonemes {
            ps = phonemes.isEmpty ? "🥷" : phonemes
        } else {
            ps = "❓"
        }

        // Determine a rating string based on the rating value.
        let rt: String
        if let rating = self.rating {
            if rating >= 5 {
                rt = "💎(5/5)"
            } else if rating == 4 {
                rt = "🏆(4/5)"
            } else if rating == 3 {
                rt = "🥈(3/5)"
            } else {
                rt = "🥉(2/5)"
            }
        } else {
            rt = "❓(UNK)"
        }

        return [text, tag, !whitespace.isEmpty, ps, rt]
    }
}
