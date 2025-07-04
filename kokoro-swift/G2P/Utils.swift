//
//  Utils.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/30/25.
//

import Foundation

// MARK: - Global Constants and Utility Functions

/**
 # Utility Functions Documentation

 This file implements a grapheme-to-phoneme conversion library using the `Lexicon` class, along with several global constants and utility functions. Below is an overview of the components:

 ## Global Constants & Utility Functions

 - **DIPHTHONGS**
   A set of characters (e.g., "AIOQWYʤʧ") used to determine stress weight in phoneme strings.

 - **stressWeight(_:)**
   Calculates a weight for a phoneme string by summing 2 for each diphthong and 1 for other characters.

 - **TokenContext (struct)**
   Holds contextual information during phoneme lookup:
   • `futureVowel`: Optional flag indicating if a vowel is expected soon.
   • `futureTo`: A Boolean flag for handling the word "to".

 - **subtokenize(_:)**
   Splits a word into subtokens using a Unicode-aware regular expression.

 - **LINK_PATTERN & linkRegex**
   Used to identify markdown-style links within text.

 - **SUBTOKEN_JUNKS, PUNCTS, NON_QUOTE_PUNCTS, PUNCT_TAGS, PUNCT_TAG_PHONEMES**
   Sets and dictionaries defining punctuation and token properties.

 - **LEXICON_ORDS**
   A set of Unicode scalar values representing valid characters for the lexicon.

 - **CONSONANTS & US_TAUS**
   Character sets used for phonetic classification of consonants and certain vowels.

 - **CURRENCIES, ORDINALS, ADD_SYMBOLS, SYMBOLS**
   Mappings for converting currency symbols, ordinal suffixes, and other special symbols to their phonetic representations.

 - **US_VOCAB & GB_VOCAB**
   Allowed phoneme symbol sets for US and British English, respectively.

 - **STRESSES, PRIMARY_STRESS, SECONDARY_STRESS, VOWELS**
   Constants used for marking stress in phoneme strings.

 - **rsplit(_:separator:maxSplits:)**
   Splits a string from the right based on a given separator.

 - **isDigit(_:)**
   Checks whether a string is composed entirely of digits.

 - **num2words(_:, to:)**
   Converts a number to its word representation using `NumberFormatter` (.spellOut or .ordinal styles).

 - **String Extension**
   • `isAlphabetic`: Returns true if the string contains only letters.
   • `subscript(safe:)`: Provides safe index access for characters.

 - **numericIfNeeded(_:)**
   Converts a numeric character to its standard numeric representation if required.

 - **lexiconIsNumber(_:, isHead:)**
   Determines if a word (after removing common suffixes) represents a number.

 - **ordinal(of:)**
   Retrieves the Unicode scalar value (ordinal) of a character.

 - **applyStress(_:, stress:)**
   Adjusts a phoneme string by applying stress markers based on a given stress value.

 */

let DIPHTHONGS: Set<Character> = Set("AIOQWYʤʧ")

func stressWeight(_ ps: String) -> Int {
    guard !ps.isEmpty else { return 0 }
    return ps.reduce(0) { $0 + (DIPHTHONGS.contains($1) ? 2 : 1) }
}

struct TokenContext {
    var futureVowel: Bool? = nil
    var futureTo: Bool = false
}

// Subtokenize – using NSRegularExpression with a Unicode‐aware pattern.
let SUBTOKEN_PATTERN = #"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$"#
let subtokenRegex = try! NSRegularExpression(pattern: SUBTOKEN_PATTERN, options: [])

func subtokenize(_ word: String) -> [String] {
    let nsWord = word as NSString
    let matches = subtokenRegex.matches(in: word, options: [], range: NSRange(location: 0, length: nsWord.length))
    return matches.map { nsWord.substring(with: $0.range) }
}

let LINK_PATTERN = #"\[([^\]]+)\]\(([^\)]*)\)"#
let linkRegex = try! NSRegularExpression(pattern: LINK_PATTERN, options: [])

// Other constants
let SUBTOKEN_JUNKS: Set<Character> = Set("',-._‘’/")
let PUNCTS: Set<Character> = Set(";:,.!?—…\"“”")
let NON_QUOTE_PUNCTS: Set<Character> = PUNCTS.subtracting(Set("\"“”"))
let PUNCT_TAGS: Set<String> = [".", ",", "-LRB-", "-RRB-", "``", "''", ":","$", "#", "NFP"]
let PUNCT_TAG_PHONEMES: [String: String] = [
    "-LRB-": "(",
    "-RRB-": ")",
    "``": String(UnicodeScalar(8220)!),
    "''": String(UnicodeScalar(8221)!),
    "\"\"": String(UnicodeScalar(8221)!)
]

let LEXICON_ORDS: Set<UInt32> = {
    var s = Set<UInt32>()
    s.insert(39)
    s.insert(45)
    for i in 65...90 { s.insert(UInt32(i)) }
    for i in 97...122 { s.insert(UInt32(i)) }
    return s
}()

let CONSONANTS: Set<Character> = Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
let US_TAUS: Set<Character> = Set("AIOWYiuæɑəɛɪɹʊʌ")

let CURRENCIES: [String: (String, String)] = [
    "$": ("dollar", "cent"),
    "£": ("pound", "pence"),
    "€": ("euro", "cent")
]

let ORDINALS: Set<String> = Set(["st", "nd", "rd", "th"])
let ADD_SYMBOLS: [String: String] = [".": "dot", "/": "slash"]
let SYMBOLS: [String: String] = ["%": "percent", "&": "and", "+": "plus", "@": "at"]

let US_VOCAB: Set<Character> = Set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ")
let GB_VOCAB: Set<Character> = Set("AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ")

let STRESSES = "ˌˈ"
let PRIMARY_STRESS: Character = STRESSES.last!   // 'ˈ'
let SECONDARY_STRESS: Character = STRESSES.first!  // 'ˌ'
let VOWELS: Set<Character> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")

// A helper for splitting from the right.
func rsplit(_ s: String, separator: Character, maxSplits: Int) -> [String] {
    let splits = s.split(separator: separator, maxSplits: maxSplits, omittingEmptySubsequences: false)
    return splits.map { String($0) }
}

// Check if a string is all digits.
func isDigit(_ text: String) -> Bool {
    let regex = try! NSRegularExpression(pattern: #"^[0-9]+$"#)
    let range = NSRange(location: 0, length: text.utf16.count)
    return regex.firstMatch(in: text, options: [], range: range) != nil
}

// Spell out numbers
func num2words(_ value: Double, to style: String? = nil) -> String {
    let formatter = NumberFormatter()
    // Use .ordinal if the style parameter is "ordinal", otherwise default to spellOut
    if let style = style, style.lowercased() == "ordinal" {
        formatter.numberStyle = .ordinal
    } else {
        formatter.numberStyle = .spellOut
    }
    return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
}

// Extension to check if a string is alphabetic.
extension String {
    var isAlphabetic: Bool {
        return self.allSatisfy { $0.isLetter }
    }

    // Safe subscript access by Index.
    subscript(safe index: Index) -> Character? {
        return indices.contains(index) ? self[index] : nil
    }
}

// Converts a character to its numeric form if needed.
func numericIfNeeded(_ c: Character) -> String {
    if !c.isNumber { return String(c) }
    if let value = c.wholeNumberValue {
        return String(value)
    }
    return String(c)
}

// Check if a word (after stripping suffixes) is a number.
func lexiconIsNumber(_ word: String, isHead: Bool) -> Bool {
    if word.allSatisfy({ !$0.isNumber }) { return false }
    let suffixes = ["ing", "'d", "ed", "'s"] + Array(ORDINALS) + ["s"]
    var modified = word
    for s in suffixes {
        if modified.hasSuffix(s) {
            modified = String(modified.dropLast(s.count))
            break
        }
    }
    for (i, c) in modified.enumerated() {
        if c.isNumber || ",.".contains(c) || (isHead && i == 0 && c == "-") {
            continue
        } else {
            return false
        }
    }
    return true
}

// Get the Unicode scalar (ordinal) value of a character.
func ordinal(of c: Character) -> UInt32? {
    return c.unicodeScalars.first?.value
}

// Applies stress modifications to a phoneme string.
func applyStress(_ ps: String, stress: Double?) -> String {
    func restress(_ ps: String) -> String {
        var ips = ps.enumerated().map { (Double($0.offset), $0.element) }
        var stresses: [Int: Double] = [:]
        for i in 0..<ips.count {
            let (_, char) = ips[i]
            if String(STRESSES).contains(char) {
                if let nextIndex = ips[i...].first(where: { VOWELS.contains($0.1) })?.0 {
                    stresses[i] = nextIndex
                }
            }
        }
        for (i, newIndex) in stresses {
            let orig = ips[i].1
            ips[i] = (newIndex - 0.5, orig)
        }
        let sortedIps = ips.sorted { $0.0 < $1.0 }
        return sortedIps.map { String($0.1) }.joined()
    }

    guard let stressVal = stress else { return ps }
    if stressVal < -1 {
        return ps.replacingOccurrences(of: String(PRIMARY_STRESS), with: "")
                 .replacingOccurrences(of: String(SECONDARY_STRESS), with: "")
    } else if stressVal == -1 || ((stressVal == 0 || stressVal == -0.5) && ps.contains(PRIMARY_STRESS)) {
        return ps.replacingOccurrences(of: String(SECONDARY_STRESS), with: "")
                 .replacingOccurrences(of: String(PRIMARY_STRESS), with: String(SECONDARY_STRESS))
    } else if [0, 0.5, 1].contains(stressVal) && !ps.contains(String(PRIMARY_STRESS)) && !ps.contains(String(SECONDARY_STRESS)) {
        if !ps.contains(where: { VOWELS.contains($0) }) { return ps }
        return restress(String(SECONDARY_STRESS) + ps)
    } else if stressVal >= 1 && !ps.contains(String(PRIMARY_STRESS)) && ps.contains(String(SECONDARY_STRESS)) {
        return ps.replacingOccurrences(of: String(SECONDARY_STRESS), with: String(PRIMARY_STRESS))
    } else if stressVal > 1 && !ps.contains(String(PRIMARY_STRESS)) && !ps.contains(String(SECONDARY_STRESS)) {
        if !ps.contains(where: { VOWELS.contains($0) }) { return ps }
        return restress(String(PRIMARY_STRESS) + ps)
    }
    return ps
}
