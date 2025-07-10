//
//  G2P.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/30/25.
//

import Foundation
import NaturalLanguage

// MARK: - Remapping from spacy to NLTagger POS names

func mapNLTagToSpacyTag(_ nlTag: NLTag) -> String {
    switch nlTag {
    case NLTag.noun:
        return "NN"
    case NLTag.verb:
        return "VB"
    case NLTag.adjective:
        return "JJ"
    case NLTag.adverb:
        return "RB"
    case NLTag.pronoun:
        return "PRP"
    case NLTag.determiner:
        return "DT"
    case NLTag.preposition:
        return "IN"
    case NLTag.particle:
        return "RP"
    case NLTag.number:
        return "CD"
    case NLTag.conjunction:
        return "CC"
    case NLTag.interjection:
        return "UH"
    default:
        return nlTag.rawValue  // Fallback: use the original rawValue if no mapping exists.
    }
}

// MARK: - NLPTok & NLPEngine

/// Represents a single NLP token with text, POS tag, and following whitespace.
class NLPTok {
    var text: String
    var tag: String
    var whitespace: String
    init(text: String, tag: String, whitespace: String) {
        self.text = text
        self.tag = tag
        self.whitespace = whitespace
    }
}

/// NLPEngine uses NLTagger from the NaturalLanguage framework to process text.
/// This replaces the spaCy dependency.
class NLPEngine {
    let tagger: NLTagger

    init() {
        // Use the lexicalClass scheme for POS tagging.
        self.tagger = NLTagger(tagSchemes: [.lexicalClass])
    }

    /// Processes the given text into an array of NLPTok tokens.
    /// The tokenization uses NLTagger and attempts to capture whitespace between tokens.
    ///
    /// - Parameter text: The input text.
    /// - Returns: An array of NLPTok tokens.
    func process(text: String) -> [NLPTok] {
        self.tagger.string = text
        var tokens = [NLPTok]()

        // Collect token ranges using NLTagger.
        var tokenRanges = [Range<String.Index>]()
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]
        self.tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass, options: options) { tag, tokenRange in
            tokenRanges.append(tokenRange)
            return true
        }

        // Compute whitespace by examining the gap between tokens.
        for (index, tokenRange) in tokenRanges.enumerated() {
            let tokenText = String(text[tokenRange])
            let (tag, _) = self.tagger.tag(at: tokenRange.lowerBound, unit: .word, scheme: .lexicalClass)
            let spacyTag = tag.map(mapNLTagToSpacyTag) ?? "NN"
            var whitespace = ""
            if index < tokenRanges.count - 1 {
                let currentEnd = tokenRange.upperBound
                let nextStart = tokenRanges[index + 1].lowerBound
                whitespace = String(text[currentEnd..<nextStart])
            } else {
                // Default to a single space if no trailing whitespace is found.
                whitespace = " "
            }
            tokens.append(NLPTok(text: tokenText, tag: spacyTag, whitespace: whitespace))
        }
        return tokens
    }
}

// MARK: - G2P Class

/**
 Core grapheme-to-phoneme conversion class.

 This class converts input text to a phonetic representation. It uses NLTagger
 (via NLPEngine) for tokenization and part-of-speech tagging, and a Lexicon instance
 to look up phoneme mappings.

 Espeak/spaCy functionality has been replaced with NLTagger stubs.
 */
class G2P {
    let british: Bool
    let nlp: NLPEngine
    let lexicon: Lexicon
    // Fallback functionality is stubbed out.
    var fallback: ((MToken) -> (String?, Int?))? = nil
    let unk: String
    let useEspeak: Bool
    let espeak: EspeakFallback
    /**
     Initializes the G2P engine.

     - Parameters:
       - british: Whether to use British English phoneme mappings.
       - unk: The unknown phoneme symbol.
       - useEspeak: true if espeak-ng should be used instead of misaki G2P backend
     */
//    init(trf: Bool = false, british: Bool = false, fallback: ((MToken) -> (String?, Int?))? = nil, unk: String = "❓") {
    init(british: Bool = false, unk: String = "❓", useEspeak: Bool = true) throws {
        self.british = british
        // Initialize NLTagger-based engine.
        self.nlp = NLPEngine()
        self.lexicon = Lexicon(british: british)
        self.fallback = nil
        self.unk = unk
        self.useEspeak = useEspeak
        try espeak = EspeakFallback(british: british)
    }

    /**
     Preprocesses the input text to separate out markdown-style links and extract features.

     - Parameter text: The input text.
     - Returns: A tuple containing the preprocessed text, an array of token strings, and a features dictionary.
     */
    static func preprocess(text: String) -> (String, [String], [Int: Any]) {
        var result = ""
        var tokens = [String]()
        var features = [Int: Any]()
        var lastEnd = text.startIndex
        let trimmedText = text.trimmingCharacters(in: .whitespaces)
        let nsText = trimmedText as NSString
        let matches = linkRegex.matches(in: trimmedText, options: [], range: NSRange(location: 0, length: nsText.length))
        for match in matches {
            guard let range0 = Range(match.range, in: trimmedText),
                  let range1 = Range(match.range(at: 1), in: trimmedText),
                  let range2 = Range(match.range(at: 2), in: trimmedText)
            else { continue }
            let prefix = String(trimmedText[lastEnd..<range0.lowerBound])
            result += prefix
            tokens.append(contentsOf: prefix.split(separator: " ").map { String($0) })
            var fStr = String(trimmedText[range2])
            if !fStr.isEmpty {
                let offset = (fStr.first == "-" || fStr.first == "+") ? 1 : 0
                let fSub = String(fStr.dropFirst(offset))
                if isDigit(fSub) {
                    if let intVal = Int(fSub) {
                        fStr = "\(intVal)"
                    }
                } else if fStr == "0.5" || fStr == "+0.5" {
                    fStr = "0.5"
                } else if fStr == "-0.5" {
                    fStr = "-0.5"
                } else if fStr.first == "/" && fStr.last == "/" && fStr.count > 1 {
                    fStr = "/" + fStr.dropFirst().trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                } else if fStr.first == "#" && fStr.last == "#" && fStr.count > 1 {
                    fStr = "#" + fStr.dropFirst().trimmingCharacters(in: CharacterSet(charactersIn: "#"))
                } else {
                    fStr = ""
                }
                if !fStr.isEmpty {
                    features[tokens.count] = fStr
                }
            }
            let group1 = String(trimmedText[range1])
            result += group1
            tokens.append(group1)
            lastEnd = range0.upperBound
        }
        if lastEnd < trimmedText.endIndex {
            let remainder = String(trimmedText[lastEnd...])
            result += remainder
            tokens.append(contentsOf: remainder.split(separator: " ").map { String($0) })
        }
        return (result, tokens, features)
    }

    /**
     Tokenizes the preprocessed text using NLTagger and adjusts tokens based on features.

     - Parameters:
       - text: The preprocessed text.
       - tokens: An array of token strings from preprocessing.
       - features: A dictionary of features keyed by token index.
     - Returns: An array of MToken objects.
     */
    func tokenize(text: String, tokens: [String], features: [Int: Any]) -> [MToken] {
        let doc = nlp.process(text: text)
        var mutableTokens = doc.map { t in
            return MToken(text: t.text, tag: t.tag, whitespace: t.whitespace)
        }
        if features.isEmpty {
            return mutableTokens
        }
        // Simple 1:1 alignment.
        for (k, v) in features {
            if k < mutableTokens.count {
                if let stressStr = v as? String {
                    if stressStr.hasPrefix("/") {
                        mutableTokens[k].isHead = (k == 0)
                        mutableTokens[k].phonemes = String(stressStr.drop(while: { $0 == "/" }))
                        mutableTokens[k].rating = 5
                    } else if stressStr.hasPrefix("#") {
                        mutableTokens[k].numFlags = String(stressStr.drop(while: { $0 == "#" }))
                    }
                } else if let numVal = Double("\(v)") {
                    mutableTokens[k].stress = numVal
                }
            }
        }
        return mutableTokens
    }

    /**
     Merges tokens from left to right if a token is not marked as the head.

     - Parameter tokens: The array of tokens.
     - Returns: A new array of merged MToken objects.
     */
    func foldLeft(tokens: [MToken]) -> [MToken] {
        var result = [MToken]()
        for t in tokens {
            if let last = result.last, !t.isHead {
                result.removeLast()
                let merged = MToken.mergeTokens([last, t], unk: self.unk)
                result.append(merged)
            } else {
                result.append(t)
            }
        }
        return result
    }

    /**
     This definition lets you work with arrays of tokens where some elements might be a single token and others are groups of tokens. In the retokenize method, you later flatten these groups (e.g., groups with a single token become just a single token)
     */
    enum TokenGroup {
        case single(MToken)
        case group([MToken])
    }

    /**
     Retokenizes tokens into groups based on special cases (e.g., punctuation, currency).

     - Parameter tokens: The original tokens.
     - Returns: An array of TokenGroup values.
     */
    static func retokenize(tokens: [MToken]) -> [TokenGroup] {
        var words = [TokenGroup]()
        var currency: String? = nil
        for token in tokens {
            var ts: [MToken]
            if token.alias == nil && token.phonemes == nil {
                let subtokens = subtokenize(token.text)
                ts = subtokens.map { subText in
                    var newToken = token
                    newToken.text = subText
                    newToken.whitespace = ""
                    return newToken
                }
            } else {
                ts = [token]
            }
            if !ts.isEmpty {
                ts[ts.count - 1].whitespace = token.whitespace
            }
            for (j, var t) in ts.enumerated() {
                if t.alias != nil || t.phonemes != nil {
                    // Already set.
                } else if t.tag == "$" && CURRENCIES.keys.contains(t.text) {
                    currency = t.text
                    t.phonemes = ""
                    t.rating = 4
                } else if t.tag == ":" && (t.text == "-" || t.text == "–") {
                    t.phonemes = "—"
                    t.rating = 3
                } else if PUNCT_TAGS.contains(t.tag) {
                    if let punct = PUNCT_TAG_PHONEMES[t.tag] {
                        t.phonemes = punct
                    } else {
                        t.phonemes = t.text.filter { PUNCTS.contains($0) }
                    }
                    t.rating = 4
                } else if let curr = currency {
                    if t.tag != "CD" {
                        currency = nil
                    } else if j + 1 == ts.count {
                        t.currency = curr
                    }
                } else if j > 0 && j < ts.count - 1 && t.text == "2" {
                    let prev = ts[j - 1]
                    let next = ts[j + 1]
                    if let lastChar = prev.text.last, let firstChar = next.text.first,
                       (String(lastChar) + String(firstChar)).isAlphabetic {
                        t.alias = "to"
                    }
                }
                if t.alias != nil || t.phonemes != nil {
                    words.append(.single(t))
                } else if case .group(var group) = words.last, group.last?.whitespace == "" {
                    t.isHead = false
                    group.append(t)
                    words[words.count - 1] = .group(group)
                } else {
                    if t.whitespace.isEmpty {
                        words.append(.group([t]))
                    } else {
                        words.append(.single(t))
                    }
                }
            }
        }
        // Flatten groups with only one token.
        var resultGroups = [TokenGroup]()
        for w in words {
            switch w {
            case .group(let arr) where arr.count == 1:
                resultGroups.append(.single(arr[0]))
            default:
                resultGroups.append(w)
            }
        }
        return resultGroups
    }

    /**
     Updates the token context based on the phoneme string and token.

     - Parameters:
       - ctx: The current TokenContext.
       - ps: The phoneme string (if any).
       - token: The current token.
     - Returns: An updated TokenContext.
     */
    static func tokenContext(ctx: TokenContext, ps: String?, token: MToken) -> TokenContext {
        var vowel = ctx.futureVowel
        if let ps = ps {
            for c in ps {
                if VOWELS.contains(c) || CONSONANTS.contains(c) || NON_QUOTE_PUNCTS.contains(c) {
                    vowel = NON_QUOTE_PUNCTS.contains(c) ? nil : VOWELS.contains(c)
                    break
                }
            }
        }
        return TokenContext(futureVowel: vowel, futureTo: token.isTo())
    }

    /**
     Resolves tokens by applying heuristics on punctuation and stress.

     - Parameter tokens: An inout array of MToken objects.
     */
    static func resolveTokens(tokens: inout [MToken]) {
        let text = tokens.dropLast().reduce("") { $0 + $1.text + $1.whitespace } + (tokens.last?.text ?? "")
        let prespace = text.contains(" ") || text.contains("/") ||
            (Set(text.filter { !SUBTOKEN_JUNKS.contains($0) }
                    .map { $0.isLetter ? 0 : (isDigit(String($0)) ? 1 : 2) }).count > 1)
        for i in tokens.indices {
            if tokens[i].phonemes == nil {
                if i == tokens.count - 1 && NON_QUOTE_PUNCTS.contains(where: { tokens[i].text.contains(String($0)) }) {
                    tokens[i].phonemes = tokens[i].text
                    tokens[i].rating = 3
                } else if tokens[i].text.allSatisfy({ SUBTOKEN_JUNKS.contains($0) }) {
                    tokens[i].phonemes = ""
                    tokens[i].rating = 3
                }
            } else if i > 0 {
                tokens[i].prespace = prespace
            }
        }
        if prespace { return }
        var indices: [(Bool, Int, Int)] = []
        for (i, token) in tokens.enumerated() {
            if let ps = token.phonemes {
                let containsPrimary = ps.contains(String(PRIMARY_STRESS))
                let weight = stressWeight(ps)
                indices.append((containsPrimary, weight, i))
            }
        }
        if indices.count == 2, let firstIndex = indices.first?.2, tokens[firstIndex].text.count == 1 {
            let i = indices[1].2
            if let ps = tokens[i].phonemes {
                tokens[i].phonemes = applyStress(ps, stress: -0.5)
            }
            return
        } else if indices.count < 2 || indices.map({ $0.0 ? 1 : 0 }).reduce(0, +) <= (indices.count + 1) / 2 {
            return
        }
        indices.sort { $0.1 < $1.1 }
        let halfCount = indices.count / 2
        for (_, _, i) in indices.prefix(halfCount) {
            if let ps = tokens[i].phonemes {
                tokens[i].phonemes = applyStress(ps, stress: -0.5)
            }
        }
    }

    /**
     Main entry point for G2P conversion.

     - Parameters:
       - text: The input text.
       - preprocessFlag: Whether to run preprocessing (default is true).
     - Returns: A tuple containing the final phoneme string and the list of tokens.
     */
    func callAsFunction(text: String, preprocess preprocessFlag: Bool = true) -> (String, [MToken]) {
        let (processedText, tokenStrs, features) = preprocessFlag ? G2P.preprocess(text: text) : (text, [String](), [Int: Any]())
        var tokens = self.tokenize(text: processedText, tokens: tokenStrs, features: features)
        tokens = self.foldLeft(tokens: tokens)
        let tokenGroups = G2P.retokenize(tokens: tokens)
        var flatTokens = [MToken]()
        for group in tokenGroups {
            switch group {
            case .single(let token):
                flatTokens.append(token)
            case .group(let tokens):
                flatTokens.append(contentsOf: tokens)
            }
        }
        var ctx = TokenContext()
        for i in stride(from: flatTokens.count - 1, through: 0, by: -1) {
            if self.useEspeak {
                let (ps, rating) = espeak.phonemize(token: flatTokens[i])
                flatTokens[i].phonemes = ps
                flatTokens[i].rating = rating
            } else {
                if flatTokens[i].phonemes == nil {
                    let (ps, rating) = self.lexicon(token: flatTokens[i], ctx: ctx)
                    flatTokens[i].phonemes = ps
                    flatTokens[i].rating = rating
                    if flatTokens[i].phonemes == nil, let fallback = self.fallback {
                        let (ps, rating) = fallback(flatTokens[i])
                        flatTokens[i].phonemes = ps
                        flatTokens[i].rating = rating
                    }
                }
            }
            ctx = G2P.tokenContext(ctx: ctx, ps: flatTokens[i].phonemes, token: flatTokens[i])
        }

        var mergedTokens = [MToken]()
        var currentGroup = [MToken]()

        var j = 0
        while j < flatTokens.count {
            let token = flatTokens[j]
            currentGroup.append(token)

            if token.whitespace.isEmpty {
                j += 1
                continue
            }

            // finalize this group
            if currentGroup.count == 1 {
                mergedTokens.append(currentGroup[0])
            } else {
                // Try to resolve phonemes in the group
                var left = 0
                var right = currentGroup.count
                var shouldFallback = false

                while left < right {
                    let groupSlice = Array(currentGroup[left..<right])
                    let merged = MToken.mergeTokens(groupSlice, unk: self.unk)

                    let (ps, rating) = self.lexicon(token: merged, ctx: ctx)

                    if let ps = ps {
                        // Assign merged result
                        currentGroup[left].phonemes = ps
                        currentGroup[left].rating = rating
                        for k in (left+1)..<right {
                            currentGroup[k].phonemes = ""
                            currentGroup[k].rating = rating
                        }
                        ctx = G2P.tokenContext(ctx: ctx, ps: ps, token: merged)
                        right = left
                        left = 0
                    } else if left + 1 < right {
                        left += 1
                    } else {
                        right -= 1
                        let lastToken = currentGroup[right]
                        if lastToken.phonemes == nil {
                            if lastToken.text.allSatisfy({ SUBTOKEN_JUNKS.contains($0) }) {
                                currentGroup[right].phonemes = ""
                                currentGroup[right].rating = 3
                            } else if self.fallback != nil {
                                shouldFallback = true
                                break
                            }
                        }
                        left = 0
                    }
                }

                if shouldFallback, let fallback = self.fallback {
                    let merged = MToken.mergeTokens(currentGroup, unk: self.unk)
                    let (ps, rating) = fallback(merged)
                    currentGroup[0].phonemes = ps
                    currentGroup[0].rating = rating
                    for k in 1..<currentGroup.count {
                        currentGroup[k].phonemes = ""
                        currentGroup[k].rating = rating
                    }
                } else {
                    G2P.resolveTokens(tokens: &currentGroup)
                }

                // Finally merge the group into one
                let merged = MToken.mergeTokens(currentGroup, unk: self.unk)
                mergedTokens.append(merged)
            }

            currentGroup.removeAll()
            j += 1
        }

        let result = mergedTokens.reduce("") { $0 + (( $1.phonemes ?? self.unk ) + $1.whitespace) }
        return (result, mergedTokens)
    }
}
