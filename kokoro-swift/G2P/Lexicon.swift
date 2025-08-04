//
//  Lexicon.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/28/25.
//

import Foundation

/**
 ## Lexicon Class

 The `Lexicon` class is the core component that converts text (graphemes) to phonemes.

 ### Properties
 - `british`: Boolean flag indicating whether to use British English phoneme mappings.
 - `capStresses`: Tuple holding stress values for capitalized words.
 - `golds` & `silvers`: Dictionaries loaded from JSON files containing primary and fallback phoneme mappings.

 ### Key Methods

 - **growDictionary(_:) (static)**
   Enhances a dictionary by adding alternate-case keys (e.g., capitalized and lowercased variants).

 - **init(british:)**
   Initializes the lexicon, loads JSON resources, and validates the vocabulary.

 - **getNNP(_:)**
   Processes proper nouns (NNP) by mapping each letter to its phoneme and applying stress. Returns a phoneme string and a rating.

 - **getSpecialCase(word:tag:stress:ctx:)**
   Handles special cases (symbols, abbreviations, etc.) by directing them to the appropriate phoneme lookup.

 - **getParentTag(tag:) (static)**
   Simplifies detailed part-of-speech tags to broader categories (e.g., "VB" to "VERB").

 - **isKnown(word:)**
   Checks if a word exists in the phoneme mappings or meets criteria to be considered "known."

 - **lookup(word:tag:stress:ctx:)**
   Retrieves a phoneme representation for a word from the golds/silvers dictionaries, handling sub-dictionaries if necessary.

 - **_s(_:)**
   Applies plural suffix rules, appending an "s" or a phonetic equivalent.

 - **stem_s(word:tag:stress:ctx:)**
   Strips and processes a plural "s" from a word, looks up the base form, and reapplies the plural transformation.

 - **_ed(_:) & stem_ed(word:tag:stress:ctx:)**
   Process the past tense ("-ed") suffix by converting the base form and reattaching the proper phoneme.

 - **_ing(_:) & stem_ing(word:tag:stress:ctx:)**
   Constructs the progressive ("-ing") form by processing the base form and attaching the "-ing" sound.

 - **get_word(word:tag:stress:ctx:)**
   The central method that decides how to process a word using special-case handling and various stemming strategies.

 - **isCurrency(_:) (static)**
   Determines if a word represents a valid currency format.

 - **get_number(word:currency:isHead:numFlags:)**
   Converts numeric strings into their word equivalents, handling digits, commas, decimals, and currency symbols.

 - **append_currency(_:, currency:)**
   Appends a phonetic representation of a currency unit to a phoneme string.

 - **isNumber(_:, isHead:) (static)**
   Checks if a word is numeric based on its formatting and content.

 - **call(token:ctx:)**
   The primary entry point for converting an `MToken` to its phoneme representation. It normalizes the token text, determines stress, performs lookups, and handles special cases (including numbers and mixed-case words).

 This documentation provides a high-level overview of the functionality. Each method and constant is designed to work together to perform detailed grapheme-to-phoneme conversion.

*/

// MARK: - Lexicon Class

class Lexicon {
    var british: Bool
    var capStresses: (Double, Double) = (0.5, 2)
    var golds: [String: Any] = [:]
    var silvers: [String: Any] = [:]

    // Grow a dictionary by adding alternate-case keys.
    static func growDictionary(_ d: [String: Any]) -> [String: Any] {
        var e: [String: Any] = [:]
        for (k, v) in d {
            if k.count < 2 { continue }
            if k == k.lowercased() {
                if k != k.capitalized {
                    e[k.capitalized] = v
                }
            } else if k == k.lowercased().capitalized {
                e[k.lowercased()] = v
            }
        }
        var merged = d
        for (k, v) in e { merged[k] = v }
        return merged
    }

    init(british: Bool) {
        self.british = british
        // Load JSON resources for golds and silvers.
        let goldFile = british ? "gb_gold" : "us_gold"
        let silverFile = british ? "gb_silver" : "us_silver"
        if let goldURL = Bundle.main.url(forResource: goldFile, withExtension: "json"),
           let goldData = try? Data(contentsOf: goldURL),
           let goldJson = try? JSONSerialization.jsonObject(with: goldData, options: []) as? [String: Any] {
            self.golds = Lexicon.growDictionary(goldJson)
        }
        if let silverURL = Bundle.main.url(forResource: silverFile, withExtension: "json"),
           let silverData = try? Data(contentsOf: silverURL),
           let silverJson = try? JSONSerialization.jsonObject(with: silverData, options: []) as? [String: Any] {
            self.silvers = Lexicon.growDictionary(silverJson)
        }
        // Validate golds vocab.
        for v in self.golds.values {
            if let s = v as? String {
                let vocab = self.british ? GB_VOCAB : US_VOCAB
                for c in s where !vocab.contains(c) {
                    fatalError("Invalid vocab in golds: \(s)")
                }
            } else if let dict = v as? [String: String?] {
                guard dict["DEFAULT"] != nil else { fatalError("Missing DEFAULT in golds: \(dict)") }
                let vocab = self.british ? GB_VOCAB : US_VOCAB
                for val in dict.values {
                    if let str = val {
                        for c in str where !vocab.contains(c) {
                            fatalError("Invalid vocab in golds dict value: \(str)")
                        }
                    }
                }
            }
        }
    }

    func getNNP(_ word: String) -> (String?, Int?) {
        var psParts: [String] = []
        for c in word where c.isLetter {
            let key = String(c).uppercased()
            if let value = self.golds[key] as? String {
                psParts.append(value)
            } else {
                return (nil, nil)
            }
        }
        let joined = psParts.joined()
        let stressed = applyStress(joined, stress: 0)
        if let range = stressed.range(of: String(SECONDARY_STRESS), options: .backwards) {
            let left = stressed[..<range.lowerBound]
            let right = stressed[range.upperBound...]
            let result = left + String(PRIMARY_STRESS) + String(right)
            return (String(result), 3)
        } else {
            return (stressed, 3)
        }
    }

    func getSpecialCase(word: String, tag: String, stress: Double?, ctx: TokenContext) -> (String?, Int?) {
        if tag == "ADD", let addSymbol = ADD_SYMBOLS[word] {
            return self.lookup(word: addSymbol, tag: nil, stress: -0.5, ctx: ctx)
        } else if let sym = SYMBOLS[word] {
            return self.lookup(word: sym, tag: nil, stress: nil, ctx: ctx)
        } else if !word.trimmingCharacters(in: CharacterSet(charactersIn: ".")).contains(".") &&
                    word.replacingOccurrences(of: ".", with: "").isAlphabetic &&
                    ((word.split(separator: ".")).map { $0.count }.max() ?? 0) < 3 {
            return self.getNNP(word)
        } else if word == "a" || (word == "A" && tag == "DT") {
            return ("ɐ", 4)
        } else if ["am", "Am", "AM"].contains(word) {
            if tag.hasPrefix("NN") {
                return self.getNNP(word)
            } else if ctx.futureVowel == nil || word != "am" || (stress != nil && stress! > 0) {
                if let amVal = self.golds["am"] as? String {
                    return (amVal, 4)
                }
            }
            return ("ɐm", 4)
        } else if ["an", "An", "AN"].contains(word) {
            if word == "AN" && tag.hasPrefix("NN") {
                return self.getNNP(word)
            }
            return ("ɐn", 4)
        } else if word == "I" && tag == "PRP" {
            return ("\(SECONDARY_STRESS)I", 4)
        } else if ["by", "By", "BY"].contains(word) && Lexicon.getParentTag(tag: tag) == "ADV" {
            return ("bˈI", 4)
        } else if word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")) {
            let fv = ctx.futureVowel
            let result: String
            switch fv {
                case false:
                    result = "tə"
                case true:
                    result = "tʊ"
                default:
                    if let val = self.golds["to"] as? String { result = val } else { result = "" }
            }
            return (result, 4)
        } else if word.lowercased() == "the" || (word == "THE" && tag == "DT") {
            return (ctx.futureVowel == true ? "ði" : "ðə", 4)
        } else if tag == "IN" && word.range(of: #"(?i)vs\.?$"#, options: .regularExpression) != nil {
            return self.lookup(word: "versus", tag: nil, stress: nil, ctx: ctx)
        } else if ["used", "Used", "USED"].contains(word) {
            if (tag == "VBD" || tag == "JJ") && ctx.futureTo {
                if let usedDict = self.golds["used"] as? [String: String],
                   let val = usedDict["VBD"] {
                    return (val, 4)
                }
            }
            if let usedDict = self.golds["used"] as? [String: String],
               let val = usedDict["DEFAULT"] {
                return (val, 4)
            }
        }
        return (nil, nil)
    }

    static func getParentTag(tag: String?) -> String? {
        guard let tag = tag else { return tag }
        if tag.hasPrefix("VB") {
            return "VERB"
        } else if tag.hasPrefix("NN") {
            return "NOUN"
        } else if tag.hasPrefix("ADV") || tag.hasPrefix("RB") {
            return "ADV"
        } else if tag.hasPrefix("ADJ") || tag.hasPrefix("JJ") {
            return "ADJ"
        }
        return tag
    }

    func isKnown(word: String) -> Bool {
        if self.golds[word] != nil || SYMBOLS[word] != nil || self.silvers[word] != nil {
            return true
        } else if !word.isAlphabetic || !word.allSatisfy({
            if let o = ordinal(of: $0) { return LEXICON_ORDS.contains(o) }
            return false
        }) {
            return false
        } else if word.count == 1 {
            return true
        } else if word == word.uppercased(), self.golds[word.lowercased()] != nil {
            return true
        }
        let index = word.index(word.startIndex, offsetBy: 1)
        return word[index...] == word[index...].uppercased()
    }

    func lookup(word: String, tag: String?, stress: Double?, ctx: TokenContext?) -> (String?, Int?) {
        var wordVar = word
        var isNNP: Bool? = nil
        if wordVar == wordVar.uppercased() && self.golds[wordVar] == nil {
            wordVar = wordVar.lowercased()
            isNNP = (tag == "NNP")
        }
        var ps: Any? = self.golds[wordVar]
        var rating = 4
        if ps == nil && isNNP != true {
            ps = self.silvers[wordVar]
            rating = 3
        }
        if let dict = ps as? [String: String] {
            var tagVar = tag
            if let ctx = ctx, ctx.futureVowel == nil, dict["None"] != nil {
                tagVar = "None"
            } else if let tagVarUnwrapped = tagVar, dict[tagVarUnwrapped] == nil {
                tagVar = Lexicon.getParentTag(tag: tagVar)
            }
            ps = dict[tagVar ?? ""] ?? dict["DEFAULT"]
        }
        if let psStr = ps as? String {
            if isNNP == true && !psStr.contains(PRIMARY_STRESS) {
                let nnp = self.getNNP(wordVar)
                return nnp
            }
            return (applyStress(psStr, stress: stress), rating)
        }
        return (nil, nil)
    }

    func _s(_ stem: String) -> String? {
        guard !stem.isEmpty else { return nil }
        if let last = stem.last, "ptkfθ".contains(last) {
            return stem + "s"
        } else if let last = stem.last, "szʃʒʧʤ".contains(last) {
            return stem + (self.british ? "ɪ" : "ᵻ") + "z"
        }
        return stem + "z"
    }

    func stem_s(word: String, tag: String?, stress: Double?, ctx: TokenContext) -> (String?, Int?) {
        var stem: String? = nil
        if word.count > 2 && word.hasSuffix("s") && !word.hasSuffix("ss") && self.isKnown(word: String(word.dropLast())) {
            stem = String(word.dropLast())
        } else if (word.hasSuffix("'s") || (word.count > 4 && word.hasSuffix("es") && !word.hasSuffix("ies"))) && self.isKnown(word: String(word.dropLast(2))) {
            stem = String(word.dropLast(2))
        } else if word.count > 4 && word.hasSuffix("ies") && self.isKnown(word: String(word.dropLast(3)) + "y") {
            stem = String(word.dropLast(3)) + "y"
        } else {
            return (nil, nil)
        }
        let lookupResult = self.lookup(word: stem!, tag: tag, stress: stress, ctx: ctx)
        if let lookupStem = lookupResult.0 {
            return (self._s(lookupStem), lookupResult.1)
        }
        return (nil, nil)
    }

    func _ed(_ stem: String) -> String? {
        guard !stem.isEmpty else { return nil }
        if let last = stem.last, "pkfθʃsʧ".contains(last) {
            return stem + "t"
        } else if stem.last == "d" {
            return stem + (self.british ? "ɪ" : "ᵻ") + "d"
        } else if stem.last != "t" {
            return stem + "d"
        } else if self.british || stem.count < 2 {
            return stem + "ɪd"
        } else {
            if let secondLast = stem[safe: stem.index(stem.endIndex, offsetBy: -2)], US_TAUS.contains(secondLast) {
                return String(stem.dropLast()) + "ɾᵻd"
            }
            return stem + "ᵻd"
        }
    }

    func stem_ed(word: String, tag: String, stress: Double?, ctx: TokenContext) -> (String?, Int?) {
        var stem: String? = nil
        if word.hasSuffix("d") && !word.hasSuffix("dd") && self.isKnown(word: String(word.dropLast())) {
            stem = String(word.dropLast())
        } else if word.hasSuffix("ed") && !word.hasSuffix("eed") && self.isKnown(word: String(word.dropLast(2))) {
            stem = String(word.dropLast(2))
        } else {
            return (nil, nil)
        }
        let lookupResult = self.lookup(word: stem!, tag: tag, stress: stress, ctx: ctx)
        if let lookupStem = lookupResult.0 {
            return (self._ed(lookupStem), lookupResult.1)
        }
        return (nil, nil)
    }

    func _ing(_ stem: String) -> String? {
        guard !stem.isEmpty else { return nil }
        if self.british {
            if let last = stem.last, "əː".contains(last) { return nil }
        } else if stem.count > 1,
                  let last = stem.last, last == "t",
                  let secondLast = stem[safe: stem.index(stem.endIndex, offsetBy: -2)],
                  US_TAUS.contains(secondLast) {
            return String(stem.dropLast()) + "ɾɪŋ"
        }
        return stem + "ɪŋ"
    }

    func stem_ing(word: String, tag: String, stress: Double?, ctx: TokenContext) -> (String?, Int?) {
        var stem: String? = nil
        if word.hasSuffix("ing") && self.isKnown(word: String(word.dropLast(3))) {
            stem = String(word.dropLast(3))
        } else if word.hasSuffix("ing") && self.isKnown(word: String(word.dropLast(3)) + "e") {
            stem = String(word.dropLast(3)) + "e"
        } else if word.range(of: #"([bcdgklmnprstvxz])\1ing$|cking$"#, options: .regularExpression) != nil && self.isKnown(word: String(word.dropLast(4))) {
            stem = String(word.dropLast(4))
        } else {
            return (nil, nil)
        }
        let lookupResult = self.lookup(word: stem!, tag: tag, stress: stress, ctx: ctx)
        if let lookupStem = lookupResult.0 {
            return (self._ing(lookupStem), lookupResult.1)
        }
        return (nil, nil)
    }

    func get_word(word: String, tag: String, stress: Double?, ctx: TokenContext) -> (String?, Int?) {
        let specialCase = self.getSpecialCase(word: word, tag: tag, stress: stress, ctx: ctx)
        if specialCase.0 != nil {
            return specialCase
        } else if self.isKnown(word: word) {
            return self.lookup(word: word, tag: tag, stress: stress, ctx: ctx)
        } else if word.hasSuffix("s'") && self.isKnown(word: String(word.dropLast(2)) + "'s") {
            return self.lookup(word: String(word.dropLast(2)) + "'s", tag: tag, stress: stress, ctx: ctx)
        } else if word.hasSuffix("'") && self.isKnown(word: String(word.dropLast())) {
            return self.lookup(word: String(word.dropLast()), tag: tag, stress: stress, ctx: ctx)
        }
        let stemS = self.stem_s(word: word, tag: tag, stress: stress, ctx: ctx)
        if let s = stemS.0 { return (s, stemS.1) }
        let stemEd = self.stem_ed(word: word, tag: tag, stress: stress, ctx: ctx)
        if let ed = stemEd.0 { return (ed, stemEd.1) }
        let effectiveStress = stress ?? 0.5
        let stemIng = self.stem_ing(word: word, tag: tag, stress: effectiveStress, ctx: ctx)
        if let ing = stemIng.0 { return (ing, stemIng.1) }
        return (nil, nil)
    }

    static func isCurrency(_ word: String) -> Bool {
        if !word.contains(".") { return true }
        else if word.filter({ $0 == "." }).count > 1 { return false }
        let parts = word.split(separator: ".")
        if parts.count > 1 {
            let cents = parts[1]
            return cents.count < 3 || cents.allSatisfy({ $0 == "0" })
        }
        return true
    }

    func get_number(word: String, currency: String?, isHead: Bool, numFlags: String) -> (String?, Int?) {
        let regexSuffix = try! NSRegularExpression(pattern: #"([a-z']+)$"#, options: [])
        let nsWord = word as NSString
        let range = NSRange(location: 0, length: nsWord.length)
        let match = regexSuffix.firstMatch(in: word, options: [], range: range)
        var suffix: String? = nil
        if let m = match {
            suffix = nsWord.substring(with: m.range(at: 1))
        }
        var baseWord = word
        if let suf = suffix { baseWord = String(word.dropLast(suf.count)) }
        var result: [(String, Int)] = []
        var mutableWord = baseWord
        if mutableWord.hasPrefix("-") {
            if let lookupMinus = self.lookup(word: "minus", tag: nil, stress: nil, ctx: nil).0 {
                result.append((lookupMinus, 4))
            }
            mutableWord = String(mutableWord.dropFirst())
        }
        func extend_num(_ num: String, first: Bool = true, escape: Bool = false) {
            let numStr: String = escape ? num : (Int(num).flatMap { num2words(Double($0)) } ?? num)
            let splits = numStr.lowercased().split(whereSeparator: { !"abcdefghijklmnopqrstuvwxyz".contains($0) }).map { String($0) }
            for (i, w) in splits.enumerated() {
                if w != "and" || numFlags.contains("&") {
                    if first && i == 0 && splits.count > 1 && w == "one" && numFlags.contains("a") {
                        result.append(("ə", 4))
                    } else {
                        let stressVal: Double? = (w == "point") ? -2 : nil
                        if let lookupRes = self.lookup(word: w, tag: nil, stress: stressVal, ctx: nil).0 {
                            result.append((lookupRes, 4))
                        }
                    }
                } else if w == "and" && numFlags.contains("n") && !result.isEmpty {
                    let last = result.removeLast()
                    result.append((last.0 + "ən", last.1))
                }
            }
        }

        if !isHead && !mutableWord.contains(".") {
            let num = mutableWord.replacingOccurrences(of: ",", with: "")
            if num.first == "0" || num.count > 3 {
                for n in num { extend_num(String(n), first: false) }
            } else if num.count == 3 && !num.hasSuffix("00") {
                extend_num(String(num.first!), first: true)
                if num[num.index(num.startIndex, offsetBy: 1)] == "0" {
                    if let lookupO = self.lookup(word: "O", tag: nil, stress: -2, ctx: nil).0 {
                        result.append((lookupO, 4))
                    }
                    extend_num(String(num.last!), first: false)
                } else {
                    let startIndex = num.index(num.startIndex, offsetBy: 1)
                    extend_num(String(num[startIndex...]), first: false)
                }
            } else {
                extend_num(num)
            }
        } else if mutableWord.filter({ $0 == "." }).count > 1 || !isHead {
            var firstFlag = true
            let parts = mutableWord.replacingOccurrences(of: ",", with: "").split(separator: ".")
            for part in parts {
                if part.isEmpty {
                    // skip empty
                } else if part.first == "0" || (part.count != 2 && part.dropFirst().contains(where: { $0 != "0" })) {
                    for ch in part { extend_num(String(ch), first: false) }
                } else {
                    extend_num(String(part), first: firstFlag)
                }
                firstFlag = false
            }
        } else if let curr = currency, CURRENCIES.keys.contains(curr), Lexicon.isCurrency(mutableWord) {
            let cleaned = mutableWord.replacingOccurrences(of: ",", with: "")
            let parts = cleaned.split(separator: ".").map { String($0) }
            var pairs: [(Int, String)] = []
            let currencyPair = CURRENCIES[curr]!
            for (numStr, unit) in zip(parts, [currencyPair.0, currencyPair.1]) {
                let number = Int(numStr) ?? 0
                pairs.append((number, unit))
            }
            if pairs.count > 1 {
                if pairs[1].0 == 0 {
                    pairs = Array(pairs.prefix(1))
                } else if pairs[0].0 == 0 {
                    pairs = Array(pairs.suffix(from: 1))
                }
            }
            for (i, pair) in pairs.enumerated() {
                if i > 0 {
                    if let lookupAnd = self.lookup(word: "and", tag: nil, stress: nil, ctx: nil).0 {
                        result.append((lookupAnd, 4))
                    }
                }
                extend_num(String(pair.0), first: i == 0)
                if abs(pair.0) != 1 && pair.1 != "pence" {
                    if let stemSRes = self.stem_s(word: pair.1 + "s", tag: nil, stress: nil, ctx: TokenContext()).0 {
                        result.append((stemSRes, 4))
                    }
                } else {
                    if let lookupUnit = self.lookup(word: pair.1, tag: nil, stress: nil, ctx: nil).0 {
                        result.append((lookupUnit, 4))
                    }
                }
            }
        } else {
            var newWord = mutableWord
            if isDigit(newWord) {
                let style = (suffix != nil && ORDINALS.contains(suffix!)) ? "ordinal" : (result.isEmpty && newWord.count == 4 ? "year" : "cardinal")
                if let intVal = Int(newWord) {
                    newWord = num2words(Double(intVal), to: style)
                }
            } else if !newWord.contains(".") {
                newWord = newWord.replacingOccurrences(of: ",", with: "")
                if let intVal = Int(newWord) {
                    let style = (suffix != nil && ORDINALS.contains(suffix!)) ? "ordinal" : "cardinal"
                    newWord = num2words(Double(intVal), to: style)
                }
            } else {
                newWord = newWord.replacingOccurrences(of: ",", with: "")
                if newWord.first == "." {
                    let rest = newWord.dropFirst().compactMap { ch -> String in
                        if let intVal = Int(String(ch)) { return num2words(Double(intVal)) }
                        return String(ch)
                    }.joined(separator: " ")
                    newWord = "point " + rest
                } else {
                    if let doubleVal = Double(newWord) {
                        newWord = num2words(doubleVal)
                    }
                }
            }
            extend_num(newWord, escape: true)
        }
        if result.isEmpty {
            logger.debug("❌ TODO:NUM \(word), \(currency ?? "")")
            return (nil, nil)
        }
        let finalResult = result.map { $0.0 }.joined(separator: " ")
        let minRating = result.map { $0.1 }.min() ?? 4
        if let suf = suffix, ["s", "'s"].contains(suf) {
            if let sRes = self._s(finalResult) {
                return (sRes, minRating)
            }
        } else if let suf = suffix, ["ed", "'d"].contains(suf) {
            if let edRes = self._ed(finalResult) {
                return (edRes, minRating)
            }
        } else if let suf = suffix, suf == "ing" {
            if let ingRes = self._ing(finalResult) {
                return (ingRes, minRating)
            }
        }
        return (finalResult, minRating)
    }

    func append_currency(_ ps: String, currency: String?) -> String {
        guard let curr = currency else { return ps }
        if let currencyPair = CURRENCIES[curr] {
            if let stemResult = self.stem_s(word: currencyPair.0 + "s", tag: nil, stress: nil, ctx: TokenContext()).0 {
                return "\(ps) \(stemResult)"
            }
        }
        return ps
    }

    // Expose global numericIfNeeded and isNumber.
    static func isNumber(_ word: String, isHead: Bool) -> Bool {
        return lexiconIsNumber(word, isHead: isHead)
    }

    // The main entry point – acts like __call__ in Python.
    func callAsFunction(token t: MToken, ctx: TokenContext) -> (String?, Int?) {
        var word = t.alias ?? t.text
        word = word.replacingOccurrences(of: "\u{2018}", with: "'")
                   .replacingOccurrences(of: "\u{2019}", with: "'")
        word = word.precomposedStringWithCompatibilityMapping
        word = word.map { numericIfNeeded($0) }.joined()
        let stress: Double? = (word == word.lowercased()) ? nil : (word == word.uppercased() ? self.capStresses.1 : self.capStresses.0)
        let lookupResult = self.get_word(word: word, tag: t.tag, stress: stress, ctx: ctx)
        if let ps = lookupResult.0 {
            return (applyStress(self.append_currency(ps, currency: t.currency), stress: t.stress), lookupResult.1)
        } else if Lexicon.isNumber(word, isHead: t.isHead) {
            let numResult = self.get_number(word: word, currency: t.currency, isHead: t.isHead, numFlags: t.numFlags)
            return (applyStress(numResult.0 ?? "", stress: t.stress), numResult.1)
        } else if !word.allSatisfy({ c in
            if let o = ordinal(of: c) { return LEXICON_ORDS.contains(o) }
            return false
        }) {
            return (nil, nil)
        }
        if word != word.lowercased() && (word == word.uppercased() || String(word.dropFirst()) == String(word.dropFirst()).lowercased()) {
            let lowerResult = self.get_word(word: word.lowercased(), tag: t.tag, stress: stress, ctx: ctx)
            if let ps = lowerResult.0 {
                return (applyStress(self.append_currency(ps, currency: t.currency), stress: t.stress), lowerResult.1)
            }
        }
        return (nil, nil)
    }
}
