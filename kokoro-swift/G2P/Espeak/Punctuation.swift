//
//  Punctuation.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/30/25.
//

import Foundation

// MARK: - Helper Struct for Mark Indices

/// Represents an index for a punctuation mark for later restoration.
struct MarkIndex {
    let index: Int    // The index of the line or utterance.
    let mark: String  // The punctuation mark itself.
    let position: String  // 'B' for beginning, 'E' for end, 'I' for intermediate, 'A' for alone.
}

// MARK: - Punctuator Class

/**
 A class to preserve or remove punctuation during phonemization.

 Backends behave differently with punctuation:
 - Some ignore and remove it silently,
 - Others might raise an error.

 The Punctuator “hides” punctuation from the phonemization backend and restores it afterwards.

 - Parameter marks: The punctuation marks to consider for processing.
   Can be provided as a string or an NSRegularExpression.
   Defaults to the default marks.
 */
class Punctuation {
    private var _marks: String? = nil
    private var _marksRe: NSRegularExpression? = nil

    var marks: String {
        get {
            if let m = _marks {
                return m
            }
            fatalError("Punctuation was initialized from a regex; cannot access marks as a string")
        }
        set {
            // Use unique characters from the provided string (order sorted for consistency)
            _marks = String(Set(newValue).sorted())
            
            // Build a regular expression pattern that catches:
            // zero or more spaces, one or more punctuation marks (escaped), and zero or more spaces.
            let escapedMarks = NSRegularExpression.escapedPattern(for: _marks!)
            // Instead of string interpolation inside a bracket, we build the pattern:
            let finalPattern = "(\\s*[\(escapedMarks)]+\\s*)+"
            do {
                _marksRe = try NSRegularExpression(pattern: finalPattern, options: [])
            } catch {
                fatalError("Invalid regex pattern for punctuation marks: \(finalPattern)")
            }
        }
    }

    /**
     Initializes the punctuator with specified marks.

     - Parameter marks: Either a String or an NSRegularExpression. Defaults to the default marks.
     */
    init(marks: Any = Punctuation.defaultMarks()) {
        if let marksString = marks as? String {
            self.marks = marksString
        } else if let regex = marks as? NSRegularExpression {
            self._marksRe = regex
            self._marks = nil
        } else {
            fatalError("Punctuation marks must be defined as a String or NSRegularExpression")
        }
    }

    /// Returns the default punctuation marks as a String.
    static func defaultMarks() -> String {
        return _DEFAULT_MARKS
    }

    // MARK: - Remove Method

    /**
     Returns the input text with all punctuation replaced by spaces.

     - Parameter text: The input text (String).
     - Returns: The text with punctuation removed.
     */
    func remove(text: String) -> String {
        guard let regex = _marksRe else { return text }
        let range = NSRange(location: 0, length: text.utf16.count)
        let replaced = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: " ")
        return replaced.trimmingCharacters(in: .whitespaces)
    }

    /**
     Returns an array of strings with punctuation removed.

     - Parameter text: An array of strings.
     - Returns: The array with punctuation removed.
     */
    func remove(text: [String]) -> [String] {
        return text.map { self.remove(text: $0) }
    }

    // MARK: - Preserve Method

    /**
     Removes punctuation from the text while preserving its locations for later restoration.

     - Parameter text: The input text as either a String or an array of Strings.
     - Returns: A tuple containing the preserved text (as an array of Strings) and a list of MarkIndex objects.

     For example, "hello, my world!" becomes (["hello", "my world"], [MarkIndex(...), MarkIndex(...)]).
     */
    func preserve(text: Any) -> ([String], [MarkIndex]) {
        let textArr = str2list(text)
        var preservedText: [String] = []
        var preservedMarks: [MarkIndex] = []
        for (num, line) in textArr.enumerated() {
            let (lineParts, marks) = _preserveLine(line: line, index: num)
            preservedText.append(contentsOf: lineParts)
            preservedMarks.append(contentsOf: marks)
        }
        let filtered = preservedText.filter { !$0.isEmpty }
        return (filtered, preservedMarks)
    }

    /// Helper method for preserve(_:) that processes a single line.
    private func _preserveLine(line: String, index num: Int) -> ([String], [MarkIndex]) {
        guard let regex = _marksRe else { return ([line], []) }
        let range = NSRange(location: 0, length: line.utf16.count)
        let matches = regex.matches(in: line, options: [], range: range)

        if matches.isEmpty {
            return ([line], [])
        }

        // If the line is composed only of punctuation marks.
        if matches.count == 1,
           let matchRange = matches.first?.range,
           let matchStr = Range(matchRange, in: line).map({ String(line[$0]) }),
           matchStr == line {
            return ([], [MarkIndex(index: num, mark: line, position: "A")])
        }

        var marks: [MarkIndex] = []
        // Determine position of each punctuation match.
        for (i, match) in matches.enumerated() {
            guard let matchRange = Range(match.range, in: line) else { continue }
            let matchStr = String(line[matchRange])
            var position = "I"
            if i == 0 && line.hasPrefix(matchStr) {
                position = "B"
            } else if i == matches.count - 1 && line.hasSuffix(matchStr) {
                position = "E"
            }
            marks.append(MarkIndex(index: num, mark: matchStr, position: position))
        }

        // Split the line into sublines at the punctuation marks.
        var preservedLine: [String] = []
        var currentLine = line
        for mark in marks {
            let parts = currentLine.components(separatedBy: mark.mark)
            if let firstPart = parts.first {
                preservedLine.append(firstPart)
            }
            currentLine = parts.dropFirst().joined(separator: mark.mark)
        }
        preservedLine.append(currentLine)
        return (preservedLine, marks)
    }

    // MARK: - Restore Method

    /**
     Restores punctuation into the phonemized text.

     - Parameters:
       - text: The phonemized text as either a String or an array of Strings.
       - marks: An array of MarkIndex objects indicating where punctuation was removed.
       - sep: A separator that was used between phonemes (e.g. a space).
       - strip: A Boolean flag indicating whether trailing separators should be removed.
     - Returns: An array of Strings with punctuation restored.

     This is the reverse operation of preserve(_:).
     */
    static func restore(text: Any, marks: [MarkIndex], sep: Separator, strip: Bool) -> [String] {
        var textArr = str2list(text)
        var punctuatedText: [String] = []
        var pos = 0
        var marks = marks  // mutable copy

        while !textArr.isEmpty || !marks.isEmpty {
            if marks.isEmpty {
                for var line in textArr {
                    if !strip && !sep.isEmpty && !line.hasSuffix(sep) {
                        line += sep
                    }
                    punctuatedText.append(line)
                }
                textArr.removeAll()
            } else if textArr.isEmpty {
                let combined = marks.map { $0.mark }.joined().replacingOccurrences(of: " ", with: sep)
                punctuatedText.append(combined)
                marks.removeAll()
            } else {
                let currentMark = marks.first!
                if currentMark.index == pos {
                    // Remove the mark from marks.
                    let markStr = marks.removeFirst().mark.replacingOccurrences(of: " ", with: sep)
                    if !sep.isEmpty, let first = textArr.first, first.hasSuffix(sep) {
                        textArr[0] = String(first.dropLast(sep.count))
                    }
                    if currentMark.position == "B" {
                        textArr[0] = markStr + textArr[0]
                    } else if currentMark.position == "E" {
                        punctuatedText.append(textArr[0] + markStr + ((strip || markStr.hasSuffix(sep)) ? "" : sep))
                        textArr.removeFirst()
                        pos += 1
                    } else if currentMark.position == "A" {
                        punctuatedText.append(markStr + ((strip || markStr.hasSuffix(sep)) ? "" : sep))
                        pos += 1
                    } else { // "I"
                        if textArr.count == 1 {
                            textArr[0] += markStr
                        } else {
                            let firstWord = textArr[0]
                            textArr.removeFirst()
                            textArr[0] = firstWord + markStr + textArr[0]
                        }
                    }
                } else {
                    punctuatedText.append(textArr[0])
                    textArr.removeFirst()
                    pos += 1
                }
            }
        }
        return punctuatedText
    }
}

// MARK: - Helper Function

/**
 Converts an input (String or [String]) into an array of Strings.
 If the input is a String, returns a single-element array.
 */
func str2list(_ input: Any) -> [String] {
    if let s = input as? String {
        return [s]
    } else if let arr = input as? [String] {
        return arr
    }
    return []
}

// MARK: - Default Punctuation Marks Constant

let _DEFAULT_MARKS = ";:,.!?¡¿—…”«»“”(){}[]"
