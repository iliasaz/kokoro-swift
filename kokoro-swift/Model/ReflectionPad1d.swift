//
//  ReflectionPad1d.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//

import Foundation
import MLX
import MLXNN

/// A 1D reflection padding layer.
///
/// This layer applies reflection-based padding to an input tensor, where the padding
/// is mirrored based on the existing edge values.
///
/// ### **Usage:**
/// - `ReflectionPad1d(padding: (left, right))`
/// - Adds `left` elements of padding to the start and `right` elements to the end of the last dimension.
///
/// ### **Inputs:**
/// - `x`: Input tensor `(batch, channels, length)`.
///
/// ### **Outputs:**
/// - Padded tensor with reflected values.
///
/// ### **Example:**
/// ```swift
/// let pad = ReflectionPad1d(padding: (2, 2))
/// let paddedX = pad(x) // Adds padding of size (2,2) along the last dimension
/// ```

class ReflectionPad1d: Module {
    let padding: (Int, Int)

    init(padding: (Int, Int)) {
        self.padding = padding
        super.init()
    }

    /// Applies reflection padding to the last dimension.
    ///
    /// - Parameter x: Input tensor `(batch, channels, length)`.
    /// - Returns: Padded tensor with reflected values.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((padding.0, padding.1))])
    }
}
