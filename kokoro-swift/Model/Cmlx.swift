//
//  Cmlx.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//

import Foundation
import Cmlx
import MLX

/// Swift MLX extensions not covered in mlx-swift package

public func roll(_ array: MLXArray, shift: Int = 0, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_roll(&result, array.ctx, shift.int32, axes.asInt32, axes.count, stream.ctx)
    return MLXArray(result)
}

extension Array where Element == Int {

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }
}

extension Int {

    /// Convenience to convert `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var int32: Int32 { Int32(self) }
}

