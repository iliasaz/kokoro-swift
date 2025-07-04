//
//  Cmlx.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 3/15/25.
//

import Foundation
import Cmlx
import MLX
import MLXNN

/// Swift MLX extensions not covered in mlx-swift package

public func roll(_ array: MLXArray, shift: Int = 0, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    var result = mlx_array_new()
    mlx_roll_axes(&result, array.ctx, [shift].asInt32, 1, axes.asInt32, axes.count, stream.ctx)
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

public func convTransposed1d_wrapped(
    _ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0,
    dilation: Int = 1, groups: Int = 1,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLX.convTransposed1d(array, weight, stride: stride, padding: padding, dilation: dilation, outputPadding: 0, groups: groups, stream: stream)
}

extension Linear {
    public convenience init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true, zeroInitialized: Bool) {
        if zeroInitialized {
            self.init(weight: .zeros([outputDimensions, inputDimensions]), bias: bias ? .zeros([outputDimensions]) : nil)
        } else {
            self.init(inputDimensions: inputDimensions, outputDimensions: outputDimensions, bias: bias)
        }
    }
}
