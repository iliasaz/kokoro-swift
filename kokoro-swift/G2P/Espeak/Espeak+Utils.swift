//
//  Espeak+Utils.swift
//  kokoro-swift
//
//  Created by Ilia Sazonov on 4/5/25.
//

import Foundation
import libespeak_ng

extension espeak_ng_STATUS {
  func check() throws {
    guard self == ENS_OK else {
      var stringBuffer = [CChar](repeating: 0, count: 512)
      let str = stringBuffer.withUnsafeMutableBufferPointer { buf in
        espeak_ng_GetStatusCodeMessage(self, buf.baseAddress!, buf.count)
        return String(cString: buf.baseAddress!)
      }
      throw NSError(domain: EspeakErrorDomain, code: Int(self.rawValue), userInfo: [ NSLocalizedFailureReasonErrorKey: str ])
    }
  }
}

