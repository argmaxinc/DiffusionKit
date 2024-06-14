//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import XCTest

@testable import DiffusionKitMLX

final class DiffusionKitMLXTests: XCTestCase {
    func testInit() throws {
        let diffusionKitMLX = DiffusionKitMLX()
        XCTAssertNotNil(diffusionKitMLX)
    }
}
