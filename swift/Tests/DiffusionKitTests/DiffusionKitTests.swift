//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import XCTest

@testable import DiffusionKit

final class DiffusionKitTests: XCTestCase {
    func testInit() throws {
        let diffusionKit = DiffusionKit()
        XCTAssertNotNil(diffusionKit)
    }
}
