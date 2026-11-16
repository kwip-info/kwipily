import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RegistryChoiceFieldTests: XCTestCase {
    func testChoiceNormalizationPreservesMachineValueAndRejectsDisplayLabel() {
        let field = ComponentField(key: "padding", title: "Padding", kind: .choice,
            defaultValue: .string("same"), min: nil, max: nil,
            choices: [.init(value: "same", title: "Preserve length"), .init(value: "valid", title: "No padding")])
        XCTAssertEqual(RegistryFieldCompatibility.normalized(.string("same"), for: field), .string("same"))
        XCTAssertNil(RegistryFieldCompatibility.normalized(.string("Preserve length"), for: field))
        XCTAssertNil(RegistryFieldCompatibility.normalized(.integer(0), for: field))
        XCTAssertNil(RegistryFieldCompatibility.normalized(.null, for: field))
        var invalid = field
        invalid.choices = [.init(value: "same", title: "One"), .init(value: "same", title: "Two")]
        XCTAssertNil(RegistryFieldCompatibility.normalized(.string("same"), for: invalid))
    }
}
