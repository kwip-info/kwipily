import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RegistryNumberFieldTests: XCTestCase {
    func testDecimalDraftCommitAndCanonicalConfiguration() {
        let field = ComponentField(key: "eps", title: "Epsilon", kind: .number, defaultValue: .number(1e-5), min: 1e-8, max: 2)
        XCTAssertEqual(RegistryFieldCompatibility.normalized(.integer(1), for: field), .number(1))
        XCTAssertNil(RegistryFieldCompatibility.normalized(.bool(true), for: field))
        var edit = NumericEditState(value: .decimal(1e-5))
        var saved = field.defaultValue
        edit.text = "2.5e-4"
        let apply: (NumericValue) -> Bool = { value in
            guard let canonical = RegistryFieldCompatibility.normalized(.number(value.double), for: field) else { return false }
            saved = canonical; return true
        }
        XCTAssertTrue(edit.commit(domain: .decimal(minimum: field.min, maximum: field.max), origin: "node", currentOrigin: "node", apply: apply))
        XCTAssertEqual(saved, .number(0.00025))
        edit.text = "0"
        XCTAssertFalse(edit.commit(domain: .decimal(minimum: field.min, maximum: field.max), origin: "node", currentOrigin: "node", apply: apply))
        XCTAssertEqual(saved, .number(0.00025))
        edit.text = "0.1"
        XCTAssertFalse(edit.commit(domain: .decimal(minimum: field.min, maximum: field.max), origin: "node", currentOrigin: "other", apply: apply))
        XCTAssertEqual(saved, .number(0.00025))
    }
}
