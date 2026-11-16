import XCTest
@testable import KwipMLCore

final class ComponentChoiceFieldTests: XCTestCase {
    private func field() -> ComponentField {
        .init(key: "padding", title: "Padding", kind: .choice, defaultValue: .string("valid"), min: nil, max: nil,
              choices: [.init(value: "valid", title: "No padding"), .init(value: "same", title: "Preserve length")])
    }
    func testKnownStringsOnlyAndCodablePreservesLabels() throws {
        let value = field()
        XCTAssertTrue(value.isValidDefinition)
        XCTAssertTrue(value.accepts(.string("valid")))
        XCTAssertTrue(value.accepts(.string("same")))
        let invalidValues: [JSONValue] = [.string("unknown"), .string("VALID"), .integer(1), .bool(true), .null, .array([])]
        for invalid in invalidValues {
            XCTAssertFalse(value.accepts(invalid))
        }
        XCTAssertEqual(value.valueWhenEnabled, .string("valid"))
        XCTAssertEqual(try JSONDecoder().decode(ComponentField.self, from: JSONEncoder().encode(value)), value)
        let legacy = Data(#"{"key":"width","title":"Width","defaultValue":2,"min":1,"max":512}"#.utf8)
        let decoded = try JSONDecoder().decode(ComponentField.self, from: legacy)
        XCTAssertNil(decoded.choices)
        XCTAssertTrue(decoded.isValidDefinition)
        let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        XCTAssertNil(encoded["choices"])
    }
    func testMalformedChoiceDefinitionsAndForeignChoicesAreRejected() {
        let changes: [(inout ComponentField) -> Void] = [
            { $0.choices = nil }, { $0.choices = [] },
            { $0.choices = [.init(value: "", title: "Blank value")] },
            { $0.choices = [.init(value: "  ", title: "Whitespace value")] },
            { $0.choices = [.init(value: String(repeating: "x", count: 129), title: "Long value")] },
            { $0.choices = [.init(value: "valid", title: String(repeating: "x", count: 129))] },
            { $0.choices = [.init(value: "valid", title: "  \n")] },
            { $0.choices = [.init(value: "valid", title: "One"), .init(value: "valid", title: "Two")] },
            { $0.choices = (0..<65).map { .init(value: "v\($0)", title: "Choice \($0)") } },
            { $0.defaultValue = .string("missing") }, { $0.defaultValue = 1 },
            { $0.min = 0 }, { $0.max = 3 }, { $0.minItems = 1 }, { $0.maxItems = 2 },
            { $0.allowsNull = true }
        ]
        for change in changes {
            var invalid = field(); change(&invalid)
            XCTAssertFalse(invalid.isValidDefinition, "\(invalid)")
        }
        var boundary = field()
        boundary.choices = (0..<64).map { .init(value: "v\($0)", title: "Choice \($0)") }
        boundary.defaultValue = .string("v0")
        XCTAssertTrue(boundary.isValidDefinition)
        var number = ComponentField(key: "width", title: "Width", defaultValue: 2)
        number.choices = field().choices
        XCTAssertFalse(number.isValidDefinition)
        number.choices = []
        XCTAssertFalse(number.isValidDefinition)
    }
    func testExposedChoiceMayNarrowButNotExpandTargetValues() {
        let target = field()
        var exposed = target
        exposed.choices = [.init(value: "same", title: "Same length")]
        exposed.defaultValue = .string("same")
        XCTAssertTrue(CompositeDefinitionValidation.canExpose(exposed, to: target))
        XCTAssertFalse(CompositeDefinitionValidation.canExpose(target, to: exposed))
        exposed.choices = [.init(value: "outside", title: "Other")]
        exposed.defaultValue = .string("outside")
        XCTAssertFalse(CompositeDefinitionValidation.canExpose(exposed, to: target))
    }
}
