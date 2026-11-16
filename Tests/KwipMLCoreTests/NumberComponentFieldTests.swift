import XCTest
@testable import KwipMLCore

final class NumberComponentFieldTests: XCTestCase {
    func field(_ lower: Double = 1e-8, _ upper: Double = 2) -> ComponentField {
        ComponentField(key: "eps", title: "Epsilon", kind: .number, defaultValue: .number(1e-5), min: lower, max: upper)
    }
    func testFiniteDecimalBoundsAndCoding() throws {
        let field = field()
        XCTAssertTrue(field.isValidDefinition)
        for value: JSONValue in [.number(1e-8), .number(0.125), .integer(2)] { XCTAssertTrue(field.accepts(value)) }
        for value: JSONValue in [.bool(true), .string("0.1"), .number(.nan), .number(.infinity), .number(0), .number(2.01)] { XCTAssertFalse(field.accepts(value)) }
        let restored = try JSONDecoder().decode(ComponentField.self, from: JSONEncoder().encode(field))
        XCTAssertEqual(restored.kind, .number)
        XCTAssertTrue(restored.accepts(.number(0.125)))
        XCTAssertFalse(self.field(.nan).isValidDefinition)
        XCTAssertFalse(self.field(3, 2).isValidDefinition)
        XCTAssertFalse(self.field(0, .infinity).isValidDefinition)
    }
    func testExposedDecimalRangeCannotWidenTarget() {
        XCTAssertTrue(CompositeDefinitionValidation.canExpose(field(1e-6, 1), to: field()))
        XCTAssertFalse(CompositeDefinitionValidation.canExpose(field(), to: field(1e-6, 1)))
        let integer = ComponentField(key: "eps", title: "Epsilon", defaultValue: 1, min: 0, max: 2)
        XCTAssertFalse(CompositeDefinitionValidation.canExpose(integer, to: field()))
    }
}
