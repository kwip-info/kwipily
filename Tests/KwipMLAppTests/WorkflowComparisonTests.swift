import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowComparisonTests: XCTestCase {
    func testFieldDifferencesPreserveTypesMissingValuesAndArrayOrder() throws {
        let a: JSONValue = .object(["seed": .integer(7), "enabled": .bool(true), "features": .array([.string("a"), .string("b")]), "removed": .null])
        let b: JSONValue = .object(["seed": .integer(8), "enabled": .integer(1), "features": .array([.string("b"), .string("a")])])
        let rows = try WorkflowComparisonFields.differences(a, b)
        XCTAssertEqual(rows.map(\.path), ["/enabled", "/features/0", "/features/1", "/removed", "/seed"])
        XCTAssertEqual(rows.first?.baseline, "true")
        XCTAssertEqual(rows.first?.candidate, "1")
        XCTAssertEqual(rows.first { $0.path == "/removed" }?.baseline, "null")
        XCTAssertEqual(rows.first { $0.path == "/removed" }?.candidate, "Not present")
    }
    func testPathsCannotCollideAndEmptyContainersAreObservable() throws {
        let a: JSONValue = .object(["a/b": .integer(1), "a": .object(["b": .integer(2)]), "empty": .array([])])
        let b: JSONValue = .object(["a/b": .integer(2), "a": .object(["b": .integer(3)]), "empty": .object([:])])
        let rows = try WorkflowComparisonFields.differences(a, b)
        XCTAssertEqual(rows.map(\.path), ["/a/b", "/a~1b", "/empty"])
        XCTAssertEqual(rows.last?.baseline, "[]")
        XCTAssertEqual(rows.last?.candidate, "{}")
        XCTAssertTrue(try WorkflowComparisonFields.differences(a, a).isEmpty)
    }
}
