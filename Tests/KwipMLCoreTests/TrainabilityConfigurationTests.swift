import Foundation
import XCTest
@testable import KwipMLCore

final class TrainabilityConfigurationTests: XCTestCase {
    private func owner(_ node: String = "first", parameter: String = "weight", mode: String = "frozen") -> JSONValue {
        .object(["node": .string(node), "parameter": .string(parameter), "mode": .string(mode)])
    }
    private func config(_ owners: [JSONValue]) -> [String: JSONValue] { ["owners": .array(owners)] }
    func testDefaultInheritAndCanonicalOwnerOrderingPreserveModes() throws {
        XCTAssertEqual(try TrainabilityConfiguration().values, config([]))
        let entries = [owner("z", mode: "trainable"), owner("a", parameter: "nested.bias", mode: "inherit"), owner("a")]
        let c = try TrainabilityConfiguration(values: config(entries))
        XCTAssertEqual(c.owners, [entries[1], entries[2], entries[0]])
        XCTAssertEqual(c.values, config(c.owners))
        XCTAssertNotEqual(c, try TrainabilityConfiguration())
        XCTAssertEqual(c, try TrainabilityConfiguration(values: config(Array(entries.reversed()))))
        let unicode = [owner("😀"), owner("\u{E000}")]
        XCTAssertEqual(try TrainabilityConfiguration(values: config(unicode)).owners, Array(unicode.reversed()), "Unicode scalar order, not UTF16 order")
    }
    func testExactOwnerShapeTypesModesAndDistinctLengthQualifiedNames() throws {
        XCTAssertThrowsError(try TrainabilityConfiguration(componentID: "linear"))
        XCTAssertThrowsError(try TrainabilityConfiguration(values: ["owners": .null]))
        XCTAssertThrowsError(try TrainabilityConfiguration(values: ["owners": .array([]), "eval": true]))
        let entries: [JSONValue] = [owner(""), owner(parameter: ""), owner("bad\0id"), owner(parameter: "bad\0name"), owner(mode: "eval"), owner(mode: "Frozen"), .object(["node": .string("first"), "parameter": .string("weight")]), .object(["node": .string("first"), "parameter": .string("weight"), "mode": .bool(true)]), .object(["node": .string("first"), "parameter": .string("weight"), "mode": .string("frozen"), "shape": .array([2, 2])])]
        for entry in entries { XCTAssertThrowsError(try TrainabilityConfiguration(values: config([entry]))) }
        XCTAssertThrowsError(try TrainabilityConfiguration(values: config([owner(), owner(mode: "trainable")])), "same owner cannot be overridden twice")
        _ = try TrainabilityConfiguration(values: config([owner("a:b", parameter: "c"), owner("a", parameter: "b:c")]))
    }
    func testInclusiveOwnerCountPortableUTF8AndNoTruncation() throws {
        let entries = (0..<128).map { owner("n\($0)") }
        XCTAssertEqual(try TrainabilityConfiguration(values: config(entries)).owners.count, 128)
        XCTAssertThrowsError(try TrainabilityConfiguration(values: config(entries + [owner("extra")])))
        XCTAssertThrowsError(try TrainabilityConfiguration(values: config([owner(String(repeating: "x", count: 32_768))])))
        XCTAssertThrowsError(try TrainabilityConfiguration(values: config([owner(String(repeating: "界", count: 11_000))])))
        XCTAssertEqual(TrainabilityConfiguration.maximumOwners, 128)
        XCTAssertEqual(TrainabilityConfiguration.maximumRecipeBytes, 32_768)
    }
    func testCompleteVersionedRecipeRoundTripNeverRepairsMissingWireFields() throws {
        let c = try TrainabilityConfiguration(values: config([owner("second", parameter: "attention.in_proj_weight", mode: "trainable"), owner(mode: "inherit")]))
        let data = try JSONEncoder().encode(c.recipeValue)
        XCTAssertEqual(try TrainabilityConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self, from: data)), c)
        let base: [String: JSONValue] = ["schemaVersion": 1, "configurationVersion": 1, "componentID": .string("trainability"), "config": .object(c.values)]
        for key in base.keys { var bad = base; bad.removeValue(forKey: key); XCTAssertThrowsError(try TrainabilityConfiguration.decodeRecipe(.object(bad))) }
        for key in ["schemaVersion", "configurationVersion"] { for value in [JSONValue.number(1), .bool(true), .integer(2)] { var bad = base; bad[key] = value; XCTAssertThrowsError(try TrainabilityConfiguration.decodeRecipe(.object(bad))) } }
        var incomplete = base; incomplete["config"] = .object([:]); XCTAssertThrowsError(try TrainabilityConfiguration.decodeRecipe(.object(incomplete)))
        var wrong = base; wrong["componentID"] = .string("gradient_policy"); XCTAssertThrowsError(try TrainabilityConfiguration.decodeRecipe(.object(wrong)))
        var extra = base; extra["source"] = .object([:]); XCTAssertThrowsError(try TrainabilityConfiguration.decodeRecipe(.object(extra)))
    }
}
