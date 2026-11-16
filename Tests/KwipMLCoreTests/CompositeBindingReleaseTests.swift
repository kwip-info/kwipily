import XCTest
@testable import KwipMLCore

final class CompositeBindingReleaseTests: XCTestCase {
    func testReleaseMaterializesOnlySelectedTargetsAndPreservesEffectiveValues() throws {
        var definition = try SeparableConv2dBlock.definition()
        let i = try XCTUnwrap(definition.parameters.firstIndex { $0.field.key == "channel_multiplier" })
        definition.parameters[i].field.defaultValue = 3
        let before = try CompositeConfigurationResolver.resolve(definition, definitions: [], registry: .builtIn())
        let released = try CompositeBindingEditing.releasingIntegerProduct(at: 0, from: definition, definitions: [], registry: .builtIn())
        XCTAssertNil(released.integerProducts)
        XCTAssertEqual(released.body.nodes[1].config["in_channels"], 6)
        XCTAssertEqual(released.body.nodes[0], definition.body.nodes[0])
        XCTAssertEqual(released.parameters, definition.parameters)
        XCTAssertEqual(try CompositeConfigurationResolver.resolve(released, definitions: [], registry: .builtIn()), before)
        XCTAssertEqual(definition.body.nodes[1].config["in_channels"], 4)
        XCTAssertThrowsError(try CompositeBindingEditing.releasingIntegerProduct(at: 1, from: definition, definitions: [], registry: .builtIn()))
    }
    func testInvalidProductCannotMaterializeAnInventedValue() throws {
        var definition = try SeparableConv2dBlock.definition()
        let i = try XCTUnwrap(definition.parameters.firstIndex { $0.field.key == "channel_multiplier" })
        definition.parameters[i].field.defaultValue = 512
        let snapshot = definition
        XCTAssertThrowsError(try CompositeBindingEditing.releasingIntegerProduct(at: 0, from: definition, definitions: [], registry: .builtIn()))
        XCTAssertEqual(definition, snapshot)
    }
}
