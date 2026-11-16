import XCTest
@testable import KwipMLCore

final class LossBuiltInBridgeTests:XCTestCase {
    func testPublishedObjectivesEnterSharedBridgeAsExactScoringControls()throws {
        let registry=try ComponentRegistry.builtIn()
        let declarations=try BuiltInCapabilityBridge.declarations(registry:registry)
        XCTAssertEqual(declarations.map(\.id),registry.components.map(\.id))
        for id in LossConfiguration.componentIDs.sorted() {
            let actual=try XCTUnwrap(declarations.first {$0.id == id})
            let definition=try XCTUnwrap(registry.component(type:id))
            let factory=try XCTUnwrap(BundledLossComponents.declarations[id])
            XCTAssertEqual(actual,try factory(definition))
            XCTAssertEqual(actual.ports.map(\.id),["out"])
            XCTAssertEqual(actual.ports[0].schema.kind,.state)
            XCTAssertEqual(actual.ports[0].schema.state?.kind,.objective)
            XCTAssertEqual(actual.ports[0].schema.state?.key,"objectiveRecipe")
            XCTAssertEqual(actual.ports[0].phases,[.train,.evaluate])
            XCTAssertFalse(actual.lifecycle.phases.contains(.forward))
            XCTAssertFalse(actual.lifecycle.phases.contains(.backward))
            XCTAssertEqual(actual.lifecycle.state.map(\.kind),[.objective])
            XCTAssertTrue(CapabilityContractValidator.validate(actual).isValid)
        }
    }
    func testLossIntegrationPreservesPriorDeclarationBytesAndOrder()throws {
        let registry=try ComponentRegistry.builtIn()
        var prior=registry;prior.components.removeAll {LossConfiguration.componentIDs.contains($0.id)}
        let unchanged=try BuiltInCapabilityBridge.declarations(registry:prior)
        let current=try BuiltInCapabilityBridge.declarations(registry:registry).filter {!LossConfiguration.componentIDs.contains($0.id)}
        XCTAssertEqual(current,unchanged)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        XCTAssertEqual(try encoder.encode(current),try encoder.encode(unchanged))
    }
    func testSharedBridgeRefusesForgedObjectivePortsControlsDuplicatesAndUnknownOperators()throws {
        let registry=try ComponentRegistry.builtIn()
        let index=try XCTUnwrap(registry.components.firstIndex {$0.id == "loss_mse"})
        var forged=registry;forged.components[index].inputs=["in"]
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:forged))
        forged=registry;forged.components[index].fields[0].defaultValue = .string("sum")
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:forged))
        forged=registry;forged.components.append(forged.components[index])
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:forged))
        forged=registry;forged.components[index].id="loss_unreviewed"
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:forged))
        forged=registry;forged.schemaVersion=1
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:forged))
    }
}
