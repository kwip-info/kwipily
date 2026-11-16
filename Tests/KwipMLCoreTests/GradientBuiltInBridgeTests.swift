import XCTest
@testable import KwipMLCore

final class GradientBuiltInBridgeTests:XCTestCase {
    private let ids:Set<String>=["gradient_policy","trainability"]
    func testPublishedTrainingPoliciesEnterSharedBridgeOnlyAsRecipeState()throws {
        let registry=try ComponentRegistry.builtIn(),values=try BuiltInCapabilityBridge.declarations(registry:registry)
        XCTAssertEqual(values.map(\.id),registry.components.map(\.id))
        for id in ids.sorted() {
            let definition=try XCTUnwrap(registry.component(type:id)),actual=try XCTUnwrap(values.first {$0.id == id})
            XCTAssertEqual(actual,try GradientControlCapabilityBridge.declaration(definition))
            XCTAssertEqual(actual.ports.count,1);XCTAssertEqual(actual.ports[0].id,"out");XCTAssertEqual(actual.ports[0].direction,.output)
            XCTAssertEqual(actual.ports[0].schema.kind,.state)
            XCTAssertEqual(actual.ports[0].schema.state?.key,id == "gradient_policy" ? "gradientPolicyRecipe":"trainabilityRecipe")
            XCTAssertEqual(actual.ports[0].schema.state?.kind,id == "gradient_policy" ? .gradientPolicy:.trainability)
            XCTAssertFalse(actual.lifecycle.phases.contains(.forward));XCTAssertFalse(actual.lifecycle.phases.contains(.backward));XCTAssertFalse(actual.lifecycle.state.contains {$0.kind == .parameters})
            XCTAssertTrue(CapabilityContractValidator.validate(actual).isValid)
        }
        XCTAssertTrue(CapabilityContractValidator.validate(try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.14.1")).isValid)
    }
    func testAddingPublishedPoliciesPreservesAllPriorDeclarationBytesAndOrder()throws {
        let registry=try ComponentRegistry.builtIn();var prior=registry;prior.components.removeAll {ids.contains($0.id)}
        let old=try BuiltInCapabilityBridge.declarations(registry:prior),current=try BuiltInCapabilityBridge.declarations(registry:registry).filter {!ids.contains($0.id)}
        XCTAssertEqual(old,current)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        XCTAssertEqual(try encoder.encode(old),try encoder.encode(current))
    }
    func testForgedPolicyForwardPortsFieldsDuplicateAndUnknownTypesRefuse()throws {
        let registry=try ComponentRegistry.builtIn()
        for id in ids {
            let index=try XCTUnwrap(registry.components.firstIndex {$0.id == id})
            var bad=registry;bad.components[index].inputs=["in"]
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
            bad=registry;bad.components[index].outputs=["out","tensor"]
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
            bad=registry;bad.components[index].fields.append(ComponentField(key:"foreign",title:"Foreign",kind:.boolean,defaultValue:.bool(false)))
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
            bad=registry;bad.components.append(bad.components[index])
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
            bad=registry;bad.components[index].id="unreviewed_forward"
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
        }
    }
}
