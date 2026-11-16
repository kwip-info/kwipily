import XCTest
@testable import KwipMLCore

final class InitializationCapabilityBridgeTests:XCTestCase {
    private var definition:ComponentDefinition {BundledInitializationComponents.definition("initialization")}
    private func fixedControl()->CapabilityDescriptor {
        CapabilityDescriptor(id:"initialization",version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[PortContract("out",direction:.output,schema:.state(StatePortContract(key:"initializationRecipe",kind:.initialization)),phases:[.initialize,.train])],fields:[],
            lifecycle:LifecycleContract(phases:[.initialize,.train,.checkpoint,.export,.reset],fitBoundary:.notApplicable,state:[StateDeclaration("initializationRecipe",kind:.initialization)]),
            devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.initialization",
            learning:LearningContract(conceptIDs:["training.starting-values"],lessonIDs:BundledInitializationComponents.metadata("initialization")?.learning?.lessonIDs ?? [],exampleIDs:BundledInitializationComponents.metadata("initialization")?.learning?.exampleIDs ?? []),shapeRuleID:"kwip.recipe.initialization.v1")
    }
    func testReviewedDescriptorIsOnlyAControlAndPublicationRemainsGated()throws {
        let value=try InitializationCapabilityBridge.reviewedDescriptor(definition)
        XCTAssertEqual(value,fixedControl())
        XCTAssertFalse(value.lifecycle.phases.contains(.forward));XCTAssertFalse(value.lifecycle.phases.contains(.fit))
        XCTAssertEqual(value.lifecycle.state.map(\.kind),[.initialization])
        if BundledInitializationComponents.metadata("initialization")?.published != true {XCTAssertThrowsError(try InitializationCapabilityBridge.declaration(definition))}
        var forged=definition;forged.inputs=["in"]
        XCTAssertThrowsError(try InitializationCapabilityBridge.reviewedDescriptor(forged))
        forged=definition;forged.outputs=["out","weights"]
        XCTAssertThrowsError(try InitializationCapabilityBridge.reviewedDescriptor(forged))
    }
    func testOnlyExactFixedInitializationControlMayAvoidTrainingPartitionFitBoundary()throws {
        let value=fixedControl()
        XCTAssertTrue(CapabilityContractValidator.validate(value).isValid)
        var tensor=value;tensor.ports[0].schema = .tensor(TensorContract([.fixed(2)]))
        XCTAssertFalse(CapabilityContractValidator.validate(tensor).isValid)
        var state=value;state.lifecycle.state.append(StateDeclaration("learned",kind:.parameters))
        XCTAssertFalse(CapabilityContractValidator.validate(state).isValid)
        var learned=value;learned.ports[0].schema = .state(StatePortContract(key:"parameters",kind:.parameters))
        XCTAssertFalse(CapabilityContractValidator.validate(learned).isValid)
        var fit=value;fit.lifecycle.phases.append(.fit)
        XCTAssertFalse(CapabilityContractValidator.validate(fit).isValid)
        var forward=value;forward.lifecycle.phases.append(.forward)
        XCTAssertFalse(CapabilityContractValidator.validate(forward).isValid)
        var restore=value;restore.lifecycle.phases.append(.resume)
        XCTAssertFalse(CapabilityContractValidator.validate(restore).isValid)
    }
    func testExistingPublishedCatalogDeclarationsRemainValidWhileInitializerIsUnpublished()throws {
        let registry=try ComponentRegistry.builtIn()
        let values=try BuiltInCapabilityBridge.declarations(registry:registry)
        XCTAssertEqual(Set(values.map(\.id)),Set(registry.components.map(\.id)))
        if BundledInitializationComponents.metadata("initialization")?.published != true {XCTAssertFalse(values.contains {$0.id == "initialization"})}
    }
}
