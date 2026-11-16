import XCTest
@testable import KwipMLCore

final class RemainingActivationIntegrationTests:XCTestCase {
    func testNewOperatorsRequireSchemaTwoAndNormalizeDeclaredDefaults()throws {
        let registry=try ComponentRegistry.builtIn()
        for id in RemainingActivationConfiguration.componentIDs {
            let node=GraphNode(id:"activation",type:id,x:0,y:0)
            XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:1,registry:registry))
            XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry),try RemainingActivationConfiguration(componentID:id,values:[:]).values)
        }
    }
    func testPreparedArchitecturesPropagateActualSampleShape()throws {
        let registry=try ComponentRegistry.builtIn()
        for id in RemainingActivationConfiguration.componentIDs {
            let graph=try ExampleFlows.remainingActivationClassifier(componentID:id)
            let validation=GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(validation.isValid,"\(id): \(validation.issues)")
            XCTAssertEqual(validation.outputTensorContracts[.init("activation","out")]?.layout.sample.dimensions,[2,3])
            XCTAssertEqual(validation.outputTensorContracts[.init("output","out")]?.layout.sample.dimensions,[2])
        }
    }
    func testWrongChannelsAndObservationAxisAreRefusedBeforeExecution()throws {
        let registry=try ComponentRegistry.builtIn()
        for (id,values) in [("prelu",["num_parameters":JSONValue.integer(3),"init":.number(0.25)]),("log_softmax",["dim":.integer(-3)])] {
            var graph=try ExampleFlows.remainingActivationClassifier(componentID:id)
            let index=try XCTUnwrap(graph.nodes.firstIndex {$0.id == "activation"})
            graph.nodes[index].config=values
            XCTAssertFalse(GraphValidator.validate(graph,registry:registry).isValid)
        }
    }
    func testRuntimeDiscoveryRequiresExactPerOperatorDescriptor()throws {
        XCTAssertNoThrow(try BuiltInCapabilityBridge.package(registry:ComponentRegistry.builtIn(),runtimeVersion:"2.14.1"))
        XCTAssertTrue(RemainingActivationRuntimeCapabilities().componentIDs.isEmpty)
        for id in RemainingActivationConfiguration.componentIDs {
            let entry=try XCTUnwrap(BundledActivationComponents.entries[id])
            let data=try JSONEncoder().encode(entry.runtimeContract)
            XCTAssertTrue(RemainingActivationRuntimeCapabilities(descriptors:[id:data]).contains(id))
            XCTAssertThrowsError(try RemainingActivationRuntimeCapabilities.verify(Data("{}".utf8),componentID:id))
            XCTAssertFalse(RemainingActivationRuntimeCapabilities(descriptors:[id:Data("{}".utf8)]).contains(id))
        }
    }
}
