import XCTest
@testable import KwipMLCore

final class BuiltInCapabilityBridgeTests:XCTestCase {
    func testSoftmaxLearningReferencesResolveBothMeasuredTasks() throws {
        let registry=try ComponentRegistry.builtIn()
        let capability=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:registry).first {$0.id == "softmax"})
        let operations=TensorLearningTask.operations.filter {TensorLearningTask.softmaxAxis(for:$0) != nil}
        let plans=try operations.map {try TensorLearningTask.plan(operation:$0,runtimeVersion:"2.14.1",seed:17)}
        XCTAssertEqual(Set(capability.learning.lessonIDs),Set(plans.map(\.id)))
        XCTAssertEqual(capability.learning.lessonIDs.count,2)
        XCTAssertEqual(capability.availability,.existingPOCAdapter)
    }
    func testVariableRankBoundsAndLegacyTensorCoding()throws {
        let legacy=TensorContract([.symbol("batch"),.fixed(2)])
        let bytes=try JSONEncoder().encode(legacy)
        XCTAssertNil(try JSONDecoder().decode(TensorContract.self,from:bytes).leadingAxes)
        let registry=try ComponentRegistry.builtIn()
        let declaration=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:registry).first {$0.id == "reshape"})
        XCTAssertEqual(try JSONDecoder().decode(CapabilityDescriptor.self,from:JSONEncoder().encode(declaration)),declaration)
        for range in [TensorLeadingAxes(minimum:-1,maximum:8),.init(minimum:3,maximum:2),.init(minimum:1,maximum:9),.init(minimum:0,maximum:0)] {
            var malformed=declaration
            malformed.ports[0].schema.tensor?.leadingAxes=range
            XCTAssertFalse(CapabilityContractValidator.validate(malformed).isValid)
        }
    }
    func testRankChangingGraphRequiresConcreteShapeSpecialization() throws {
        let registry=try ComponentRegistry.builtIn()
        let package=try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.14.1")
        let reshape=try XCTUnwrap(package.capabilities.first {$0.id == "reshape"})
        XCTAssertEqual(reshape.ports.first?.schema.tensor?.leadingAxes?.maximum,8)
        XCTAssertTrue(BuiltInCapabilityBridge.validate(graph:ExampleFlows.tensorLayoutClassifier,registry:registry,runtimeVersion:"2.14.1").isValid)
        let unresolved=CapabilityContractValidator.connection(from:.init(reshape,instanceID:"a",portID:"out"),to:.init(reshape,instanceID:"b",portID:"in"),phase:.forward)
        XCTAssertFalse(unresolved.isValid)
        XCTAssertTrue(unresolved.issues.contains {$0.code == "shape.unresolved"})
        var wrong=ExampleFlows.tensorLayoutClassifier
        let index=try XCTUnwrap(wrong.nodes.firstIndex(where:{$0.id == "shape"}))
        wrong.nodes[index].config["shape"] = .array([5,-1])
        XCTAssertFalse(BuiltInCapabilityBridge.validate(graph:wrong,registry:registry,runtimeVersion:"2.14.1").isValid)
    }
    func testAllRegisteredPrimitivesHaveConcreteTypedContracts() throws {
        let registry=try ComponentRegistry.builtIn()
        let package=try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.10.0")
        XCTAssertEqual(Set(package.capabilities.map(\.id)),Set(registry.components.map(\.id)))
        XCTAssertTrue(CapabilityContractValidator.validate(package).isValid)
        for capability in package.capabilities {
            let definition=try XCTUnwrap(registry.component(type:capability.id))
            XCTAssertEqual(capability.ports.filter {$0.direction == .input}.map(\.id),definition.inputs)
            XCTAssertEqual(capability.ports.filter {$0.direction == .output}.map(\.id),definition.outputs)
            XCTAssertEqual(capability.fields,definition.fields)
            XCTAssertEqual(capability.availability,.existingPOCAdapter)
            if ["prelu", "dropout"].contains(capability.id) {
                XCTAssertEqual(capability.lifecycle.resume, .weightsOnly)
                XCTAssertTrue(capability.lifecycle.phases.contains(.resume))
                XCTAssertEqual(capability.lifecycle.state.map(\.kind), capability.id == "prelu" ? [.parameters] : [.randomGenerator])
            } else if AttentionConfiguration.componentIDs.contains(capability.id) {
                let utility = ["attention_mask", "rotary_position"].contains(capability.id)
                XCTAssertEqual(capability.lifecycle.resume, utility ? .none : .weightsOnly)
                XCTAssertEqual(capability.lifecycle.phases.contains(.resume), !utility)
                XCTAssertEqual(capability.lifecycle.state.map(\.kind), utility ? [] : capability.id == "scaled_dot_product_attention" ? [.randomGenerator] : [.parameters, .randomGenerator])
                XCTAssertEqual(capability.lifecycle.fitBoundary, utility ? .notApplicable : .trainingPartitionOnly)
            } else if RecurrentConfiguration.componentIDs.contains(capability.id) {
                XCTAssertEqual(capability.lifecycle.resume,.weightsOnly)
                XCTAssertTrue(capability.lifecycle.phases.contains(.resume))
                XCTAssertEqual(capability.lifecycle.state.map(\.kind),capability.id.hasSuffix("_cell") ? [.parameters]:[.parameters,.randomGenerator])
                XCTAssertEqual(capability.lifecycle.fitBoundary,.trainingPartitionOnly)
            } else if LossConfiguration.componentIDs.contains(capability.id) || ["gradient_policy","trainability"].contains(capability.id) {
                XCTAssertEqual(capability.lifecycle.resume,.none)
                XCTAssertEqual(capability.lifecycle.phases,[.initialize,.train,.evaluate,.checkpoint,.resume,.export,.reset])
                XCTAssertEqual(capability.lifecycle.fitBoundary,.notApplicable)
                XCTAssertEqual(capability.lifecycle.state.map(\.kind),LossConfiguration.componentIDs.contains(capability.id) ? [.objective] : capability.id == "gradient_policy" ? [.gradientPolicy]:[.trainability])
            } else if ["dataset_source","data_loader"].contains(capability.id) {
                XCTAssertEqual(capability.lifecycle.resume,.none)
                XCTAssertEqual(capability.lifecycle.phases,[.initialize,.train,.checkpoint,.resume,.export,.reset])
                XCTAssertEqual(capability.lifecycle.state.map(\.kind),[.dataCursor])
                XCTAssertEqual(capability.lifecycle.state.map(\.key),[capability.id == "dataset_source" ? "datasetRecipe":"loaderRecipe"])
                XCTAssertEqual(capability.lifecycle.fitBoundary,.trainingPartitionOnly)
            } else if OptimizerConfiguration.componentIDs.contains(capability.id) {
                XCTAssertEqual(capability.lifecycle.resume, .none)
                XCTAssertTrue(capability.lifecycle.phases.contains(.resume))
                XCTAssertEqual(capability.lifecycle.state.map(\.kind), [.optimizer])
                XCTAssertEqual(capability.lifecycle.fitBoundary, .trainingPartitionOnly)
            } else if SchedulerConfiguration.componentIDs.contains(capability.id) {
                XCTAssertEqual(capability.lifecycle.resume, .none)
                XCTAssertTrue(capability.lifecycle.phases.contains(.resume))
                XCTAssertEqual(capability.lifecycle.state.map(\.kind), [.scheduler])
                XCTAssertEqual(capability.lifecycle.fitBoundary, .trainingPartitionOnly)
            } else {
                XCTAssertEqual(capability.lifecycle.resume, .none)
                XCTAssertFalse(capability.lifecycle.phases.contains(.resume))
                if StatefulNormalizationConfiguration.componentIDs.contains(capability.id) {
                    XCTAssertEqual(capability.lifecycle.state.map(\.kind), capability.id == "group_norm" ? [.parameters] : [.parameters, .buffers])
                }
            }
            XCTAssertEqual(capability.devices,[.cpu,.mps])
        }
        let linear=try XCTUnwrap(package.capabilities.first {$0.id == "linear"})
        XCTAssertEqual(linear.lifecycle.state.map(\.kind),[.parameters])
        XCTAssertEqual(linear.ports[0].schema.tensor?.dimensions.last,.parameter("in_features"))
    }
    func testReferenceGraphValidatedWithoutMigrationAndMismatchRejected() throws {
        let registry=try ComponentRegistry.builtIn()
        let graph=ExampleFlows.parallelClassifier
        let encoded=try JSONEncoder().encode(graph)
        XCTAssertTrue(BuiltInCapabilityBridge.validate(graph:graph,registry:registry,runtimeVersion:"2.10.0").isValid)
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:encoded),graph)
        var wrong=graph;wrong.nodes[6].config["in_features"]=15
        XCTAssertFalse(BuiltInCapabilityBridge.validate(graph:wrong,registry:registry,runtimeVersion:"2.10.0").isValid)
    }
    func testUnknownRegistryOperatorOrUnpinnedRuntimeCannotClaimAdapter() throws {
        var registry=try ComponentRegistry.builtIn()
        registry.components.append(ComponentDefinition(id:"attention",title:"Attention"))
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.10.0"))
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry:try .builtIn(),runtimeVersion:""))
        registry=try .builtIn();registry.components[1].inputs=["wrong"]
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.10.0"))
    }
}
