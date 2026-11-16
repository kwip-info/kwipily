import XCTest
@testable import KwipMLCore

final class TensorCompositionIntegrationTests: XCTestCase {
    func testLegacyEmptyConfigurationKeepsDefaultsAndNewMembersRequireSchemaTwo() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in ["concat","add"] {
            for version in [1,2] {
                XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:GraphNode(type:id),schemaVersion:version,registry:registry),try TensorCompositionConfiguration(componentID:id,values:[:]).values)
            }
        }
        for id in ["stack","permute"] {
            XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:GraphNode(type:id,config:try TensorCompositionConfiguration(componentID:id,values:[:]).values),schemaVersion:1,registry:registry))
        }
    }
    func testRealPreparedExamplesInferNondefaultShapesAndDeclareLessons() throws {
        let registry = try ComponentRegistry.builtIn()
        let descriptors = try BuiltInCapabilityBridge.declarations(registry:registry)
        for id in TensorCompositionConfiguration.componentIDs.sorted() {
            let graph = try ExampleFlows.tensorCompositionClassifier(componentID:id)
            let validation = GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(validation.isValid,"\(id): \(validation.issues)")
            XCTAssertEqual(validation.tensorContracts["composition"]?.layout.sample.elementCount,id == "concat" ? 9 : id == "stack" ? 12 : id == "permute" ? 24 : 6)
            XCTAssertEqual(descriptors.first {$0.id == id}?.learning.lessonIDs,BundledTensorCompositionComponents.entries[id]?.learning?.lessonIDs)
            XCTAssertNoThrow(try PreparedWorkflowExample.component(id).document(device:"cpu"))
        }
    }
    func testRuntimeDescriptorsAreIndependentAndRejectDrift() throws {
        let encoder = JSONEncoder()
        for id in TensorCompositionConfiguration.componentIDs {
            let data = try encoder.encode(BundledTensorCompositionComponents.entries[id]!.runtimeContract)
            let support = TensorCompositionRuntimeCapabilities(descriptors:[id:data])
            XCTAssertEqual(support.componentIDs,[id])
            XCTAssertTrue(TensorCompositionRuntimeCapabilities(descriptors:[id:Data("{}".utf8)]).componentIDs.isEmpty)
            XCTAssertThrowsError(try TensorCompositionRuntimeCapabilities.verify(data,componentID:id == "stack" ? "concat" : "stack"))
        }
    }
}
