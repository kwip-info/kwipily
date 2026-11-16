import XCTest
@testable import KwipMLCore

final class SpatialPatchDeliveryTests: XCTestCase {
    func testPreparedModelsPropagatePatchGeometryAndCanonicalConfiguration() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in ["unfold","fold"] {
            let graph = try ExampleFlows.spatialPatchClassifier(componentID:id)
            let bytes = try JSONEncoder().encode(graph)
            let restored = try JSONDecoder().decode(GraphDocument.self,from:bytes)
            XCTAssertEqual(restored,graph)
            if let destination = ProcessInfo.processInfo.environment["KWIPML_PATCH_EXAMPLE_CAPTURE"] {
                let directory = URL(fileURLWithPath:destination,isDirectory:true)
                try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                try bytes.write(to:directory.appendingPathComponent(id+".json"),options:.atomic)
            }
            let validation = GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(validation.isValid,"\(id): \(validation.issues)")
            XCTAssertEqual(validation.outputTensorContracts[.init("patch","out")]?.layout.sample.dimensions,id == "unfold" ? [6,8]:[1,4,5])
            let patch = try XCTUnwrap(graph.nodes.first { $0.id == "patch" })
            XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:patch,schemaVersion:2,registry:registry),patch.config)
            XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:patch,schemaVersion:1,registry:registry))
            let descriptor = try BuiltInCapabilityBridge.declarations(registry:registry).first { $0.id == id }
            XCTAssertNotNil(descriptor)
        }
    }
    func testPreparedComponentOpenersKeepModelAndPinnedSourceTogether() throws {
        for id in ["unfold","fold"] {
            let document = try PreparedWorkflowExample.component(id).document(device:"cpu")
            XCTAssertFalse(document.nodes.isEmpty)
            XCTAssertEqual(PreparedWorkflowExample.component(id).observationsResource,"workflow-observations")
        }
    }
}
