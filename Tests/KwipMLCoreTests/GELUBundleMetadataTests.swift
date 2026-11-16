import XCTest
@testable import KwipMLCore

final class GELUBundleMetadataTests: XCTestCase {
    func testPublishedMetadataLinksBothExistingLessonsAndRunnableModeExamplesWithoutChangingVersions() throws {
        let registry = try ComponentRegistry.builtIn()
        let descriptor = try XCTUnwrap(try BuiltInCapabilityBridge.declarations(registry: registry).first { $0.id == "gelu" })
        XCTAssertEqual(descriptor.version, ContractVersion(1,1,0))
        XCTAssertEqual(descriptor.learning.lessonIDs, [ModelConstructionLessonIdentity.geluExact.assessmentID, ModelConstructionLessonIdentity.geluTanh.assessmentID])
        XCTAssertEqual(descriptor.learning.exampleIDs, ["kwip.example.gelu-exact-classifier.v1", "kwip.example.gelu-tanh-classifier.v1"])
        for identity in [ModelConstructionLessonIdentity.geluExact, .geluTanh] {
            let task = try ModelConstructionLessonTasks.publicTask(identity: identity)
            XCTAssertTrue(task.allowedNodeTypes.contains("gelu"))
            let plan = try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 42)
            XCTAssertEqual(plan.assessmentVersion, ContractVersion())
            XCTAssertEqual(plan.runtime.capabilities["gelu"], ContractVersion(), "Reviewed lesson evidence retains its distinct historical 1.0.0 contract.")
        }
        for mode in [false, true] {
            let document = try PreparedWorkflowExample.gelu(approximateTanh: mode).document(device: "cpu")
            let configs = document.nodes.compactMap { node -> [String: JSONValue]? in
                if case .modelPrimitive(let type, _, let config) = node.payload, type == "gelu" { return config }
                return nil
            }
            XCTAssertEqual(configs, [["approximate_tanh": .bool(mode)]])
        }
    }
}
