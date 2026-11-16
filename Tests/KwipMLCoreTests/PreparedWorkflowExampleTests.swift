import XCTest
@testable import KwipMLCore

final class PreparedWorkflowExampleTests: XCTestCase {
    func testCategoricalExampleSelectsItsOwnDataAndIntegerArchitecture() throws {
        let example = PreparedWorkflowExample.embedding
        XCTAssertEqual(example.observationsResource, "embedding-observations")
        let document = try example.document(device: "cpu")
        let registry = try ComponentRegistry.builtIn()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: registry, dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.settings.device, "cpu")
        XCTAssertEqual(try WorkflowMappingRequirements.inputContract(plan.model.graph, registry: registry).dtype, .int64)
        XCTAssertTrue(plan.model.graph.nodes.contains { $0.type == "embedding" })
    }

    func testConvolutionExampleUsesExistingNumericObservationsWithoutClaimingTemporalData() throws {
        let example = PreparedWorkflowExample.conv1d
        XCTAssertEqual(example.observationsResource, "workflow-observations")
        XCTAssertEqual(example.sourceResources.count, 1)
        XCTAssertEqual(example.sourceResources.first?.address, WorkflowExamples.sourceAddress)
        XCTAssertEqual(example.sourceResources.first?.resource, "workflow-observations")
        let document = try example.document(device: "cpu")
        XCTAssertNil(document.guidedBuild)
        XCTAssertEqual(document.nodes.filter {
            if case .modelPrimitive(type: "conv1d", configurationVersion: _, config: _) = $0.payload { return true }
            return false
        }.count, 1)
    }

    func testExistingExamplesKeepTheirNumericDatasetAndGuidedProgress() throws {
        let examples: [PreparedWorkflowExample] = [.parallel, .tensorLayout, .softmax,
            .layerNorm(normalizeWholeSample: false), .layerNorm(normalizeWholeSample: true),
            .gelu(approximateTanh: false), .gelu(approximateTanh: true), .guided]
        for example in examples {
            XCTAssertEqual(example.observationsResource, "workflow-observations")
            let document = try example.document(device: "cpu")
            if case .guided = example { XCTAssertEqual(document.guidedBuild?.completedSteps, 1) }
            else { XCTAssertNil(document.guidedBuild) }
        }
    }
}
