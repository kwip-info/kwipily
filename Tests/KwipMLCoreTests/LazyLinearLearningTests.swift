import XCTest
@testable import KwipMLCore

final class LazyLinearLearningTests: XCTestCase {
    private func graph(output: Int = 4, bias: Bool = true) -> GraphDocument {
        GraphDocument(name: "My inferred map", nodes: [
            .init(id: "source", type: "input", name: "Measurements", x: 0, y: 0),
            .init(id: "map", type: "lazy_linear", name: "My layer", x: 100, y: 0, config: ["out_features": .integer(Int64(output)), "bias": .bool(bias)]),
            .init(id: "result", type: "output", name: "Results", x: 200, y: 0)
        ], edges: [.init(id: "a", source: "source", target: "map"), .init(id: "b", source: "map", target: "result")])
    }
    func testTaskHasIndependentShapeAndStateCriteriaWithoutPrivateHandout() throws {
        let plan = try LazyLinearAssessment.plan(runtimeVersion: "2.14.1", seed: 42)
        XCTAssertEqual(plan.id, "assessment.lazy-linear-inferred-width")
        XCTAssertEqual(plan.runtime.capabilities["lazy_linear"], ContractVersion())
        XCTAssertEqual(Set(plan.criteria.map(\.id)), ["output-shape", "inferred-width", "weight-shape", "parameter-count", "repeat-state", "reload-state", "rng-preserved"])
        for fixture in plan.fixtures {
            let input = try XCTUnwrap(fixture.inputs["x"])
            XCTAssertEqual(Array(input.shape.dropFirst()), [2, 3])
            XCTAssertEqual(fixture.expected["output-shape"]?.values, [Double(input.shape[0]), 2, 4])
            XCTAssertEqual(fixture.expected["inferred-width"]?.values, [3])
            XCTAssertEqual(fixture.expected["weight-shape"]?.values, [4, 3])
            XCTAssertEqual(fixture.expected["parameter-count"]?.values, [1]) // actual inventory must match independently computed weight + chosen bias count
            for key in ["repeat-state", "reload-state", "rng-preserved"] { XCTAssertEqual(fixture.expected[key]?.values, [1]) }
        }
        let publicData = try JSONEncoder().encode(plan.publicTask(mode: .assessment))
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: publicData) as? [String: Any])
        XCTAssertNil(fields["fixtures"]); XCTAssertNil(fields["fixtureSeed"])
        XCTAssertTrue(TensorLearningTask.operations.contains("lazy-linear-inferred-width"))
    }
    func testSubmissionPreservesBothBiasModesAndRejectsEagerOrDisconnectedGraph() throws {
        for bias in [false, true] {
            let authored = graph(bias: bias)
            let submission = try LazyLinearLearningSubmission(graph: authored)
            XCTAssertEqual(try LazyLinearLearningSubmission.decode(submission.encoded()), submission)
            XCTAssertEqual(submission.graph.nodes.first { $0.type == "lazy_linear" }?.config["bias"], .bool(bias))
        }
        var eager = graph(); eager.nodes[1].type = "linear"
        eager.nodes[1].config["in_features"] = 3
        XCTAssertThrowsError(try LazyLinearLearningSubmission(graph: eager))
        var disconnected = graph(); disconnected.edges.removeLast()
        XCTAssertThrowsError(try LazyLinearLearningSubmission(graph: disconnected))
    }
    func testWrongWidthIsValidConstructionButIndependentShapeCriterionFails() throws {
        let submission = try LazyLinearLearningSubmission(graph: graph(output: 5))
        XCTAssertEqual(submission.graph.nodes[1].config["out_features"], 5)
        let plan = try LazyLinearAssessment.plan(runtimeVersion: "2.14.1", seed: 13)
        var fixtures = plan.fixtures.map { LearningFixtureEvidence(fixtureID: $0.id, tensors: $0.expected) }
        fixtures[0].tensors["output-shape"] = .init(shape: [3], values: [Double(plan.fixtures[0].inputs["x"]!.shape[0]), 2, 5])
        let evidence = LearningEvidence(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion, fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: String(repeating: "a", count: 64), fixtures: fixtures, elapsedMilliseconds: 1, maximumTensorElements: 100)
        XCTAssertEqual(try LearningAssessor.evaluate(plan, evidence: evidence).status, .failed)
    }
}
