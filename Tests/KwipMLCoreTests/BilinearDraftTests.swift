import XCTest
@testable import KwipMLCore

final class BilinearDraftTests: XCTestCase {
    static func graph(bias: Bool) -> GraphDocument {
        let roles = ["left", "right", "weight"] + (bias ? ["bias"] : [])
        var nodes = roles.map { GraphNode(id: $0, type: "input", config: ["fixture": .string($0)]) }
        nodes += [.init(id: "interaction", type: "bilinear", config: ["bias": .bool(bias)]), .init(id: "output", type: "output")]
        let ports = ["a", "b", "weight"] + (bias ? ["bias"] : [])
        var edges = zip(roles, ports).map { GraphEdge(source: $0.0, target: "interaction", targetPort: $0.1) }
        edges.append(.init(source: "interaction", target: "output"))
        return GraphDocument(nodes: nodes, edges: edges)
    }
    func testTaskRoutingAndIncompleteDraftRecovery() throws {
        for task in BilinearLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for: task.rawValue), task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation: task.rawValue, runtimeVersion: "test", seed: 7), try BilinearAssessment.plan(task: task, runtimeVersion: "test", seed: 7))
            var draft = TensorLearningDraft(projectID: UUID().uuidString, operation: task.rawValue, mode: .practice)
            XCTAssertNoThrow(try draft.validate())
            draft.graph = Self.graph(bias: task == .withBias)
            let final = draft.graph.edges.removeLast()
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()), draft)
            XCTAssertThrowsError(try draft.submissionData())
            draft.graph.edges.append(final)
            XCTAssertNoThrow(try draft.submissionData())
            XCTAssertThrowsError(try draft.submission())
            draft.pendingFields["unfinished"] = "-"
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields = [:]; draft.parameters = [:]
            XCTAssertThrowsError(try draft.validate())
        }
    }
    func testReceiptReplayBindsTaskSeedAndGradients() throws {
        for task in BilinearLearningTask.allCases {
            var draft = TensorLearningDraft(projectID: UUID().uuidString, operation: task.rawValue, mode: .assessment)
            draft.graph = Self.graph(bias: task == .withBias)
            let bytes = try draft.submissionData()
            let plan = try BilinearAssessment.plan(task: task, runtimeVersion: "test", seed: 7)
            let evidence = LearningEvidence(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion, fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: learningHash(bytes), fixtures: plan.fixtures.map { .init(fixtureID: $0.id, tensors: $0.expected) }, elapsedMilliseconds: 1, maximumTensorElements: 4096)
            let receipt = try TensorLearningReceipt(projectID: draft.projectID, operation: task.rawValue, mode: .assessment, submissionData: bytes, plan: plan, evidence: evidence)
            XCTAssertEqual(try TensorLearningReceipt.decode(receipt.encoded()).attempt.grade.status, .passed)
            var changed = receipt; changed.fixtureSeed += 1
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID: draft.projectID))
            changed = receipt; changed.evidence.fixtures[0].tensors["weightGradient"]!.values[0] += 1
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID: draft.projectID))
        }
    }
}
