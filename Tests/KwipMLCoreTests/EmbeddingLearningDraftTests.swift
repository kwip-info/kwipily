import XCTest
@testable import KwipMLCore

final class EmbeddingLearningDraftTests:XCTestCase {
    func testTaskDispatchAndIncompleteDraftRecovery()throws {
        for task in EmbeddingLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"test",seed:3),try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:3))
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
            draft.graph.nodes=[.init(id:"lookup",type:"embedding",config:try task.configuration.values)]
            draft.pendingFields=["lookup.max_norm":"unfinished"]
            draft.notes="Repeated indices share table rows."
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:]
            draft.graph.nodes.insert(.init(id:"i",type:"input"),at:0)
            draft.graph.nodes.append(.init(id:"o",type:"output"))
            draft.graph.edges=[.init(source:"i",target:"lookup"),.init(source:"lookup",target:"o")]
            XCTAssertEqual(try EmbeddingLearningSubmission.decode(draft.submissionData()).graph,draft.graph)
            XCTAssertThrowsError(try draft.submission())
            draft.parameters=[:]
            XCTAssertThrowsError(try draft.validate(),"Fixture-owned tables must not be replaced by draft parameter maps")
        }
    }
}
