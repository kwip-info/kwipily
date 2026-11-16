import XCTest
@testable import KwipMLCore
final class AttentionLearningRoutingTests: XCTestCase {
    func testSharedFiveTaskIdentityAndIndependentPlans() throws {
        for task in AttentionLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let plan = try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:41)
            XCTAssertEqual(plan,try AttentionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41))
            XCTAssertEqual(plan.fixtures.count,4)
        }
    }
    func testDraftCompletePartialPendingAndRoundtripPreserveMaskGraph() throws {
        for task in AttentionLearningTask.allCases {
            var draft = TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
            draft.graph = AttentionLearningTests.graph(task)
            XCTAssertNoThrow(try draft.validate())
            let data = try draft.submissionData()
            XCTAssertEqual(try AttentionLearningSubmission.decode(data,task:task).graph,try AttentionLearningSubmission(graph:draft.graph,task:task).graph)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submission())
            draft.pendingFields["op.dropout"] = "unfinished"
            XCTAssertNoThrow(try draft.validate()); XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields = [:]; draft.graph.edges.removeLast()
            XCTAssertNoThrow(try draft.validate()); XCTAssertThrowsError(try draft.submissionData())
        }
    }
}
