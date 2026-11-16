import XCTest
@testable import KwipMLCore

final class MatrixDraftTests:XCTestCase {
    func testIncompleteDraftRecoversButCannotSubmitUntilWired()throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:MatrixLearningTask.broadcast.rawValue,mode:.practice)
        XCTAssertNoThrow(try draft.validate())
        draft.graph=MatrixSubmissionTests.graph();draft.graph.edges.removeLast()
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
        XCTAssertThrowsError(try draft.submissionData())
        draft.graph.edges.append(.init(source:"m",target:"o"))
        XCTAssertEqual(try MatrixLearningSubmission.decode(draft.submissionData()).graph,draft.graph)
        draft.pendingFields["unfinished"]="-"
        XCTAssertThrowsError(try draft.submissionData())
    }
    func testMatrixTaskRoutesAndRejectsForeignState()throws {
        for task in MatrixLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:0),try MatrixAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:0))
        }
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:MatrixLearningTask.contraction.rawValue,mode:.guided)
        draft.graph=MatrixSubmissionTests.graph()
        draft.parameters=[:]
        XCTAssertThrowsError(try draft.validate())
        draft.parameters=nil;draft.graph.nodes[0].config["fixture"] = .string("answers")
        XCTAssertThrowsError(try draft.validate())
    }
}
