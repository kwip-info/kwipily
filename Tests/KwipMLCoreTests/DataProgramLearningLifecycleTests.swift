import XCTest
@testable import KwipMLCore

final class DataProgramLearningLifecycleTests:XCTestCase {
    func testTenTaskRegistrationAndIndependentPlanDispatch()throws {
        for task in DataLoadingLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"test",seed:17),try DataLoadingAssessment.plan(task:task,runtimeVersion:"test",seed:17))
        }
        for task in ProgramExportLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"test",seed:17),try ProgramExportAssessment.plan(task:task,runtimeVersion:"test",seed:17))
        }
    }
    func testEightBlankPartialConnectedDraftsReopenAndPendingEditsBlock()throws {
        for task in DataLoadingLearningTask.allCases {
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice)
            XCTAssertNoThrow(try draft.validate())
            XCTAssertThrowsError(try draft.submissionData())
            let full=try DataLoadingExamples.lessonGraph(task:task)
            draft.graph.nodes=[full.nodes[2]]
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            draft.graph.nodes.append(full.nodes[0]);XCTAssertNoThrow(try draft.validate())
            draft.graph=full;draft.pendingFields=["groups.batch_size":"unfinished"]
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:]
            XCTAssertEqual(try DataLoadingLearningSubmission.decode(draft.submissionData(),task:task).graph,full)
            XCTAssertThrowsError(try draft.submission(),"Recipe controls are not an ordinary tensorGraph")
            draft.graph.edges[0].targetPort="data"
            XCTAssertThrowsError(try draft.validate(),"Cannot replace Train data with a recipe")
        }
    }
    func testTwoActionDraftsReopenAndCannotCarryModelOrPrivateParameterState()throws {
        for task in ProgramExportLearningTask.allCases {
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
            XCTAssertEqual(draft.programExportRecipe,try ProgramExportConfiguration(values:task.values).recipeValue)
            XCTAssertNoThrow(try draft.validate())
            let bytes=try draft.submissionData()
            XCTAssertEqual(try ProgramExportLearningSubmission.decode(bytes,task:task).configuration.values,task.values)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            draft.pendingFields=["program.example_batch_size":"unfinished"]
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:];draft.parameters=[:]
            XCTAssertThrowsError(try draft.validate())
            draft.parameters=nil;draft.graph.nodes=[.init(id:"fake",type:"input")]
            XCTAssertThrowsError(try draft.validate())
        }
    }
    func testTenCanonicalReceiptReplayRejectsStaleAndForeignIdentity()throws {
        let operations=DataLoadingLearningTask.allCases.map(\.rawValue)+ProgramExportLearningTask.allCases.map(\.rawValue)
        for operation in operations {
            let plan=try TensorLearningTask.plan(operation:operation,runtimeVersion:"test",seed:19)
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:operation,mode:.practice)
            if let task=DataLoadingLearningTask(rawValue:operation) {draft.graph=try DataLoadingExamples.lessonGraph(task:task)}
            let bytes=try draft.submissionData()
            // Synthetic exact evidence tests local replay only; actual managed
            // numerical execution has separate worker/lifecycle acceptance.
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:1000)
            let receipt=try TensorLearningReceipt(projectID:draft.projectID,operation:operation,mode:.practice,submissionData:bytes,plan:plan,evidence:evidence)
            let cold=try TensorLearningReceipt.decode(receipt.encoded())
            XCTAssertEqual(cold.attempt.grade.status,.passed)
            XCTAssertEqual(try cold.verifiedReplay(expectedProjectID:draft.projectID),plan)
            var stale=cold;stale.fixtureSeed &+= 1
            XCTAssertThrowsError(try stale.verifiedReplay(expectedProjectID:draft.projectID))
            XCTAssertThrowsError(try cold.verifiedReplay(expectedProjectID:UUID().uuidString))
        }
    }
}
