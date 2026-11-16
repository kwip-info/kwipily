import XCTest
@testable import KwipMLCore

final class InitializationLearningReceiptTests:XCTestCase {
    func testBothTaskReceiptsColdReplayMeasuredEvidenceAndRejectChangedBinding()throws {
        for task in InitializationLearningTask.allCases {
            let project=UUID().uuidString,plan=try InitializationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:617)
            let submission=try InitializationLearningSubmission(graph:InitializationLearningExamples.preparedGraph(task:task),task:task).encoded()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map {LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:1000)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:project,operation:task.rawValue,mode:mode,submissionData:submission,plan:plan,evidence:evidence)
                let cold=try TensorLearningReceipt.decode(receipt.encoded())
                XCTAssertEqual(cold.attempt.grade.status,.passed)
                XCTAssertEqual(try cold.verifiedReplay(expectedProjectID:project),plan)
                XCTAssertThrowsError(try cold.verifiedReplay(expectedProjectID:UUID().uuidString))
                var tampered=cold;tampered.submissionSHA256=String(repeating:"0",count:64)
                XCTAssertThrowsError(try tampered.encoded())
                tampered=cold;tampered.operation=task == .startingValues ? InitializationLearningTask.startingSpread.rawValue:InitializationLearningTask.startingValues.rawValue
                XCTAssertThrowsError(try tampered.encoded())
                tampered=cold;tampered.evidence.fixtures[0].tensors["startingSettings"]!.values[0]=0.5
                XCTAssertThrowsError(try tampered.encoded(),"A saved pass cannot retain a changed measured result.")
            }
        }
    }
}
