import XCTest
@testable import KwipMLCore

final class GradientLearningDispatchTests:XCTestCase {
    func testSharedFourTaskIdentityAndExactRuntimePlans()throws {
        for task in GradientLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:41),try GradientLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41))
        }
    }
    func testProtectedPartialDraftColdRoundtripPendingRefusalAndQualifiedSubmission()throws {
        for task in GradientLearningTask.allCases {
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice)
            XCTAssertEqual(draft.graph,try GradientLearningExamples.defaultGraph(task:task))
            XCTAssertNoThrow(try draft.validate())
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=["gradient.accumulation_steps":"-"]
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            draft.graph=try GradientLearningExamples.preparedGraph(task:task)
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:]
            let raw=try draft.submissionData()
            XCTAssertEqual(try GradientLearningSubmission.decode(raw,task:task).graph,draft.graph)
            var changed=draft;changed.graph.nodes[2].config["lr"] = .number(1)
            XCTAssertThrowsError(try changed.encoded())
            changed=draft;changed.parameters=["first":["weight":.init(shape:[1],values:[0])]]
            XCTAssertThrowsError(try changed.encoded())
        }
    }
    func testCanonicalReceiptsColdRestorePrivateAndMissingCriterionRefusal()throws {
        for task in GradientLearningTask.allCases {
            let project=UUID().uuidString,plan=try GradientLearningAssessment.plan(task:task,runtimeVersion:"fixture-runtime",seed:71)
            let raw=try GradientLearningSubmission(graph:GradientLearningExamples.preparedGraph(task:task),task:task).encoded()
            // Synthetic measured-comparison transport only; actual native worker proof is separate.
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(raw),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:4096)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:project,operation:task.rawValue,mode:mode,submissionData:raw,plan:plan,evidence:evidence)
                let reopened=try TensorLearningReceipt.decode(receipt.encoded())
                XCTAssertEqual(try reopened.verifiedReplay(expectedProjectID:project),plan)
                XCTAssertThrowsError(try reopened.verifiedReplay(expectedProjectID:UUID().uuidString))
                if mode == .assessment {XCTAssertNil(GradientFormativeProjection.project(plan:plan,evidence:reopened.evidence,currentSubmission:raw,mode:mode,attemptMode:mode))}
                var missing=reopened;missing.evidence.fixtures[0].tensors.removeValue(forKey:try XCTUnwrap(plan.criteria.first?.id))
                XCTAssertThrowsError(try missing.encoded())
            }
        }
    }
}
