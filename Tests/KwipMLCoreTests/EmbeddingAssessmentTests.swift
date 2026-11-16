import XCTest
@testable import KwipMLCore

final class EmbeddingAssessmentTests:XCTestCase {
    private func grade(_ plan:LearningAssessmentPlan,_ transform:(LearningFixture)->[String:LearningTensor])throws->LearningGradeStatus {
        try LearningAssessor.evaluate(plan,evidence:.init(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:transform($0))},elapsedMilliseconds:1,maximumTensorElements:100)).status
    }
    func testSeededPlansRequireOutputsTableStateAndScatterGradient()throws {
        for task in EmbeddingLearningTask.allCases {
            for seed:UInt64 in [0,17,UInt64.max] {
                let plan=try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:seed)
                XCTAssertNoThrow(try plan.validate())
                XCTAssertEqual(plan,try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:seed))
                XCTAssertNotEqual(plan.fixtureDigest,try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:seed &+ 1).fixtureDigest)
                XCTAssertEqual(try grade(plan){$0.expected},.passed)
                XCTAssertEqual(try grade(plan){["output":$0.expected["output"]!]},.incomplete)
                XCTAssertFalse(plan.fixtures.contains {$0.expected["inputGradient"] != nil})
                let publicJSON=String(data:try JSONEncoder().encode(plan.publicTask(mode:.assessment)),encoding:.utf8)!
                XCTAssertFalse(publicJSON.contains("fixtureSeed"))
                XCTAssertFalse(publicJSON.contains("expected"))
            }
        }
    }
    func testOrdinaryScatterCannotPassPaddingFrequencyPolicy()throws {
        let plan=try EmbeddingAssessment.plan(task:.paddingFrequency,runtimeVersion:"test",seed:7)
        for omittedPolicy in ["padding_idx","scale_grad_by_freq"] {
            var configuration=try EmbeddingLearningTask.paddingFrequency.configuration.values
            configuration[omittedPolicy]=omittedPolicy == "padding_idx" ? .null : .bool(false)
            let wrong=try EmbeddingConfiguration(values:configuration)
            var wrongEvidence:[String:[String:LearningTensor]]=[:]
            for fixture in plan.fixtures {
                let indices=fixture.inputs["indices"]!
                let reference=try EmbeddingLessonReference(indices:indices.values.map {Int64($0)},shape:indices.shape,table:fixture.inputs["table"]!,configuration:wrong)
                var measured=fixture.expected
                measured["tableGradient"]=try reference.tableGradient(for:fixture.inputs["upstream"]!)
                wrongEvidence[fixture.id]=measured
            }
            XCTAssertEqual(try grade(plan){wrongEvidence[$0.id]!},.failed,omittedPolicy)
        }
    }
    func testIncorrectGradientAndUnreportedMutationFail()throws {
        for task in EmbeddingLearningTask.allCases {
            let plan=try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:19)
            XCTAssertEqual(try grade(plan){ fixture in
                var values=fixture.expected
                var gradient=values["tableGradient"]!
                gradient.values=gradient.values.map {$0/2}
                values["tableGradient"]=gradient
                return values
            },.failed)
        }
        let plan=try EmbeddingAssessment.plan(task:.renormalize,runtimeVersion:"test",seed:19)
        XCTAssertEqual(try grade(plan){ fixture in
            var values=fixture.expected;values["tableAfterForward"]=fixture.inputs["table"];return values
        },.failed)
    }
}
