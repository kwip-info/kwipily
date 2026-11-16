import XCTest
@testable import KwipMLCore

final class TensorTransformAssessmentTests:XCTestCase {
    func testVersionedPlansSeparateValuesAndGradientMisconceptions()throws {
        for operation in ["reshape","transpose","flatten"] {
            let plan=try TensorTransformAssessment.plan(operation:operation,runtimeVersion:"test",seed:42)
            XCTAssertNoThrow(try plan.validate())
            XCTAssertEqual(plan.criteria.count,2)
            XCTAssertNotEqual(plan.fixtureDigest,try TensorTransformAssessment.plan(operation:operation,runtimeVersion:"test",seed:43).fixtureDigest)
            func grade(_ mutate:(inout [String:LearningTensor])->Void)throws->LearningGradeStatus {
                let fixtures=plan.fixtures.map {fixture -> LearningFixtureEvidence in
                    var values=fixture.expected;mutate(&values)
                    return .init(fixtureID:fixture.id,tensors:values)
                }
                return try LearningAssessor.evaluate(plan,evidence:.init(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:fixtures,elapsedMilliseconds:1,maximumTensorElements:12)).status
            }
            XCTAssertEqual(try grade {_ in},.passed)
            XCTAssertEqual(try grade {$0["inputGradient"]!.values=Array(repeating:1,count:$0["inputGradient"]!.values.count)},.failed)
            XCTAssertEqual(try grade {$0["output"]!.values.reverse()},.failed)
            XCTAssertEqual(try grade {$0.removeValue(forKey:"inputGradient")},.incomplete)
        }
        let transpose=try TensorTransformAssessment.plan(operation:"transpose",runtimeVersion:"test",seed:0)
        XCTAssertNotEqual(transpose.fixtures[0].inputs["x"]!.values,transpose.fixtures[0].expected["output"]!.values)
        XCTAssertThrowsError(try TensorTransformAssessment.plan(operation:"unknown",runtimeVersion:"test",seed:0))
    }
}
