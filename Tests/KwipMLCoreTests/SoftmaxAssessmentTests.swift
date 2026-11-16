import XCTest
@testable import KwipMLCore

final class SoftmaxAssessmentTests:XCTestCase {
    func testPlansDistinguishAxesAndCoupledDerivativesAcrossSeeds()throws {
        for axis in [1,-1] {
            for seed:UInt64 in [0,1,42,UInt64.max] {
                let plan=try SoftmaxAssessment.plan(axis:axis,runtimeVersion:"test",seed:seed)
                XCTAssertNoThrow(try plan.validate())
                XCTAssertEqual(plan.id,axis == 1 ? "assessment.softmax-first-sample-axis":"assessment.softmax-last-axis")
                XCTAssertEqual(plan.criteria.count,2)
                XCTAssertNotEqual(plan.fixtureDigest,try SoftmaxAssessment.plan(axis:axis,runtimeVersion:"test",seed:seed &+ 1).fixtureDigest)
                func grade(_ tensors:(LearningFixture)throws->[String:LearningTensor])throws->LearningGradeStatus {
                    let fixtures=try plan.fixtures.map {LearningFixtureEvidence(fixtureID:$0.id,tensors:try tensors($0))}
                    return try LearningAssessor.evaluate(plan,evidence:.init(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:fixtures,elapsedMilliseconds:1,maximumTensorElements:100)).status
                }
                XCTAssertEqual(try grade {$0.expected},.passed)
                XCTAssertEqual(try grade { fixture in
                    let wrong=try SoftmaxLessonReference(input:fixture.inputs["x"]!,axis:axis == 1 ? -1:1)
                    return ["output":wrong.output,"inputGradient":try wrong.inputGradient(for:fixture.inputs["upstream"]!)]
                },.failed)
                XCTAssertEqual(try grade { fixture in
                    var values=fixture.expected
                    let p=values["output"]!,g=fixture.inputs["upstream"]!
                    values["inputGradient"]=LearningTensor(shape:p.shape,values:zip(p.values,g.values).map {$0*(1-$0)*$1})
                    return values
                },.failed,"Elementwise derivative omits cross-coordinate interactions")
                XCTAssertEqual(try grade { ["output":$0.expected["output"]!] },.incomplete)
            }
        }
        XCTAssertThrowsError(try SoftmaxAssessment.plan(axis:0,runtimeVersion:"test",seed:0))
    }
}
