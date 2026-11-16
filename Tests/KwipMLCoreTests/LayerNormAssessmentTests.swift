import XCTest
@testable import KwipMLCore

final class LayerNormAssessmentTests: XCTestCase {
    func grade(_ plan: LearningAssessmentPlan, _ values: (LearningFixture) throws -> [String:LearningTensor]) throws -> LearningGradeStatus {
        let measured = try plan.fixtures.map { LearningFixtureEvidence(fixtureID:$0.id,tensors:try values($0)) }
        return try LearningAssessor.evaluate(plan,evidence:.init(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:measured,elapsedMilliseconds:1,maximumTensorElements:100)).status
    }
    func testSeededPlansAndIncompleteDerivativeEvidence() throws {
        for task in LayerNormLearningTask.allCases {
            for seed: UInt64 in [0,17,UInt64.max] {
                let plan = try LayerNormAssessment.plan(task:task,runtimeVersion:"test",seed:seed)
                XCTAssertNoThrow(try plan.validate())
                XCTAssertEqual(plan.id,"assessment."+task.rawValue)
                XCTAssertEqual(plan.criteria.count,task == .affine ? 4 : 2)
                XCTAssertEqual(plan,try LayerNormAssessment.plan(task:task,runtimeVersion:"test",seed:seed))
                let other = try LayerNormAssessment.plan(task:task,runtimeVersion:"test",seed:seed &+ 1)
                XCTAssertNotEqual(plan.fixtures.last?.inputs["x"],other.fixtures.last?.inputs["x"])
                XCTAssertEqual(try grade(plan) {$0.expected},.passed)
                XCTAssertEqual(try grade(plan) {["output":$0.expected["output"]!]},.incomplete)
            }
        }
    }
    func testWrongAxisSampleVarianceAndFrozenStatisticsFail() throws {
        for task in [LayerNormLearningTask.rows,.sample] {
            let plan = try LayerNormAssessment.plan(task:task,runtimeVersion:"test",seed:13)
            XCTAssertEqual(try grade(plan) { fixture in
                let wrong = try LayerNormLessonReference(input:fixture.inputs["x"]!,normalizedShape:task == .rows ? [2,3] : [3],epsilon:task.epsilon)
                return ["output":wrong.output,"inputGradient":try wrong.gradients(for:fixture.inputs["upstream"]!).input]
            },.failed)
            XCTAssertEqual(try grade(plan) { fixture in
                var measured=fixture.expected
                let input=fixture.inputs["x"]!,width=task.normalizedShape.reduce(1,*)
                var output=input.values
                for start in stride(from:0,to:output.count,by:width) {
                    let slice=Array(input.values[start..<start+width]),mean=slice.reduce(0,+)/Double(width)
                    let sampleVariance=slice.reduce(0) {$0+pow($1-mean,2)}/Double(width-1)
                    for i in 0..<width {output[start+i]=(slice[i]-mean)/sqrt(sampleVariance+task.epsilon)}
                }
                measured["output"] = .init(shape:input.shape,values:output)
                return measured
            },.failed)
            XCTAssertEqual(try grade(plan) { fixture in
                var measured=fixture.expected
                let input=fixture.inputs["x"]!,width=task.normalizedShape.reduce(1,*)
                var output=input.values
                for start in stride(from:0,to:output.count,by:width) {
                    let slice=Array(input.values[start..<start+width]),mean=slice.reduce(0,+)/Double(width)
                    let variance=slice.reduce(0) {$0+pow($1-mean,2)}/Double(width)
                    for i in 0..<width {output[start+i]=(slice[i]-mean)/(sqrt(variance)+task.epsilon)}
                }
                measured["output"] = .init(shape:input.shape,values:output)
                return measured
            },.failed,"Epsilon outside the square root changes the operation")
            XCTAssertEqual(try grade(plan) { fixture in
                var measured=fixture.expected
                measured["inputGradient"]=fixture.inputs["upstream"]
                return measured
            },.failed)
        }
    }
    func testAffineParameterGradientsMustSumSharedSlices() throws {
        let task=LayerNormLearningTask.affine
        let plan=try LayerNormAssessment.plan(task:task,runtimeVersion:"test",seed:9)
        XCTAssertEqual(task.scale?.values,[0.5,-1,2])
        XCTAssertEqual(task.bias?.values,[1,-0.5,0.25])
        XCTAssertEqual(try grade(plan) { fixture in
            var measured=fixture.expected
            for key in ["scaleGradient","biasGradient"] {
                var gradient=measured[key]!
                gradient.values=gradient.values.map {$0/Double(fixture.inputs["x"]!.values.count/3)}
                measured[key]=gradient
            }
            return measured
        },.failed)
    }
}
