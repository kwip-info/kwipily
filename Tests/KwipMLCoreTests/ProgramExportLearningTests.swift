import XCTest
@testable import KwipMLCore

final class ProgramExportLearningTests:XCTestCase {
    private let inputs:[String:LearningTensor]=[
        "sourceRows":.init(shape:[2,2],values:[1,2,-1,3]),
        "parameter.weight":.init(shape:[2,2],values:[2,-1,0.5,3]),
        "parameter.bias":.init(shape:[2],values:[0.25,-0.5])]
    func testIndependentAffineReloadReferenceAndActualBoundaryCriteria()throws {
        for task in ProgramExportLearningTask.allCases {
            let reference=try ProgramExportLessonReference(task:task,inputs:inputs)
            XCTAssertEqual(reference.expected["reloadedOutput"],.init(shape:[2,2],values:[0.25,6,-4.75,8]))
            XCTAssertEqual(reference.expected["constraintChecks"],.init(shape:[4],values:[1,1,1,1]))
            XCTAssertEqual(reference.expected["artifactChecks"],.init(shape:[3],values:[1,1,1]))
            XCTAssertEqual(reference.expected["parentByteDifference"],.init(shape:[1],values:[0]))
        }
    }
    func testTwoTypedActionExamplesRoundTripAndRejectUnknownSubmissionEvidence()throws {
        for task in ProgramExportLearningTask.allCases {
            let example=try ProgramExportLearningExamples.submission(task:task)
            XCTAssertEqual(try ProgramExportLearningSubmission.decode(example.encoded(),task:task),example)
            XCTAssertEqual(example.kind,"programExportLearningAction")
            var value=try XCTUnwrap(JSONSerialization.jsonObject(with:example.encoded()) as? [String:Any]);value["expected"]=["reloadedOutput":[0]]
            XCTAssertThrowsError(try ProgramExportLearningSubmission.decode(JSONSerialization.data(withJSONObject:value),task:task))
            value.removeValue(forKey:"expected");value["taskID"]="foreign"
            XCTAssertThrowsError(try ProgramExportLearningSubmission.decode(JSONSerialization.data(withJSONObject:value),task:task))
        }
    }
    func testPrivatePlansKeepIndependentArithmeticAndRejectAlteredCriteria()throws {
        for task in ProgramExportLearningTask.allCases {
            let plan=try ProgramExportAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:23)
            try ProgramExportAssessment.validate(plan)
            XCTAssertEqual(plan.fixtures.count,4)
            for fixture in plan.fixtures {XCTAssertEqual(fixture.expected,try ProgramExportLessonReference(task:task,inputs:fixture.inputs).expected)}
            var altered=plan;altered.fixtures[0].expected["reloadedOutput"]?.values[0] += 1
            XCTAssertThrowsError(try ProgramExportAssessment.validate(altered))
            altered=plan;altered.criteria.removeLast();XCTAssertThrowsError(try ProgramExportAssessment.validate(altered))
        }
    }
    func testMalformedOrUnalignedParameterFixtureNeverInventsOutput()throws {
        var malformed=inputs;malformed["parameter.weight"] = .init(shape:[1,2],values:[1,2])
        XCTAssertThrowsError(try ProgramExportLessonReference(task:.computation,inputs:malformed))
        malformed=inputs;malformed["sourceRows"]?.values[0] = .infinity
        XCTAssertThrowsError(try ProgramExportLessonReference(task:.sizeRange,inputs:malformed))
    }
}
