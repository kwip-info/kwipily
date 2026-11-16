import XCTest
@testable import KwipMLCore

final class TensorMeasuredOutputTests:XCTestCase {
    func testSharedParameterAxesAreNeverInterpretedAsBatch() throws {
        let parameter=LearningTensor(shape:[2,3],values:[1,2,3,4,5,6])
        for kind in [TensorMeasuredOutput.scaleGradient,.biasGradient,.weightGradient] {
            XCTAssertTrue(kind.isParameter)
            XCTAssertEqual(kind.displayTensor(parameter,observation:0),parameter)
            XCTAssertEqual(kind.displayTensor(parameter,observation:1),parameter)
        }
        let batched=LearningTensor(shape:[2,2,3],values:(0..<12).map(Double.init))
        XCTAssertEqual(TensorMeasuredOutput.inputGradient.displayTensor(batched,observation:1)?.values,[6,7,8,9,10,11])
        XCTAssertNil(TensorMeasuredOutput.output.displayTensor(batched,observation:2))
        XCTAssertNil(TensorMeasuredOutput.scaleGradient.displayTensor(.init(shape:[3],values:[1]),observation:0))
    }
    func testChoicesRequireMeasuredAndExpectedValues() {
        let tensor=LearningTensor(shape:[1,3],values:[1,2,3])
        let fixture=TensorFormativeFixture(id:"f",input:tensor,upstream:tensor,measured:["output":tensor,"scaleGradient":tensor,"biasGradient":tensor,"unknown":tensor],expected:["output":tensor,"scaleGradient":tensor])
        XCTAssertEqual(TensorMeasuredOutput.available(in:fixture),[.output,.scaleGradient])
    }
}
