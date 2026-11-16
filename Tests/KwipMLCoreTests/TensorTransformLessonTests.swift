import XCTest
@testable import KwipMLCore

final class TensorTransformLessonTests:XCTestCase {
    func testTransposeReferenceDistinguishesOrderAndRoutesGradients()throws {
        let input=LearningTensor(shape:[2,2,3],values:(0..<12).map(Double.init))
        let reference=try TensorTransformLessonReference(input:input,operation:"transpose",configuration:["dim0":1,"dim1":2])
        XCTAssertEqual(reference.output.shape,[2,3,2])
        XCTAssertEqual(reference.output.values,[0,3,1,4,2,5,6,9,7,10,8,11])
        let gradient=try reference.inputGradient(for:LearningTensor(shape:[2,3,2],values:(10..<22).map(Double.init)))
        XCTAssertEqual(gradient.shape,input.shape)
        XCTAssertEqual(gradient.values,[10,12,14,11,13,15,16,18,20,17,19,21])
        XCTAssertThrowsError(try reference.inputGradient(for:LearningTensor(shape:[2,2,3],values:Array(repeating:1,count:12))))
        let reshape=try TensorTransformLessonReference(input:input,operation:"reshape",configuration:["shape":.array([3,2])])
        XCTAssertEqual(reshape.output.shape,reference.output.shape)
        XCTAssertNotEqual(reshape.output.values,reference.output.values)
        XCTAssertEqual(reshape.output.values,input.values)
    }
    func testFlattenAndInvalidLessonInputs()throws {
        let input=LearningTensor(shape:[1,2,3],values:[1,2,3,4,5,6])
        let reference=try TensorTransformLessonReference(input:input,operation:"flatten",configuration:["start_dim":1,"end_dim":-1])
        XCTAssertEqual(reference.output,LearningTensor(shape:[1,6],values:input.values))
        for invalid in [LearningTensor(shape:[0,2],values:[]),LearningTensor(shape:[1,2],values:[1]),LearningTensor(shape:[1,2],values:[1,.nan])] {
            XCTAssertThrowsError(try TensorTransformLessonReference(input:invalid,operation:"reshape",configuration:["shape":.array([-1])]))
        }
    }
}
