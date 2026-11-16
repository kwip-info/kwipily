import XCTest
@testable import KwipMLCore

final class SoftmaxLessonReferenceTests:XCTestCase {
    func testStableShiftInvariantSlicesAndBatchIsolation() throws {
        let x=LearningTensor(shape:[2,3],values:[1000,1001,1002,-1000,-999,-998])
        let result=try SoftmaxLessonReference(input:x,axis:-1)
        XCTAssertEqual(result.output.shape,x.shape)
        for i in 0..<3 {XCTAssertEqual(result.output.values[i],result.output.values[i+3],accuracy:1e-14)}
        XCTAssertEqual(result.output.values.prefix(3).reduce(0,+),1,accuracy:1e-14)
        XCTAssertEqual(result.output.values[0],1/(1+exp(1)+exp(2)),accuracy:1e-14)
        XCTAssertThrowsError(try SoftmaxLessonReference(input:x,axis:0))
        XCTAssertThrowsError(try SoftmaxLessonReference(input:x,axis:Int.min))
    }
    func testAxisSpecificGradientMatchesFiniteDifferences() throws {
        let x=LearningTensor(shape:[1,2,3],values:[-1,0,2,3,-2,1])
        let g=LearningTensor(shape:x.shape,values:[1,-2,0.5,2,3,-1])
        for axis in [1,2,-1] {
            let reference=try SoftmaxLessonReference(input:x,axis:axis)
            let gradient=try reference.inputGradient(for:g)
            for i in x.values.indices {
                var plus=x;var minus=x
                plus.values[i]+=1e-5;minus.values[i]-=1e-5
                let yp=try SoftmaxLessonReference(input:plus,axis:axis).output.values
                let ym=try SoftmaxLessonReference(input:minus,axis:axis).output.values
                let numerical=zip(zip(yp,ym),g.values).reduce(0.0) {$0+($1.0.0-$1.0.1)*$1.1/2e-5}
                XCTAssertEqual(gradient.values[i],numerical,accuracy:1e-8)
            }
        }
    }
    func testRejectsMalformedInputsAndUpstream() throws {
        XCTAssertThrowsError(try SoftmaxLessonReference(input:.init(shape:[1,2],values:[.nan,0]),axis:1))
        XCTAssertThrowsError(try SoftmaxLessonReference(input:.init(shape:[1,2],values:[0]),axis:1))
        let reference=try SoftmaxLessonReference(input:.init(shape:[1,2],values:[0,0]),axis:1)
        XCTAssertThrowsError(try reference.inputGradient(for:.init(shape:[2,1],values:[1,1])))
        let gradient=try reference.inputGradient(for:.init(shape:[1,2],values:[1,1]))
        XCTAssertEqual(gradient.values,[0,0])
    }
}
