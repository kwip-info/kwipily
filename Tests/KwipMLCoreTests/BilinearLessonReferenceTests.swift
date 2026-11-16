import XCTest
@testable import KwipMLCore

final class BilinearLessonReferenceTests: XCTestCase {
    func testHandCalculatedContractionAndEveryGradient() throws {
        let a = LearningTensor(shape: [1,2], values: [1,2])
        let b = LearningTensor(shape: [1,2], values: [3,4])
        let w = LearningTensor(shape: [1,2,2], values: [1,2,3,4])
        let reference = try BilinearLessonReference(left:a,right:b,weight:w,bias:.init(shape:[1],values:[5]))
        XCTAssertEqual(reference.output, .init(shape:[1,1],values:[66]))
        let gradients = try reference.gradients(for: .init(shape:[1,1],values:[1]))
        XCTAssertEqual(gradients.left.values,[11,25]); XCTAssertEqual(gradients.right.values,[7,10])
        XCTAssertEqual(gradients.weight.values,[3,4,6,8]); XCTAssertEqual(gradients.bias?.values,[1])
    }
    func testSharedWeightsAccumulateButInputsDoNotMixObservations() throws {
        let reference = try BilinearLessonReference(left:.init(shape:[2,1,2],values:[1,2,9,8]),
            right:.init(shape:[2,1,2],values:[3,4,7,6]),weight:.init(shape:[1,2,2],values:[1,2,3,4]),bias:nil)
        let gradients = try reference.gradients(for:.init(shape:[2,1,1],values:[1,0]))
        XCTAssertEqual(gradients.left.values,[11,25,0,0]); XCTAssertEqual(gradients.right.values,[7,10,0,0])
        XCTAssertEqual(gradients.weight.values,[3,4,6,8]); XCTAssertNil(gradients.bias)
        XCTAssertThrowsError(try reference.gradients(for:.init(shape:[1,1],values:[1])))
    }
    func testRejectsImplicitBroadcastAndWeightOrBiasMismatch() throws {
        let a = LearningTensor(shape:[2,2],values:[1,2,3,4])
        let b = LearningTensor(shape:[1,2],values:[1,2])
        let w = LearningTensor(shape:[1,2,2],values:[1,2,3,4])
        XCTAssertThrowsError(try BilinearLessonReference(left:a,right:b,weight:w,bias:nil))
        XCTAssertThrowsError(try BilinearLessonReference(left:a,right:a,weight:.init(shape:[1,2,1],values:[1,2]),bias:nil))
        XCTAssertThrowsError(try BilinearLessonReference(left:a,right:a,weight:w,bias:.init(shape:[2],values:[1,2])))
    }
    func testAllParameterAndInputGradientsMatchFiniteDifferences() throws {
        let a = LearningTensor(shape:[2,2],values:[0.5,-1,2,0.25])
        let b = LearningTensor(shape:[2,3],values:[1,-0.5,2,0.25,1,-2])
        let w = LearningTensor(shape:[2,2,3],values:[1,0.5,-1,2,-0.25,1,0.1,0.2,0.3,0.4,0.5,0.6])
        let c = LearningTensor(shape:[2],values:[0.2,-0.3])
        let upstream = LearningTensor(shape:[2,2],values:[1,-2,0.5,3])
        let reference = try BilinearLessonReference(left:a,right:b,weight:w,bias:c)
        let gradient = try reference.gradients(for:upstream)
        let inputs = [a,b,w,c], gradients = [gradient.left,gradient.right,gradient.weight,try XCTUnwrap(gradient.bias)]
        func loss(_ inputs:[LearningTensor]) throws -> Double {
            let output = try BilinearLessonReference(left:inputs[0],right:inputs[1],weight:inputs[2],bias:inputs[3]).output
            return zip(output.values,upstream.values).reduce(0) { $0 + $1.0*$1.1 }
        }
        for role in inputs.indices {
            for index in inputs[role].values.indices {
                var plus=inputs, minus=inputs
                plus[role].values[index] += 1e-5; minus[role].values[index] -= 1e-5
                let numerical = try (loss(plus)-loss(minus))/2e-5
                XCTAssertEqual(gradients[role].values[index],numerical,accuracy:1e-8,"role \(role), index \(index)")
            }
        }
    }

}
