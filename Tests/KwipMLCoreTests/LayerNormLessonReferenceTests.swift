import XCTest
@testable import KwipMLCore

final class LayerNormLessonReferenceTests: XCTestCase {
    func testPopulationVarianceEpsilonAndIndependentSlices() throws {
        let input = LearningTensor(shape: [2,2,3], values: [1,2,3,10,10,10,101,102,103,-2,0,2])
        let reference = try LayerNormLessonReference(input: input, normalizedShape: [3], epsilon: 0.5)
        let expected = 1 / sqrt(2.0/3 + 0.5)
        XCTAssertEqual(reference.output.values[0], -expected, accuracy: 1e-14)
        XCTAssertEqual(reference.output.values[2], expected, accuracy: 1e-14)
        XCTAssertEqual(Array(reference.output.values[3..<6]), [0,0,0])
        XCTAssertEqual(Array(reference.output.values[0..<3]), Array(reference.output.values[6..<9]))
        let whole = try LayerNormLessonReference(input: input, normalizedShape: [2,3], epsilon: 0.5)
        XCTAssertNotEqual(whole.output.values, reference.output.values)
        let shifted = LearningTensor(shape: input.shape, values: input.values.map { $0 + 1e8 })
        let shiftedResult = try LayerNormLessonReference(input: shifted, normalizedShape: [3], epsilon: 0.5)
        for (a,b) in zip(reference.output.values, shiftedResult.output.values) { XCTAssertEqual(a,b,accuracy:1e-8) }
    }
    func testInputScaleAndBiasGradientsAgainstCentralDifferences() throws {
        let x = LearningTensor(shape: [2,2,3], values: [-2,0,3,1,-1,4,2,5,-3,0,1,2])
        let g = LearningTensor(shape: x.shape, values: [1,-2,0.5,2,3,-1,-0.5,4,2,1,-3,0.25])
        for shape in [[3], [2,3]] {
            let count = shape.reduce(1,*)
            let scale = LearningTensor(shape: shape, values: (0..<count).map { 0.5 + Double($0)/3 })
            let bias = LearningTensor(shape: shape, values: (0..<count).map { Double($0)-2 })
            let result = try LayerNormLessonReference(input: x, normalizedShape: shape, epsilon: 0.03, scale: scale, bias: bias)
            let gradient = try result.gradients(for: g)
            func loss(_ x: LearningTensor, _ scale: LearningTensor, _ bias: LearningTensor) throws -> Double {
                let output = try LayerNormLessonReference(input: x, normalizedShape: shape, epsilon: 0.03, scale: scale, bias: bias).output
                return zip(output.values,g.values).reduce(0) { $0 + $1.0 * $1.1 }
            }
            for coordinate in x.values.indices {
                var plus=x;var minus=x;plus.values[coordinate]+=1e-5;minus.values[coordinate]-=1e-5
                XCTAssertEqual(gradient.input.values[coordinate], try (loss(plus,scale,bias)-loss(minus,scale,bias))/2e-5, accuracy:1e-8)
            }
            for coordinate in scale.values.indices {
                var plus=scale;var minus=scale;plus.values[coordinate]+=1e-5;minus.values[coordinate]-=1e-5
                XCTAssertEqual(try XCTUnwrap(gradient.scale).values[coordinate], try (loss(x,plus,bias)-loss(x,minus,bias))/2e-5, accuracy:1e-8)
                plus=bias;minus=bias;plus.values[coordinate]+=1e-5;minus.values[coordinate]-=1e-5
                XCTAssertEqual(try XCTUnwrap(gradient.bias).values[coordinate], try (loss(x,scale,plus)-loss(x,scale,minus))/2e-5, accuracy:1e-8)
            }
        }
    }
    func testConstantInputSingletonAndNoAffineGradients() throws {
        let x = LearningTensor(shape:[2,3],values:[4,4,4,-2,-2,-2])
        let reference = try LayerNormLessonReference(input:x,normalizedShape:[3],epsilon:0.25)
        let gradient = try reference.gradients(for:.init(shape:x.shape,values:[1,2,3,0,0,0]))
        XCTAssertEqual(gradient.input.values,[-2,0,2,0,0,0])
        XCTAssertNil(gradient.scale);XCTAssertNil(gradient.bias)
        let scaleOnly = try LayerNormLessonReference(input:x,normalizedShape:[3],epsilon:0.25,scale:.init(shape:[3],values:[1,2,3]))
        let scaledGradient = try scaleOnly.gradients(for:.init(shape:x.shape,values:[1,2,3,0,0,0]))
        XCTAssertEqual(scaledGradient.scale?.values,[0,0,0])
        XCTAssertNil(scaledGradient.bias)
        for (actual,expected) in zip(scaledGradient.input.values,[-22.0/3,-4.0/3,26.0/3,0,0,0]) { XCTAssertEqual(actual,expected,accuracy:1e-12) }
        let singleton = try LayerNormLessonReference(input:.init(shape:[2,1],values:[5,-7]),normalizedShape:[1],epsilon:1e-5)
        XCTAssertEqual(singleton.output.values,[0,0])
        XCTAssertEqual(try singleton.gradients(for:.init(shape:[2,1],values:[1,2])).input.values,[0,0])
    }
    func testRejectsInvalidShapeParametersAndNonfiniteArithmetic() throws {
        let x = LearningTensor(shape:[1,2,3],values:[1,2,3,4,5,6])
        for shape in [[],[2],[1,2,3],[-1],[0]] {
            XCTAssertThrowsError(try LayerNormLessonReference(input:x,normalizedShape:shape,epsilon:1e-5))
        }
        for epsilon in [0,-1,.nan,.infinity,1e-300,1e300] {
            XCTAssertThrowsError(try LayerNormLessonReference(input:x,normalizedShape:[3],epsilon:epsilon))
        }
        XCTAssertThrowsError(try LayerNormLessonReference(input:x,normalizedShape:[3],epsilon:1e-5,scale:.init(shape:[2],values:[1,1])))
        XCTAssertThrowsError(try LayerNormLessonReference(input:x,normalizedShape:[3],epsilon:1e-5,bias:.init(shape:[3],values:[0,0,0])))
        XCTAssertThrowsError(try LayerNormLessonReference(input:.init(shape:[1,2],values:[1e308,-1e308]),normalizedShape:[2],epsilon:1e-5))
        let reference = try LayerNormLessonReference(input:x,normalizedShape:[3],epsilon:1e-5)
        XCTAssertThrowsError(try reference.gradients(for:.init(shape:[2,3],values:x.values)))
    }
}
