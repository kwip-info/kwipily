import XCTest

@testable import KwipMLCore

final class AffineReferenceTests: XCTestCase {
  func testOriginalMapOrderAndBiasAgainstIndependentNumericalOracle() throws {
    let reference = LearningExamples.affineReference
    XCTAssertEqual(reference.first.weight.values, [2, -1, 0.5, 3])
    XCTAssertEqual(reference.second.weight.values, [1, 2, -2, 1])
    XCTAssertEqual(reference.first.inputDimension, 2)
    XCTAssertEqual(reference.second.outputDimension, 2)
    XCTAssertEqual(try reference.apply([0, 0]), [-2.75, -3])
    XCTAssertEqual(try reference.apply([1, 0]), [0.25, -6.5])
    XCTAssertEqual(try reference.apply([0, 1]), [2.25, 2])
    XCTAssertEqual(try reference.apply([2, -1]), [-1.75, -15])
    let plan = LearningExamples.affineComposition(runtimeVersion: "test", seed: 734)
    XCTAssertEqual(plan.task, reference.taskText)
    for fixture in plan.fixtures {
      let input = try XCTUnwrap(fixture.inputs["x"])
      let expected = try XCTUnwrap(fixture.expected["output"])
      for row in 0..<2 {
        let actual = try reference.apply(Array(input.values[(row * 2)..<(row * 2 + 2)]))
        for column in 0..<2 {
          XCTAssertEqual(actual[column], expected.values[row * 2 + column], accuracy: 1e-12)
        }
      }
    }
  }
  func testMalformedDimensionsAndNonfiniteValuesRejected() throws {
    XCTAssertThrowsError(
      try AffineReferenceMap(
        name: "bad", weight: LearningTensor(shape: [2, 2], values: [1, 2, 3, 4]),
        bias: LearningTensor(shape: [3], values: [1, 2, 3])))
    XCTAssertThrowsError(
      try AffineReferenceMap(
        name: "bad", weight: LearningTensor(shape: [1, 1], values: [.infinity]),
        bias: LearningTensor(shape: [1], values: [0])))
    let map = try AffineReferenceMap(
      name: "three inputs", weight: LearningTensor(shape: [1, 3], values: [1, 2, 3]),
      bias: LearningTensor(shape: [1], values: [0]))
    XCTAssertThrowsError(
      try AffineCompositionReference(first: LearningExamples.affineReference.first, second: map))
    XCTAssertThrowsError(try LearningExamples.affineReference.apply([1]))
    XCTAssertThrowsError(try LearningExamples.affineReference.apply([.nan, 1]))
  }
}
