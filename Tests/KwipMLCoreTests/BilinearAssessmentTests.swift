import XCTest
@testable import KwipMLCore

final class BilinearAssessmentTests: XCTestCase {
    func testSeededPlansGradeBothInputsAndSharedParametersWithExactBiasMode() throws {
        XCTAssertEqual(Set(BilinearLearningTask.allCases.map(\.rawValue)), ["bilinear-with-bias", "bilinear-without-bias"])
        for task in BilinearLearningTask.allCases {
            let plan = try BilinearAssessment.plan(task: task, runtimeVersion: "2.14.1", seed: 17)
            XCTAssertNoThrow(try plan.validate())
            XCTAssertEqual(plan, try BilinearAssessment.plan(task: task, runtimeVersion: "2.14.1", seed: 17))
            XCTAssertNotEqual(plan.fixtures, try BilinearAssessment.plan(task: task, runtimeVersion: "2.14.1", seed: 18).fixtures)
            XCTAssertEqual(plan.runtime.capabilities["bilinear"], ContractVersion())
            let outputs: Set<String> = Set(["output", "leftGradient", "rightGradient", "weightGradient"] + (task == .withBias ? ["biasGradient"] : []))
            XCTAssertEqual(Set(plan.criteria.map(\.id)), outputs)
            XCTAssertEqual(plan.fixtures.count, 4)
            XCTAssertGreaterThan(Set(plan.fixtures.compactMap { $0.inputs["left"]?.shape.count }).count, 1)
            for fixture in plan.fixtures {
                XCTAssertEqual(Set(fixture.expected.keys), outputs)
                XCTAssertEqual(Set(fixture.inputs.keys), Set(["left", "right", "weight", "upstream"] + (task == .withBias ? ["bias"] : [])))
                let left = try XCTUnwrap(fixture.inputs["left"]), right = try XCTUnwrap(fixture.inputs["right"])
                XCTAssertNotEqual(left.shape.last, right.shape.last)
                XCTAssertEqual(Array(left.shape.dropLast()), Array(right.shape.dropLast()))
                let reference = try BilinearLessonReference(left: left, right: right, weight: XCTUnwrap(fixture.inputs["weight"]), bias: fixture.inputs["bias"])
                XCTAssertEqual(fixture.expected["output"], reference.output)
                let gradients = try reference.gradients(for: XCTUnwrap(fixture.inputs["upstream"]))
                XCTAssertEqual(fixture.expected["leftGradient"], gradients.left)
                XCTAssertEqual(fixture.expected["rightGradient"], gradients.right)
                XCTAssertEqual(fixture.expected["weightGradient"], gradients.weight)
                XCTAssertEqual(fixture.expected["biasGradient"], gradients.bias)
            }
        }
    }
}
