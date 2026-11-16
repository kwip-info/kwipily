import XCTest
@testable import KwipMLCore

final class FrozenStandardizationLessonTests: XCTestCase {
    func testFrozenStatisticsDefineAffineMapWithoutRefitting() throws {
        let reference = LearningExamples.frozenStandardizationReference
        XCTAssertEqual(try reference.apply([2, 12]), [0, 0])
        XCTAssertEqual(try reference.apply([6, 4]), [2, -2])
        XCTAssertEqual(try reference.apply([202, -388]), [100, -100])
        XCTAssertThrowsError(try reference.apply([1]))
        XCTAssertThrowsError(try reference.apply([.nan, 0]))
        XCTAssertTrue(reference.instructions.contains("μ"))
        XCTAssertTrue(reference.instructions.contains("refit"))
    }
    func testVersionedPlanGradesNewInputsAndRejectsRawCenteredOrRefittedOutputs() throws {
        let identity = ModelConstructionLessonIdentity.frozenStandardization
        let plan = try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "test", fixtureSeed: 42)
        XCTAssertEqual(plan.objectiveID, "data.frozen-standardization-as-affine")
        XCTAssertEqual(plan.assessmentVersion, ContractVersion())
        XCTAssertEqual(plan.fixtures.count, 4)
        XCTAssertNotEqual(plan.fixtureDigest, try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "test", fixtureSeed: 43).fixtureDigest)
        XCTAssertThrowsError(try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "test", fixtureSeed: 42, assessmentVersion: .init(2,0,0)))
        func grade(_ transform: ([Double]) -> [Double]) throws -> LearningGrade {
            let fixtures = plan.fixtures.map { fixture in
                LearningFixtureEvidence(fixtureID: fixture.id, tensors: ["output": .init(shape: [2,2], values: transform(fixture.inputs["x"]!.values))])
            }
            return try LearningAssessor.evaluate(plan, evidence: .init(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion,
                fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: String(repeating: "a", count: 64),
                fixtures: fixtures, elapsedMilliseconds: 1, maximumTensorElements: 16))
        }
        XCTAssertEqual(try grade { values in values.enumerated().map { ($0.element - ($0.offset % 2 == 0 ? 2 : 12)) / ($0.offset % 2 == 0 ? 2 : 4) } }.status, .passed)
        XCTAssertEqual(try grade { $0 }.status, .failed)
        XCTAssertEqual(try grade { values in values.enumerated().map { $0.element - ($0.offset % 2 == 0 ? 2 : 12) } }.status, .failed)
        XCTAssertEqual(try grade { _ in [0,0,0,0] }.status, .failed)
        let task = try ModelConstructionLessonTasks.publicTask(identity: identity)
        XCTAssertEqual(task.allowedNodeTypes, ["input", "linear", "add", "concat", "output"])
    }
}
