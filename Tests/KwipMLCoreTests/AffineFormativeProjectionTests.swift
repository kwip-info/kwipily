import XCTest
@testable import KwipMLCore

final class AffineFormativeProjectionTests: XCTestCase {
    let plan = LearningExamples.affineComposition(runtimeVersion: "2.14.1", seed: 42)
    func evidence() -> LearningEvidence {
        LearningEvidence(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion,
                         fixtureDigest: plan.fixtureDigest, runtime: plan.runtime,
                         submissionDigest: String(repeating: "a", count: 64),
                         fixtures: plan.fixtures.map { LearningFixtureEvidence(fixtureID: $0.id, tensors: $0.expected) },
                         elapsedMilliseconds: 10, maximumTensorElements: 16)
    }
    func testActualFailedValuesAndSignedDifferenceArePreservedInPlanOrder() throws {
        var measured = evidence()
        measured.fixtures[0].tensors["output"]!.values[0] += 2
        measured.fixtures.reverse()
        let projection = try XCTUnwrap(AffineFormativeProjection.make(plan: plan, evidence: measured, recordedMode: .practice))
        XCTAssertEqual(projection.fixtures.count, 4)
        XCTAssertEqual(projection.fixtures[0].id, plan.fixtures[0].id)
        XCTAssertEqual(projection.fixtures[0].input, plan.fixtures[0].inputs["x"])
        XCTAssertEqual(projection.fixtures[0].difference.values, [2,0,0,0])
        XCTAssertEqual(projection.fixtures[0].measured.values[0], projection.fixtures[0].expected.values[0] + 2)
        XCTAssertEqual(projection.submissionDigest, measured.submissionDigest)
        XCTAssertNotNil(AffineFormativeProjection.make(plan: plan, evidence: evidence(), recordedMode: .guided))
    }
    func testProjectionCarriesRecordedCriterionTolerances() throws {
        let projection = try XCTUnwrap(AffineFormativeProjection.make(plan: plan, evidence: evidence(), recordedMode: .guided))
        guard case .tensor(_, let absolute, let relative) = plan.criteria[0].rule else { return XCTFail("Expected affine numerical criterion") }
        XCTAssertEqual(projection.absoluteTolerance, absolute)
        XCTAssertEqual(projection.relativeTolerance, relative)
    }

    func testAssessmentAndUnusableEvidenceNeverExposeAnswers() {
        XCTAssertNil(AffineFormativeProjection.make(plan: plan, evidence: evidence(), recordedMode: .assessment))
        let changes: [(inout LearningEvidence) -> Void] = [
            { $0.cancelled = true }, { $0.backendError = "failure" }, { $0.fixtures.removeLast() },
            { $0.fixtureDigest = "wrong" }, { $0.runtime.version = "other" },
            { $0.fixtures[0].tensors["output"]!.values[0] = .infinity },
            { $0.fixtures[0].tensors["output"]!.shape = [4] },
            { $0.fixtures[0].tensors["extra"] = LearningTensor(shape: [1], values: [0]) },
            { $0.maximumTensorElements = 1 }, { $0.elapsedMilliseconds = 100_000 }
        ]
        for change in changes {
            var value = evidence(); change(&value)
            XCTAssertNil(AffineFormativeProjection.make(plan: plan, evidence: value, recordedMode: .guided))
        }
    }
    func testModifiedPlanCannotSmuggleUnreviewedFixtures() {
        var changed = plan
        changed.fixtures[0].inputs["x"]!.values[0] += 1
        var measured = evidence(); measured.fixtureDigest = changed.fixtureDigest
        XCTAssertNil(AffineFormativeProjection.make(plan: changed, evidence: measured, recordedMode: .practice))
    }
}
