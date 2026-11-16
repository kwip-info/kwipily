import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientLearningResultsPresentationTests: XCTestCase {
    private func projection(_ task: GradientLearningTask, mode: LearningMode, perturb: Bool = false) throws -> GradientFormativeProjection? {
        let plan = try GradientLearningAssessment.plan(task: task, runtimeVersion: "2.14.1", seed: 41)
        let submission = try GradientLearningSubmission(graph: GradientLearningExamples.preparedGraph(task: task), task: task).encoded()
        let fixtures = plan.fixtures.map { fixture -> LearningFixtureEvidence in
            var tensors = fixture.expected
            if perturb, let key = tensors.keys.sorted().first, let tensor = tensors[key] {
                var values = tensor.values; values[0] += 0.125; tensors[key] = .init(shape: tensor.shape, values: values)
            }
            return .init(fixtureID: fixture.id, tensors: tensors)
        }
        let evidence = LearningEvidence(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion,
            fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: WorkflowPhaseArtifacts.hash(submission),
            fixtures: fixtures, elapsedMilliseconds: 10, maximumTensorElements: 1000)
        return GradientFormativeProjection.project(plan: plan, evidence: evidence, currentSubmission: submission, mode: mode, attemptMode: mode)
    }
    func testAllFourPublicLessonsShowFiveFaithfulMeasuredTablesAcrossFourFixtures() throws {
        for task in GradientLearningTask.allCases {
            for mode in [LearningMode.guided, .practice] {
                let projection = try XCTUnwrap(projection(task, mode: mode, perturb: true))
                let shown = try XCTUnwrap(GradientLearningResultsPresentation.project(projection, task: task, mode: mode))
                XCTAssertEqual(shown.fixtures.map(\.id), projection.fixtures.map(\.fixtureID))
                XCTAssertEqual(shown.fixtures.count, 4)
                for (display, source) in zip(shown.fixtures, projection.fixtures) {
                    XCTAssertEqual(display.tables.map(\.key), source.tensors.keys.sorted())
                    XCTAssertEqual(display.tables.count, 5)
                    for table in display.tables {
                        XCTAssertEqual(table.measured, source.tensors[table.key])
                        XCTAssertEqual(table.reference.shape, table.measured.shape)
                        XCTAssertFalse(table.title.isEmpty); XCTAssertFalse(table.meaning.isEmpty)
                    }
                }
            }
        }
    }
    func testPrivateAssessmentNeverExposesTablesEvenWithPublicProjectionSupplied() throws {
        for task in GradientLearningTask.allCases {
            XCTAssertNil(try projection(task, mode: .assessment))
            let publicProjection = try XCTUnwrap(projection(task, mode: .guided))
            XCTAssertNil(GradientLearningResultsPresentation.project(publicProjection, task: task, mode: .assessment))
            XCTAssertNil(GradientLearningResultsPresentation.project(nil, task: task, mode: .practice))
        }
    }
    func testScheduleAndRecordingAreActualCodesWhileDifferenceReferenceIsZero() throws {
        let schedule = try XCTUnwrap(GradientLearningResultsPresentation.project(projection(.combineSmallGroups, mode: .guided), task: .combineSmallGroups, mode: .guided))
        let recorded = try XCTUnwrap(schedule.fixtures.first?.tables.first { $0.key == "windowSchedule" })
        XCTAssertEqual(recorded.reference.values, [1,2,4,1,1,1,1,2,2,2,4,3,2,1,1,4])
        XCTAssertTrue(recorded.meaning.contains("pass")); XCTAssertTrue(recorded.meaning.contains("adjustment"))
        let keep = try XCTUnwrap(GradientLearningResultsPresentation.project(projection(.keepSettings, mode: .practice), task: .keepSettings, mode: .practice))
        let availability = try XCTUnwrap(keep.fixtures.first?.tables.first { $0.key == "recordingAvailability" })
        XCTAssertEqual(availability.reference.values, [1,0,-1]); XCTAssertTrue(availability.meaning.contains("unavailable"))
        for table in keep.fixtures.flatMap(\.tables) where table.key != "recordingAvailability" {
            XCTAssertTrue(table.reference.values.allSatisfy { $0 == 0 }); XCTAssertTrue(table.meaning.contains("measured minus reference"))
        }
        XCTAssertTrue(GradientLearningResultsPresentation.introduction.contains("not the raw derivative"))
    }
}
