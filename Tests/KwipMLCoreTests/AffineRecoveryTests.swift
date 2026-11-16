import XCTest
@testable import KwipMLCore

final class AffineRecoveryTests: XCTestCase {
    let identity = AffineLessonIdentity.recovery(taskSeed: 42)
    func testTaskStableAcrossChecksAndFixturesObeyProbeDerivedMap() throws {
        let task = try AffineLessonTasks.publicTask(identity: identity)
        XCTAssertEqual(task.probes.map(\.id), ["zero", "basis-0", "basis-1"])
        let a = try AffineLessonTasks.plan(identity: identity, runtimeVersion: "2.14.1", fixtureSeed: 7)
        let b = try AffineLessonTasks.plan(identity: identity, runtimeVersion: "2.14.1", fixtureSeed: 8)
        XCTAssertNotEqual(a.fixtureDigest, b.fixtureDigest)
        XCTAssertEqual(a.task, b.task)
        XCTAssertEqual(a.objectiveID, "matrices.affine-from-probes")
        XCTAssertNotEqual(task.probes, try AffineLessonTasks.publicTask(identity: .recovery(taskSeed: 43)).probes)
        for fixture in a.fixtures {
            let x = try XCTUnwrap(fixture.inputs["x"])
            let output = try XCTUnwrap(fixture.expected["output"])
            for row in 0..<2 { for component in 0..<2 {
                let bias = task.probes[0].output[component]
                let expected = bias + x.values[row*2] * (task.probes[1].output[component]-bias) + x.values[row*2+1] * (task.probes[2].output[component]-bias)
                XCTAssertEqual(output.values[row*2+component], expected, accuracy: 1e-10)
            }}
        }
    }
    func testVersionedDraftAndStrictTaskIdentity() throws {
        let graph = ExampleFlows.parallelClassifier
        let old = AffineLearningDraft(projectID: UUID().uuidString, graph: graph)
        let bytes = try AffineLearningCodec.encodeDraft(old)
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("lessonIdentity"))
        XCTAssertEqual(try AffineLearningCodec.encodeDraft(AffineLearningCodec.decodeDraft(bytes)), bytes)
        var draft = AffineLearningDraft(projectID: UUID().uuidString, graph: graph, lessonIdentity: identity)
        XCTAssertEqual(draft.schemaVersion, 2)
        XCTAssertEqual(try AffineLearningCodec.decodeDraft(AffineLearningCodec.encodeDraft(draft)).lessonIdentity, identity)
        XCTAssertThrowsError(try draft.submission()) // Nonlinear recovery graphs prohibited before missing parameter checks.
        draft.lessonID = "assessment.affine-composition"
        XCTAssertThrowsError(try draft.validate())
        XCTAssertThrowsError(try AffineLessonTasks.publicTask(identity: .init(assessmentID: "unknown")))
        XCTAssertThrowsError(try AffineLessonTasks.plan(identity: identity, runtimeVersion: "2.14.1", fixtureSeed: 1, assessmentVersion: ContractVersion(2)))
    }
    func testReceiptReplaysExactTaskAndRejectsSeedOrVersionMismatch() throws {
        let plan = try AffineLessonTasks.plan(identity: identity, runtimeVersion: "2.14.1", fixtureSeed: 7)
        let evidence = LearningEvidence(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion, fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: String(repeating: "a", count: 64), fixtures: plan.fixtures.map { LearningFixtureEvidence(fixtureID: $0.id, tensors: $0.expected) }, elapsedMilliseconds: 1, maximumTensorElements: 16)
        let grade = try LearningAssessor.evaluate(plan, evidence: evidence)
        let attempt = LearningAttempt(objectiveID: plan.objectiveID, contentVersion: plan.contentVersion, assessmentVersion: plan.assessmentVersion, mode: .practice, grade: grade)
        var receipt = AffineLearningReceipt(attempt: attempt, projectID: UUID().uuidString, submission: AffineLearningSubmission(graph: ExampleFlows.parallelClassifier, parameters: [:]), submissionSHA256: evidence.submissionDigest, runtime: plan.runtime, fixtureSeed: plan.fixtureSeed, fixtureDigest: plan.fixtureDigest, notes: "", evidence: evidence, lessonIdentity: identity)
        XCTAssertEqual(receipt.schemaVersion, 2)
        XCTAssertEqual(try receipt.replayPlan(), plan)
        let decoded = try JSONDecoder().decode(AffineLearningReceipt.self, from: JSONEncoder().encode(receipt))
        XCTAssertEqual(try decoded.replayPlan(), plan)
        receipt.lessonIdentity = .recovery(taskSeed: 43)
        XCTAssertThrowsError(try receipt.replayPlan())
        receipt.lessonIdentity = identity; receipt.schemaVersion = 1
        XCTAssertThrowsError(try receipt.resolvedLessonIdentity())
        receipt.schemaVersion = 2; receipt.attempt.assessmentVersion = ContractVersion(9)
        XCTAssertThrowsError(try receipt.replayPlan())
    }

    func testRecoveryFormativeProjectionBindsTaskSeed() throws {
        let plan = try AffineLessonTasks.plan(identity: identity, runtimeVersion: "2.14.1", fixtureSeed: 7)
        let evidence = LearningEvidence(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion, fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: String(repeating: "a", count: 64), fixtures: plan.fixtures.map { LearningFixtureEvidence(fixtureID: $0.id, tensors: $0.expected) }, elapsedMilliseconds: 1, maximumTensorElements: 16)
        XCTAssertNotNil(AffineFormativeProjection.make(plan: plan, evidence: evidence, recordedMode: .guided, lessonIdentity: identity))
        XCTAssertNil(AffineFormativeProjection.make(plan: plan, evidence: evidence, recordedMode: .assessment, lessonIdentity: identity))
        XCTAssertNil(AffineFormativeProjection.make(plan: plan, evidence: evidence, recordedMode: .guided, lessonIdentity: .recovery(taskSeed: 43)))
    }
}
