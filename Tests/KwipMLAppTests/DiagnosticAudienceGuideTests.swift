import XCTest
@testable import KwipMLApp

final class DiagnosticAudienceGuideTests: XCTestCase {
    func testPresentationCoversExactCurrentRegistriesWithoutClaimingNewRules() {
        let expected = Set(DiagnosticRuleRegistry.builtIn.rules.map(\.id))
            .union(MeasuredDiagnostics.currentRules.map(\.id))
        XCTAssertEqual(DiagnosticAudienceGuide.knownRuleIDs, expected)
        XCTAssertEqual(expected.count, 17)
        for id in expected {
            let guide = DiagnosticAudienceGuide.make(ruleID: id, observedFact: "Actual captured observation")
            XCTAssertEqual(guide.observedFact, "Actual captured observation")
            XCTAssertFalse(guide.isFallback, id)
            XCTAssertFalse(guide.meaning.isEmpty, id)
            XCTAssertFalse(guide.action.isEmpty, id)
            XCTAssertFalse(guide.alternatives.isEmpty, id)
            XCTAssertFalse(guide.limits.isEmpty, id)
            XCTAssertFalse(guide.action.contains("Open Data"), id)
            XCTAssertFalse(guide.action.contains("automatically apply"), id)
        }
    }
    func testUnknownRuleRetainsObservedEvidenceAndUsesSafeFallback() {
        let guide = DiagnosticAudienceGuide.make(ruleID: "future.check.v2", observedFact: "Unknown reader's original observation")
        XCTAssertTrue(guide.isFallback)
        XCTAssertEqual(guide.observedFact, "Unknown reader's original observation")
        XCTAssertTrue(guide.limits.contains("cannot explain"))
        XCTAssertTrue(guide.action.contains("original"))
    }
    func testDataAndShapeGuidanceUseExistingUniversalCanvasActions() {
        let data = guide("error.data.v1")
        XCTAssertTrue(data.action.contains("Build"))
        XCTAssertTrue(data.action.contains("Inspect"))
        let shape = guide("error.shape.v1")
        XCTAssertTrue(shape.action.contains("Configuration"))
        XCTAssertTrue(shape.limits.contains("message"))
    }
    func testSampledSignalsDoNotBecomeWholeLayerOrCausalClaims() {
        XCTAssertTrue(guide("activation.zero-sample.v1").limits.contains("eight"))
        XCTAssertTrue(guide("activation.saturation-sample.v1").limits.contains("whole"))
        XCTAssertTrue(guide("gradient.growth.v1").limits.contains("cause"))
        XCTAssertTrue(guide("gradient.missing.v1").meaning.contains("different from zero"))
        XCTAssertTrue(guide("runtime.memory.v1").limits.contains("measurement"))
    }
    func testProposedRepairsRemainExplicitAndBounded() {
        for id in ["gradient.growth.v1", "gradient.nonfinite.v1"] {
            let value = guide(id)
            XCTAssertTrue(value.action.contains("Propose lower learning rate"))
            XCTAssertTrue(value.limits.contains("does not run training"))
        }
        XCTAssertTrue(guide("trend.generalization-gap.v1").limits.contains("not proof"))
        XCTAssertTrue(guide("generalization.divergence.v1").limits.contains("not proof"))
        XCTAssertTrue(guide("generalization.divergence.v1").limits.contains("twenty"))
        XCTAssertTrue(guide("data.coarse-holdout.v1").limits.contains("independent"))
    }
    func testPresentationDoesNotMutatePinnedRegistryOrAssessmentState() throws {
        let before = DiagnosticRuleRegistry.builtIn
        let ids = MeasuredDiagnostics.currentRules.map(\.id)
        for id in DiagnosticAudienceGuide.knownRuleIDs { _ = guide(id) }
        XCTAssertEqual(DiagnosticRuleRegistry.builtIn, before)
        try DiagnosticRuleRegistry.builtIn.validate()
        XCTAssertEqual(MeasuredDiagnostics.currentRules.map(\.id), ids)
        XCTAssertTrue(DiagnosticAudienceGuide.lessonSearchTerm.contains("diagnostics"))
    }
    private func guide(_ id: String) -> DiagnosticAudienceGuide {
        .make(ruleID: id, observedFact: "Captured observation")
    }
}
