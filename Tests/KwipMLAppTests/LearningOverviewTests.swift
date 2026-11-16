import XCTest
@testable import KwipMLApp

final class LearningOverviewTests: XCTestCase {
    func testPassingGuidedWorkIsPracticeEvidenceNotAssessmentCompletion() {
        XCTAssertEqual(LearningOverviewStatus.classify(passed: true, assessment: false, matchesDraft: true), .practiced)
        XCTAssertEqual(LearningOverviewStatus.classify(passed: true, assessment: true, matchesDraft: true), .verified)
    }
    func testChangedConstructionAndFailedAttemptsNeverCountAsVerified() {
        XCTAssertEqual(LearningOverviewStatus.classify(passed: true, assessment: true, matchesDraft: false), .needsRecheck)
        XCTAssertEqual(LearningOverviewStatus.classify(passed: false, assessment: true, matchesDraft: true), .inProgress)
    }
    func testDiscoveryCombinesWordsAndKeepsFrameworkAndKindBoundaries() {
        let item = LearnCatalogItem(id: "tensor", title: "Tensor layout", category: "Tensors & linear algebra", kind: "Practice", backend: "PyTorch", searchText: "reshape transpose gradients")
        XCTAssertTrue(item.matches(section: "Library", search: "TRANSPOSE gradients"))
        XCTAssertFalse(item.matches(section: "TensorFlow", search: "transpose"))
        XCTAssertFalse(item.matches(section: "Examples", search: ""))
        XCTAssertFalse(item.matches(section: "Library", search: "transpose absent"))
        XCTAssertTrue(item.matches(section: "Learning paths", search: "reshape"))
    }
}
