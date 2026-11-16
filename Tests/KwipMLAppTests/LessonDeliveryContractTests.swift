import XCTest
@testable import KwipMLApp

@MainActor final class LessonDeliveryContractTests: XCTestCase {
    func testGuideIsReferenceContentRatherThanAFourthDiscoveryType() {
        XCTAssertFalse(LearningLibraryPresentation.types.contains("Guide"))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID: "linear-overview"))
    }
    func testConstructionIsDistinctFromRunnablePracticeAndReferences() {
        XCTAssertEqual(LearningLibraryPresentation.items.first {$0.id=="affine"}?.kind,"Construction")
        XCTAssertEqual(LearningLibraryPresentation.items.first {$0.id=="tensor-layout"}?.kind,"Construction")
        XCTAssertEqual(LearningLibraryPresentation.items.first {$0.id=="model-output"}?.kind,"Practice")
        XCTAssertEqual(LearningLibraryPresentation.items.first {$0.id=="guided"}?.kind,"Guided")
    }
    func testRetiredPathNavigationFallsBackToTopicOverview() {
        let navigation = LearningLibraryNavigation()
        navigation.showSection("Learning paths")
        XCTAssertEqual(navigation.section, "Overview")
    }
}
