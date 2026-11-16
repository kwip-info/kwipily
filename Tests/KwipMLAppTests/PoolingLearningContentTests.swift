import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PoolingLearningContentTests:XCTestCase {
    @MainActor func testOverviewCoversEightPublishedVariantsAndOriginalAssessedRoutes() {
        let entries=PoolingLearningContent.entries
        let published=Set(BundledPoolingComponents.entries.compactMap { $0.value.published ? $0.key:nil })
        XCTAssertEqual(Set(entries.map(\.componentID)),published)
        XCTAssertEqual(entries.count,8)
        XCTAssertEqual(Set(entries.map(\.lessonID)),["pooling","adaptive-pooling"])
        let fixed:Set<String>=["max_pool1d","max_pool2d","max_pool3d","avg_pool1d"]
        for entry in entries {
            XCTAssertEqual(entry.lessonID,fixed.contains(entry.componentID) ? "pooling":"adaptive-pooling")
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:entry.lessonID),entry.lessonID)
            XCTAssertFalse(entry.title.isEmpty)
            XCTAssertFalse(entry.summary.isEmpty)
        }
        XCTAssertEqual(PoolingLearningContent.lessonID,"pooling-overview")
        XCTAssertEqual(PoolingLearningContent.defaultComponentID,"max_pool2d")
    }

    func testOverviewDoesNotReplaceMathematicalTaskIdentityOrOfferUnsupportedVariants() {
        XCTAssertEqual(Set(PoolingLearningTask.allCases.map(\.rawValue)),["max-pool1d-windows","max-pool2d-windows","max-pool3d-windows","avg-pool1d-windows"])
        XCTAssertEqual(Set(AdaptivePoolingLearningTask.allCases.map(\.componentID)),["avg_pool2d","adaptive_avg_pool1d","adaptive_avg_pool2d","adaptive_max_pool2d"])
        let ids=Set(PoolingLearningContent.entries.map(\.componentID))
        for unsupported in ["avg_pool3d","adaptive_max_pool1d","adaptive_max_pool3d","adaptive_avg_pool3d"] {XCTAssertFalse(ids.contains(unsupported))}
    }
}
