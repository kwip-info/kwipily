import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NormalizationLearningContentTests:XCTestCase {
    @MainActor func testOverviewCoversPublishedVariantsAndOriginalAssessedRoutes() {
        let entries=NormalizationLearningContent.entries
        let published=Set(BundledNormalizationComponents.entries.compactMap { $0.value.published ? $0.key:nil })
        XCTAssertEqual(Set(entries.map(\.componentID)),published)
        XCTAssertEqual(entries.count,7)
        XCTAssertEqual(Set(entries.map(\.lessonID)),["layernorm","channel-normalization","rms_norm"])
        for entry in entries {
            XCTAssertEqual(entry.lessonID,entry.componentID == "rms_norm" ? "rms_norm":entry.componentID == "layer_norm" ? "layernorm":"channel-normalization")
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:entry.lessonID),entry.lessonID)
            XCTAssertFalse(entry.title.isEmpty)
            XCTAssertFalse(entry.summary.isEmpty)
        }
        XCTAssertEqual(NormalizationLearningContent.lessonID,"normalization-overview")
        XCTAssertEqual(NormalizationLearningContent.defaultComponentID,"layer_norm")
        XCTAssertNotEqual(NormalizationLearningContent.lessonID,"normalization")
    }

    func testOriginalChannelTasksRemainDistinctFromUnsupportedOperations() {
        XCTAssertEqual(Set(StatefulNormalizationLearningTask.allCases.map(\.componentID)),["batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm"])
        let ids=Set(NormalizationLearningContent.entries.map(\.componentID))
        for unsupported in ["batch_norm3d","instance_norm3d","local_response_norm"] {XCTAssertFalse(ids.contains(unsupported))}
    }
}
