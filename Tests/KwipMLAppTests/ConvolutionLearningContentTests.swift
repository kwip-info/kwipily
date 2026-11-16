import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConvolutionLearningContentTests:XCTestCase {
    @MainActor func testOverviewCoversSevenPublishedVariantsAndOriginalLessonRoutes() throws {
        let entries=ConvolutionLearningContent.entries
        let published=Set(BundledConvolutionComponents.entries.compactMap { $0.value.published ? $0.key:nil })
        XCTAssertEqual(Set(entries.map(\.componentID)),published.union(["depthwise_conv2d"]))
        XCTAssertNotNil(try ComponentRegistry.builtIn().component(type:"depthwise_conv2d"))
        XCTAssertEqual(entries.count,7)
        XCTAssertEqual(Set(entries.map(\.lessonID)),["convolution","transposed-convolution","depthwise"])
        for entry in entries {
            let expected=entry.componentID == "depthwise_conv2d" ? "depthwise":entry.componentID.hasPrefix("conv_transpose") ? "transposed-convolution":"convolution"
            XCTAssertEqual(entry.lessonID,expected)
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:entry.lessonID),entry.lessonID)
            XCTAssertFalse(entry.title.isEmpty)
            XCTAssertFalse(entry.summary.isEmpty)
        }
        XCTAssertEqual(ConvolutionLearningContent.lessonID,"convolution-overview")
        XCTAssertEqual(ConvolutionLearningContent.defaultComponentID,"conv2d")
    }

    func testOverviewPreservesMathematicalTasksAndExcludesUnsupportedDepthwiseDimensions() {
        XCTAssertEqual(Set(ConvolutionLearningTask.allCases.map(\.componentID)),["conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d"])
        let ids=Set(ConvolutionLearningContent.entries.map(\.componentID))
        XCTAssertFalse(ids.contains("depthwise_conv1d"))
        XCTAssertFalse(ids.contains("depthwise_conv3d"))
        XCTAssertFalse(ids.contains("separable_conv2d"))
    }
}
