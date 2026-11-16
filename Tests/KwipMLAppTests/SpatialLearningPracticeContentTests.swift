import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SpatialLearningPracticeContentTests: XCTestCase {
    func testEveryDeliveredSpatialPrimitiveAndSeparableRecipeHasPracticeAndLaunchMapping() throws {
        let registry = try ComponentRegistry.builtIn()
        let ids = Set(registry.components.map(\.id).filter { $0.contains("conv") || $0.contains("pool") })
        let laterPooling: Set<String> = ["avg_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d"]
        XCTAssertTrue(laterPooling.isSubset(of: ids))
        XCTAssertTrue(laterPooling.isSubset(of: Set(LearnCatalogContent.componentExampleEntries.map(\.componentID))))
        let content = SpatialLearningPracticeContent.entries
        XCTAssertEqual(Set(content.map(\.id)), ids.subtracting(laterPooling).union(["separable_conv2d"]))
        XCTAssertEqual(content.count, 12)
        XCTAssertEqual(Set(content.map(\.id)).count, content.count)
        XCTAssertEqual(Set(content.map(\.launcher)).count, content.count)
    }
    func testTasksAlwaysPredictChangeExecuteAndInspectWithoutClaimingAssessment() {
        for entry in SpatialLearningPracticeContent.entries {
            XCTAssertFalse(entry.prediction.isEmpty, entry.id)
            XCTAssertFalse(entry.change.isEmpty, entry.id)
            XCTAssertFalse(entry.inspection.isEmpty, entry.id)
            XCTAssertFalse(entry.isAssessed, entry.id)
        }
        XCTAssertTrue(SpatialLearningPracticeContent.disclosure.contains("not scored"))
        XCTAssertTrue(SpatialLearningPracticeContent.disclosure.contains("synthetic"))
        XCTAssertTrue(SpatialLearningPracticeContent.runInstruction.contains("actual run"))
        XCTAssertTrue(SpatialLearningPracticeContent.runInstruction.contains("unavailable"))
    }
    func testSharedTermsAndMisconceptionBoundariesAreExplained() {
        let terms = SpatialLearningPracticeContent.terms
        for name in ["Window", "Channel", "Stride", "Padding", "Dilation", "Batch"] {
            XCTAssertTrue(terms.contains("**" + name + ":**"), name)
        }
        let transposed = SpatialLearningPracticeContent.entries.filter { $0.id.contains("transpose") }
        XCTAssertEqual(transposed.count, 3)
        XCTAssertTrue(transposed.allSatisfy { $0.prediction.contains("not an inverse") })
        let average = SpatialLearningPracticeContent.entries.first { $0.id == "avg_pool1d" }!
        XCTAssertTrue(average.change.contains("padding"))
        XCTAssertTrue(average.inspection.contains("divisor"))
    }
}
