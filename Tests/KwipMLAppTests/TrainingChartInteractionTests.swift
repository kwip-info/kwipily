import XCTest
@testable import KwipMLApp

final class TrainingChartInteractionTests:XCTestCase {
    func testFitResetsBothBoundsAndSubsequentSliderWriteCannotCollapseIt() {
        var viewport=TrainingChartViewport()
        viewport.setUpper(0.047)
        viewport.setLower(0.02)
        XCTAssertFalse(viewport.isFitted)
        viewport.fit()
        XCTAssertEqual(viewport.lower,0)
        XCTAssertEqual(viewport.upper,1)
        // Stable slider domains must not feed a newly clamped endpoint back into Fit.
        viewport.setLower(0)
        XCTAssertEqual(viewport.upper,1)
        XCTAssertTrue(viewport.isFitted)
    }
    func testWindowEdgesRemainOrderedAndRejectNonfiniteValues() {
        var viewport=TrainingChartViewport()
        viewport.setLower(1)
        XCTAssertLessThan(viewport.lower,viewport.upper)
        viewport.setUpper(0)
        XCTAssertGreaterThanOrEqual(viewport.upper-viewport.lower,0.000999999)
        let saved=viewport
        viewport.setLower(.nan);viewport.setUpper(.infinity)
        XCTAssertEqual(viewport,saved)
    }
    func testChartGrowsWithViewportButKeepsTinyDocksScrollable() {
        let small=TrainingChartLayout.plotHeight(availableHeight:220,expanded:false)
        let medium=TrainingChartLayout.plotHeight(availableHeight:500,expanded:false)
        let tall=TrainingChartLayout.plotHeight(availableHeight:800,expanded:false)
        XCTAssertGreaterThanOrEqual(small,120)
        XCTAssertGreaterThan(medium,small)
        XCTAssertGreaterThan(tall,medium)
        XCTAssertLessThan(tall,800)
        XCTAssertEqual(TrainingChartLayout.plotHeight(availableHeight:nil,expanded:false),210)
    }
}
