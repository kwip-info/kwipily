import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasNodeSurfaceStyleTests: XCTestCase {
    func testLegacySavedDetailsCannotChangeUniversalFrameOrSelection() {
        let baseline = CanvasNodeSurfaceStyle(detail:.medium)
        for detail in CanvasNodeDetail.allCases {
            let style = CanvasNodeSurfaceStyle(detail:detail)
            XCTAssertEqual(style,baseline)
            XCTAssertGreaterThan(style.cornerRadius,0)
            XCTAssertGreaterThan(style.borderWidth(isSelected:true),style.borderWidth(isSelected:false))
            XCTAssertLessThan(style.iconSize,CanvasNodeGeometry.headerHeight)
        }
    }
}
