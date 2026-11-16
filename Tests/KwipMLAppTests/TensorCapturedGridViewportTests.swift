import AppKit
import XCTest
@testable import KwipMLApp

@MainActor final class TensorCapturedGridViewportTests:XCTestCase {
    func testHorizontalRangeTracksCurrentColumnsAndClampsOldPosition() {
        let view=TensorCapturedGridScrolls(frame:.init(x:0,y:0,width:1000,height:240))
        func columns(_ count:Int) {
            view.mainTable.tableColumns.forEach {view.mainTable.removeTableColumn($0)}
            for index in 0..<count {
                let column=NSTableColumn(identifier:.init(String(index)));column.width=170
                view.mainTable.addTableColumn(column)
            }
            view.layoutSubtreeIfNeeded();view.resizeDocuments()
        }
        columns(10)
        XCTAssertTrue(view.main.hasHorizontalScroller)
        XCTAssertGreaterThan(view.mainTable.frame.width,view.main.contentSize.width)
        view.main.contentView.scroll(to:.init(x:500,y:0))
        columns(3)
        XCTAssertFalse(view.main.hasHorizontalScroller)
        XCTAssertEqual(view.mainTable.frame.width,view.main.contentSize.width,accuracy:1)
        XCTAssertEqual(view.main.contentView.bounds.minX,0,accuracy:1)
        XCTAssertFalse(view.gutter.hasHorizontalScroller)
    }
}
