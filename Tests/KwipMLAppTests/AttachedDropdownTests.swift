import XCTest
@testable import KwipMLApp

final class AttachedDropdownTests: XCTestCase {
    private let screen = CGRect(x: 0, y: 0, width: 1000, height: 800)

    func testClosingKeyDropdownRestoresDocumentButNeverAnotherAppOrWindow() {
        XCTAssertTrue(AttachedDropdownFocus.shouldRestoreOwner(panelWasKey: true, ownerVisible: true, applicationActive: true))
        XCTAssertTrue(AttachedDropdownFocus.shouldRestoreOwner(panelWasKey: false, ownerVisible: true, applicationActive: true, keyWindowIsUnassigned: true))
        XCTAssertFalse(AttachedDropdownFocus.shouldRestoreOwner(panelWasKey: false, ownerVisible: true, applicationActive: true))
        XCTAssertFalse(AttachedDropdownFocus.shouldRestoreOwner(panelWasKey: true, ownerVisible: true, applicationActive: false))
        XCTAssertFalse(AttachedDropdownFocus.shouldRestoreOwner(panelWasKey: true, ownerVisible: false, applicationActive: true))
    }

    func testNormalDropdownSharesTriggerWidthAndTouchesItsBottomEdge() {
        let anchor = CGRect(x: 120, y: 500, width: 240, height: 36)
        let frame = AttachedDropdownPlacement.frame(anchor: anchor, visibleScreen: screen, preferredHeight: 190)
        XCTAssertEqual(frame, CGRect(x: 120, y: 310, width: 240, height: 190))
        XCTAssertEqual(frame.maxY, anchor.minY)
    }

    func testLowerScreenEdgePlacesMenuAboveInsteadOfClippingRows() {
        let anchor = CGRect(x: 120, y: 20, width: 240, height: 36)
        let frame = AttachedDropdownPlacement.frame(anchor: anchor, visibleScreen: screen, preferredHeight: 300)
        XCTAssertEqual(frame, CGRect(x: 120, y: 56, width: 240, height: 300))
        XCTAssertEqual(frame.minY, anchor.maxY)
    }

    func testLargeMenusUseTheRoomierSideAndBoundTheirScrollableHeight() {
        let anchor = CGRect(x: 120, y: 300, width: 240, height: 36)
        let frame = AttachedDropdownPlacement.frame(anchor: anchor, visibleScreen: screen, preferredHeight: 900)
        XCTAssertEqual(frame, CGRect(x: 120, y: 336, width: 240, height: 464))
        XCTAssertTrue(screen.contains(frame))
    }

    func testExplicitRunMenuWidthAndNonOriginScreenStayWithinVisibleBounds() {
        let screen = CGRect(x: -1200, y: 40, width: 1000, height: 800)
        let anchor = CGRect(x: -250, y: 600, width: 40, height: 36)
        let frame = AttachedDropdownPlacement.frame(anchor: anchor, visibleScreen: screen, preferredHeight: 200, width: 320)
        XCTAssertEqual(frame, CGRect(x: -520, y: 400, width: 320, height: 200))
        XCTAssertTrue(screen.contains(frame))
    }
}
