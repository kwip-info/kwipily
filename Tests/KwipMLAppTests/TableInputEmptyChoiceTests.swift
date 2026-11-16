import XCTest
@testable import KwipMLApp

final class TableInputEmptyChoiceTests: XCTestCase {
    func testExplicitEmptyChoiceShowsAutomaticOrNoneRatherThanMissingSelection() {
        for label in ["Automatic", "None"] {
            let control = FlatChoiceControl(title: "Reader choice", selection: "", options: [.init(id: "", title: label), .init(id: "ms", title: "ms")]) { _ in }
            XCTAssertEqual(control.selectedTitle, label)
        }
        let missing = FlatChoiceControl(title: "Reader choice", selection: "", options: [.init(id: "ms", title: "ms")]) { _ in }
        XCTAssertEqual(missing.selectedTitle, "Choose an option")
    }
}
