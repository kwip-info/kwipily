import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LessonNodeLinkTests:XCTestCase {
    func testLinkResolvesExactAddressAndRejectsNonCanvasDestinations() {
        XCTAssertEqual(LessonNodeLink.address(URL(string:"kwip-node://data/observations")!),.init(domain:.data,localID:"observations"))
        XCTAssertEqual(LessonNodeLink.address(URL(string:"kwip-node://model/classifier")!),.init(domain:.model,localID:"classifier"))
        for text in ["https://data/observations","kwip-node://data/observations?x=1","kwip-node://other/observations","kwip-node://data/a/b","kwip-node://data/a%2Fb","kwip-node://data/observations#fragment"] {
            XCTAssertNil(LessonNodeLink.address(URL(string:text)!))
        }
    }
}
