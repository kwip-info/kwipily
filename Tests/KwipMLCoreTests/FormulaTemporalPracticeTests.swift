import XCTest
@testable import KwipMLCore

final class FormulaTemporalPracticeTests: XCTestCase {
    func testUnscoredTemporalPracticeKeepsTypedFormulaAndViewFlow() throws {
        let document = try FormulaTemporalPracticeExample.document(projectID: "00000000-0000-0000-0000-000000000013")
        XCTAssertEqual(document.nodes.count, 3)
        XCTAssertEqual(document.edges.count, 2)
        XCTAssertEqual(FormulaTemporalPracticeExample.exampleID, "formula-temporal")
        guard case .array(let operations) = FormulaTemporalPracticeExample.configuration["operations"] else { return XCTFail("Expected ordered operations") }
        XCTAssertEqual(operations.count, 4)
        let data = try Data(contentsOf: FormulaTemporalPracticeExample.observationsURL())
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(text.contains("2024-11-03T01:30:00"))
        XCTAssertTrue(text.contains("2024-03-10T02:30:00"))
    }
}
