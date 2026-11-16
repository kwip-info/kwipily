import XCTest
@testable import KwipMLApp

final class CSVSourceDiagnosticTests: XCTestCase {
    private func event(_ updates: [String: Any] = [:]) throws -> Data {
        var diagnostic: [String: Any] = ["code": "rowWidth", "record": 28, "line": 30, "message": "private source text must not reach presentation"]
        diagnostic.merge(updates) { _, new in new }
        return try JSONSerialization.data(withJSONObject: ["event": "error", "message": "raw worker error", "csvDiagnostic": diagnostic])
    }
    func testPositionAndCanonicalMessageSurviveFailureChannel() throws {
        let diagnostic = try XCTUnwrap(CSVSourceDiagnostic.decodeEvent(event()))
        XCTAssertEqual(diagnostic.code, "rowWidth")
        XCTAssertEqual(diagnostic.record, 28)
        XCTAssertEqual(diagnostic.line, 30)
        XCTAssertFalse(diagnostic.message.contains("private"))
        let status = SourceInspectionStatus(phase: .failed, detail: diagnostic.transport)
        XCTAssertEqual(status.csvDiagnostic, diagnostic)
        XCTAssertTrue(status.detail.hasPrefix("Record 28, line 30:"))
        XCTAssertFalse(status.detail.contains("kwip-csv-diagnostic"))
        XCTAssertNil(SourceInspectionStatus(phase: .cancelled, detail: "Cancelled").csvDiagnostic)
    }
    func testRejectsUnknownUnboundedOrMistypedEvidence() throws {
        let updates: [[String: Any]] = [["code": "unknown"], ["record": -1], ["line": 10_000_001], ["record": true], ["line": 1.5], ["message": String(repeating: "x", count: 1025)], ["source": "secret"]]
        for update in updates {
            XCTAssertNil(CSVSourceDiagnostic.decodeEvent(try event(update)))
        }
        XCTAssertNil(CSVSourceDiagnostic.decodeTransport("Source inspection failed"))
        XCTAssertNil(CSVSourceDiagnostic.decodeTransport("kwip-csv-diagnostic:not-base64"))
    }
}
