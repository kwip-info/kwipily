import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationPhaseMeasurementsTests: XCTestCase {
    private func event(_ change: (inout [String: JSONValue]) -> Void = { _ in }) throws -> Data {
        var record: [String: JSONValue] = ["nodeID": .string("source"), "status": .string("completed"), "cacheHit": .bool(false),
            "milliseconds": .object(Dictionary(uniqueKeysWithValues: PreparationPhaseMeasurements.phases.map { ($0, .number(1)) })),
            "wallMilliseconds": .number(8), "unattributedMilliseconds": .number(2)]
        change(&record)
        return try JSONEncoder().encode(["event": JSONValue.string("dataflowNodeTiming"), "measurement": .object(record)])
    }
    func testBoundedMetadataIsAdmittedOncePerCapturedComponent() throws {
        var ledger = PreparationPhaseMeasurements()
        let data = try event()
        try ledger.consume(data, allowedNodes: ["source"], enabled: true)
        XCTAssertEqual(ledger.records.count, 1)
        XCTAssertThrowsError(try ledger.consume(data, allowedNodes: ["source"], enabled: true))
        var disabled = PreparationPhaseMeasurements()
        XCTAssertThrowsError(try disabled.consume(data, allowedNodes: ["source"], enabled: false))
        XCTAssertThrowsError(try disabled.consume(data, allowedNodes: ["other"], enabled: true))
    }
    func testPrivateProseAndMalformedTimingNeverEnterLedger() throws {
        let mutations: [(inout [String: JSONValue]) -> Void] = [
            { $0["message"] = .string("PRIVATE_PAYLOAD") },
            { $0["wallMilliseconds"] = .number(-1) },
            { $0["wallMilliseconds"] = .bool(true) },
            { $0["wallMilliseconds"] = .number(1) },
            { $0["unattributedMilliseconds"] = .number(100) },
            { $0["status"] = .string("PRIVATE_ERROR_PROSE") },
            { $0["milliseconds"] = .object(["PRIVATE_PHASE": .integer(0)]) }
        ]
        for mutation in mutations {
            var ledger = PreparationPhaseMeasurements()
            XCTAssertThrowsError(try ledger.consume(event(mutation), allowedNodes: ["source"], enabled: true))
            XCTAssertTrue(ledger.records.isEmpty)
        }
    }
    func testFailedObservationRetainsOnlyFixedPhaseAndNotException() throws {
        var ledger = PreparationPhaseMeasurements()
        try ledger.consume(event { $0["status"] = .string("failed"); $0["failurePhase"] = .string("observation") }, allowedNodes: ["source"], enabled: true)
        XCTAssertEqual(ledger.records["source"]?["failurePhase"], .string("observation"))
        var refused = PreparationPhaseMeasurements()
        XCTAssertThrowsError(try refused.consume(event { $0["status"] = .string("failed"); $0["failurePhase"] = .string("/private/path") }, allowedNodes: ["source"], enabled: true))
    }
}
