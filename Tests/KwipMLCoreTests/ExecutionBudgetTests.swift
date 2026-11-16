import XCTest
@testable import KwipMLCore

final class ExecutionBudgetTests: XCTestCase {
    func testResolvesGlobalDefaultAndExplicitProjectOverride() throws {
        let global = try ExecutionBudgetPreferences(maximumRunSeconds: 3600)
        let inherited = try ResolvedExecutionBudget.resolve(global: global, projectMaximumRunSeconds: nil)
        XCTAssertEqual(inherited.requestedSeconds, 3600)
        XCTAssertEqual(inherited.maximumRunSeconds, 3600)
        XCTAssertEqual(inherited.source, .application)
        let project = try ResolvedExecutionBudget.resolve(global: global, projectMaximumRunSeconds: 120)
        XCTAssertEqual(project.requestedSeconds, 120)
        XCTAssertEqual(project.maximumRunSeconds, 120)
        XCTAssertEqual(project.source, .project)
        XCTAssertEqual(global.maximumRunSeconds, 3600)
    }

    func testCapabilityCeilingIsDisclosedWithoutErasingRequestedPolicy() throws {
        let global = try ExecutionBudgetPreferences(maximumRunSeconds: 3600)
        let result = try ResolvedExecutionBudget.resolve(global: global, projectMaximumRunSeconds: 300, capabilityMaximumSeconds: 60)
        XCTAssertEqual(result.requestedSeconds, 300)
        XCTAssertEqual(result.maximumRunSeconds, 60)
        XCTAssertEqual(result.capabilityMaximumSeconds, 60)
        XCTAssertEqual(result.source, .project)
        let stricter = try ResolvedExecutionBudget.resolve(global: global, projectMaximumRunSeconds: 10, capabilityMaximumSeconds: 60)
        XCTAssertEqual(stricter.maximumRunSeconds, 10)
    }

    func testCapturedBudgetDoesNotChangeWhenDefaultsChange() throws {
        var global = try ExecutionBudgetPreferences(maximumRunSeconds: 90)
        let captured = try ResolvedExecutionBudget.resolve(global: global, projectMaximumRunSeconds: nil)
        global = try ExecutionBudgetPreferences(maximumRunSeconds: 600)
        XCTAssertEqual(captured.maximumRunSeconds, 90)
        XCTAssertEqual(try ResolvedExecutionBudget.resolve(global: global, projectMaximumRunSeconds: nil).maximumRunSeconds, 600)
        let data = try JSONEncoder().encode(captured)
        XCTAssertEqual(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: data), captured)
    }

    func testRejectsInvalidRangesAndMalformedEncodedPolicy() throws {
        for seconds in [Int.min, -1, 0, 86401, Int.max] {
            XCTAssertThrowsError(try ExecutionBudgetPreferences(maximumRunSeconds: seconds))
            XCTAssertThrowsError(try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: seconds))
            XCTAssertThrowsError(try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil, capabilityMaximumSeconds: seconds))
        }
        for seconds in [1, 86400] {
            XCTAssertEqual(try ExecutionBudgetPreferences(maximumRunSeconds: seconds).maximumRunSeconds, seconds)
        }
        for json in [
            #"{"schemaVersion":2,"maximumRunSeconds":60}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":true}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":1.5}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":60,"ignored":1}"#,
            #"{"schemaVersion":1}"#
        ] {
            XCTAssertThrowsError(try JSONDecoder().decode(ExecutionBudgetPreferences.self, from: Data(json.utf8)), json)
        }
    }

    func testOperationWindowSharesOneAllowanceAcrossPhasesAndRejectsExpiredLaunches() throws {
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: 100)
        XCTAssertEqual(try window.remainingSeconds(atUptime: 100), 60)
        XCTAssertEqual(try window.remainingSeconds(atUptime: 120.25), 39.75)
        XCTAssertEqual(try window.remainingSeconds(atUptime: 159.5), 0.5)
        XCTAssertThrowsError(try window.remainingSeconds(atUptime: 160))
        XCTAssertThrowsError(try window.remainingSeconds(atUptime: 161))
        XCTAssertThrowsError(try window.remainingSeconds(atUptime: 99))
        for invalid in [Double.nan, .infinity, -.infinity, -1] {
            XCTAssertThrowsError(try ExecutionBudgetWindow(policy: policy, startedAtUptime: invalid))
            XCTAssertThrowsError(try window.remainingSeconds(atUptime: invalid))
        }
    }

    func testRejectsForgedResolvedBudgetAndUnknownFields() throws {
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 300, capabilityMaximumSeconds: 60)
        let valid = try JSONEncoder().encode(policy)
        for (key, value) in [("maximumRunSeconds", 61 as Any), ("requestedSeconds", 0 as Any), ("schemaVersion", 2 as Any), ("source", "unknown" as Any), ("ignored", 1 as Any)] {
            var fields = try XCTUnwrap(JSONSerialization.jsonObject(with: valid) as? [String: Any])
            fields[key] = value
            XCTAssertThrowsError(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: JSONSerialization.data(withJSONObject: fields)), key)
        }
    }
}
