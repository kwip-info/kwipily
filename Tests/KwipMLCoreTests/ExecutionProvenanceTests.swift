import XCTest
@testable import KwipMLCore

final class ExecutionProvenanceTests: XCTestCase {
    func testResolvedPolicySerializesGlobalStorageAndIndependentResourceOrigins() throws {
        let defaults = try ExecutionPreferences(cpuThreads: 4, maximumResidentBytes: 536_870_912, managedStorageLimitBytes: 2_000_000_000)
        let selected = try defaults.resolve(project: .init(cpuThreads: 2, memory: .unlimited)).budget
        let bytes = try JSONEncoder().encode(selected)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertEqual(raw["schemaVersion"] as? Int, 3)
        let provenance = try XCTUnwrap(raw["provenance"] as? [String: Any])
        XCTAssertEqual(provenance["storageScope"] as? String, "workspace")
        XCTAssertEqual(provenance["storageSource"] as? String, "application")
        XCTAssertEqual(provenance["managedStorageLimitBytes"] as? Int64, 2_000_000_000)
        XCTAssertEqual(provenance["cpuThreadsSource"] as? String, "project")
        XCTAssertEqual(provenance["maximumResidentBytesSource"] as? String, "project")
        XCTAssertEqual(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: bytes), selected)
        let unlimited = try JSONSerialization.jsonObject(with: JSONEncoder().encode(ExecutionPreferences.factory.resolve().budget)) as! [String: Any]
        let unlimitedOrigin = try XCTUnwrap(unlimited["provenance"] as? [String: Any])
        XCTAssertNil(unlimitedOrigin["managedStorageLimitBytes"])
        XCTAssertEqual(unlimitedOrigin["cpuThreadsSource"] as? String, "application")
    }
    func testLegacyRemainsUnchangedAndForgedProvenanceIsRejected() throws {
        let legacy = Data(#"{"schemaVersion":1,"requestedSeconds":60,"maximumRunSeconds":60,"source":"application"}"#.utf8)
        let decoded = try JSONDecoder().decode(ResolvedExecutionBudget.self, from: legacy)
        let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        XCTAssertNil(encoded["provenance"])
        let current = try JSONSerialization.jsonObject(with: JSONEncoder().encode(ExecutionPreferences.factory.resolve().budget)) as! [String: Any]
        let origin = try XCTUnwrap(current["provenance"] as? [String: Any])
        for (key, value): (String, Any) in [("storageScope", "project"), ("storageSource", "project"), ("cpuThreadsSource", "unknown"), ("maximumResidentBytesSource", "unknown"), ("managedStorageLimitBytes", 0), ("managedStorageLimitBytes", NSNull()), ("extra", true)] {
            var invalid = current, changed = origin; changed[key] = value; invalid["provenance"] = changed
            XCTAssertThrowsError(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: JSONSerialization.data(withJSONObject: invalid)), key)
        }
        var downgraded = current; downgraded["schemaVersion"] = 2
        XCTAssertThrowsError(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: JSONSerialization.data(withJSONObject: downgraded)))
    }
}
