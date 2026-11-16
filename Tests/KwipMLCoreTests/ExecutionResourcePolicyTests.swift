import XCTest
@testable import KwipMLCore

final class ExecutionResourcePolicyTests: XCTestCase {
    func testExplicitResourcesUseVersionTwoAndRetainTimeResolution() throws {
        let resources = try ExecutionResourcePolicy(cpuThreads: 8, maximumResidentBytes: 536_870_912)
        let value = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 300,
            capabilityMaximumSeconds: 60, resources: resources)
        XCTAssertEqual(value.schemaVersion, 2); XCTAssertEqual(value.resources, resources)
        XCTAssertEqual(value.requestedSeconds, 300); XCTAssertEqual(value.maximumRunSeconds, 60)
        XCTAssertEqual(value.source, .project)
        XCTAssertEqual(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: JSONEncoder().encode(value)), value)
        let window = try ExecutionBudgetWindow(policy: value, startedAtUptime: 100)
        XCTAssertEqual(window.policy.resources?.cpuThreads, 8)
        XCTAssertEqual(try window.remainingSeconds(atUptime: 120), 40)
    }
    func testHistoricalVersionOneRoundTripsWithoutInventingResources() throws {
        let legacy = Data(#"{"maximumRunSeconds":60,"requestedSeconds":60,"schemaVersion":1,"source":"application"}"#.utf8)
        let decoded = try JSONDecoder().decode(ResolvedExecutionBudget.self, from: legacy)
        XCTAssertNil(decoded.resources); XCTAssertEqual(decoded.schemaVersion, 1)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        XCTAssertEqual(try encoder.encode(decoded), legacy)
        let inherited = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
        XCTAssertEqual(inherited.schemaVersion, 1); XCTAssertNil(inherited.resources)
        let explicitOne = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil,
            resources: ExecutionResourcePolicy(cpuThreads: 1))
        XCTAssertEqual(explicitOne.schemaVersion, 2)
        XCTAssertEqual(explicitOne.resources?.cpuThreads, 1)
    }
    func testResourceBoundsAndExactFieldsRejectInvalidOrUnsupportedValues() throws {
        for threads in [1, 64] { XCTAssertEqual(try ExecutionResourcePolicy(cpuThreads: threads).cpuThreads, threads) }
        for threads in [Int.min, -1, 0, 65, Int.max] { XCTAssertThrowsError(try ExecutionResourcePolicy(cpuThreads: threads)) }
        for bytes in [134_217_728, 1_099_511_627_776] {
            XCTAssertEqual(try ExecutionResourcePolicy(cpuThreads: 1, maximumResidentBytes: bytes).maximumResidentBytes, bytes)
        }
        for bytes in [0, 134_217_727, 1_099_511_627_777, Int.max] {
            XCTAssertThrowsError(try ExecutionResourcePolicy(cpuThreads: 1, maximumResidentBytes: bytes))
        }
        for json in [
            #"{"schemaVersion":1,"cpuThreads":true}"#,
            #"{"schemaVersion":1,"cpuThreads":1.5}"#,
            #"{"schemaVersion":1,"cpuThreads":"4"}"#,
            #"{"schemaVersion":2,"cpuThreads":4}"#,
            #"{"schemaVersion":1}"#,
            #"{"schemaVersion":1,"cpuThreads":4,"gpuMemoryBytes":1}"#,
            #"{"schemaVersion":1,"cpuThreads":4,"maximumResidentBytes":null}"#
        ] { XCTAssertThrowsError(try JSONDecoder().decode(ExecutionResourcePolicy.self, from: Data(json.utf8)), json) }
        let noMemory = try JSONEncoder().encode(ExecutionResourcePolicy(cpuThreads: 4))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: noMemory) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["schemaVersion", "cpuThreads"])
    }
    func testBudgetVersionCannotOmitSmuggleOrNullResources() throws {
        let base: [String: Any] = ["schemaVersion": 2, "requestedSeconds": 60, "maximumRunSeconds": 60, "source": "application"]
        let resources: [String: Any] = ["schemaVersion": 1, "cpuThreads": 4]
        for fields in [base, base.merging(["resources": NSNull()]) { _, new in new },
                       base.merging(["schemaVersion": 1, "resources": resources]) { _, new in new },
                       base.merging(["schemaVersion": 3, "resources": resources]) { _, new in new },
                       base.merging(["resources": resources, "maximumRunSeconds": 61]) { _, new in new }] {
            XCTAssertThrowsError(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: JSONSerialization.data(withJSONObject: fields)))
        }
    }
}
