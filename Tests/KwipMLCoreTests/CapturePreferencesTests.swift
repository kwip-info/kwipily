import XCTest
@testable import KwipMLCore

final class CapturePreferencesTests: XCTestCase {
    private let minimal = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false)
    private func decode<T: Decodable>(_ type: T.Type, _ value: [String: Any]) throws -> T {
        try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: value))
    }
    func testLegacyPreferencesMigrateWithoutOverridingResources() throws {
        let global = try decode(ExecutionPreferences.self, ["schemaVersion": 1, "maximumRunSeconds": 91,
            "cpuThreads": 3, "maximumResidentBytes": 268_435_456, "managedStorageLimitBytes": 12345])
        XCTAssertEqual(global.schemaVersion, 3); XCTAssertEqual(global.sampleCapture, .factory)
        XCTAssertEqual(global.maximumRunSeconds, 91); XCTAssertEqual(global.cpuThreads, 3)
        XCTAssertEqual(global.maximumResidentBytes, 268_435_456); XCTAssertEqual(global.managedStorageLimitBytes, 12345)
        let project = try decode(ProjectExecutionOverrides.self, ["schemaVersion": 1,
            "maximumRunSeconds": 17, "cpuThreads": 2, "memory": ["mode": "unlimited"]])
        XCTAssertEqual(project.schemaVersion, 2); XCTAssertNil(project.sampleCapture)
        XCTAssertEqual(project.maximumRunSeconds, 17); XCTAssertEqual(project.cpuThreads, 2); XCTAssertEqual(project.memory, .unlimited)
        XCTAssertEqual(try JSONDecoder().decode(ExecutionPreferences.self, from: JSONEncoder().encode(global)), global)
        XCTAssertEqual(try JSONDecoder().decode(ProjectExecutionOverrides.self, from: JSONEncoder().encode(project)), project)
    }
    func testNewSchemaRequiresGlobalPolicyAndRejectsNullOrLegacyInjection() throws {
        let policy: [String: Any] = ["schemaVersion": 1, "activationValues": false, "predictionRows": true]
        let global: [String: Any] = ["schemaVersion": 2, "maximumRunSeconds": 60, "cpuThreads": 2, "sampleCapture": policy]
        let project: [String: Any] = ["schemaVersion": 2, "memory": ["mode": "inherit"], "sampleCapture": policy]
        XCTAssertEqual(try decode(ExecutionPreferences.self, global).sampleCapture.predictionRows, true)
        XCTAssertEqual(try decode(ProjectExecutionOverrides.self, project).sampleCapture?.activationValues, false)
        var missing = global; missing.removeValue(forKey: "sampleCapture")
        XCTAssertThrowsError(try decode(ExecutionPreferences.self, missing))
        var inherited = project; inherited.removeValue(forKey: "sampleCapture")
        XCTAssertNil(try decode(ProjectExecutionOverrides.self, inherited).sampleCapture)
        for version in [1, 2] {
            for null in [false, true] {
                var g = global, p = project; g["schemaVersion"] = version; p["schemaVersion"] = version
                if null { g["sampleCapture"] = NSNull(); p["sampleCapture"] = NSNull() }
                if version == 1 || null {
                    XCTAssertThrowsError(try decode(ExecutionPreferences.self, g))
                    XCTAssertThrowsError(try decode(ProjectExecutionOverrides.self, p))
                }
            }
        }
        let invalidPolicies: [Any] = [0, ["schemaVersion": 1, "activationValues": 0, "predictionRows": true],
                             ["schemaVersion": 1, "activationValues": false, "predictionRows": true, "extra": false]]
        for invalid in invalidPolicies {
            var g = global, p = project; g["sampleCapture"] = invalid; p["sampleCapture"] = invalid
            XCTAssertThrowsError(try decode(ExecutionPreferences.self, g))
            XCTAssertThrowsError(try decode(ProjectExecutionOverrides.self, p))
        }
    }
    func testResolvedCaptureHasIndependentOriginAndDoesNotChangeBudget() throws {
        let global = try ExecutionPreferences(maximumRunSeconds: 77, cpuThreads: 3, sampleCapture: minimal)
        let inherited = try global.resolve(project: .init(cpuThreads: 2))
        XCTAssertEqual(inherited.sampleCapture, minimal); XCTAssertEqual(inherited.sampleCaptureSource, .application)
        let overridden = try global.resolve(project: .init(cpuThreads: 2, sampleCapture: .factory))
        XCTAssertEqual(overridden.sampleCapture, .factory); XCTAssertEqual(overridden.sampleCaptureSource, .project)
        XCTAssertEqual(overridden.budget, inherited.budget)
    }
    func testCaptureReplacementAndResourceResetAreIndependent() throws {
        let global = try ExecutionPreferences(maximumRunSeconds: 77, cpuThreads: 4,
            maximumResidentBytes: 536_870_912, managedStorageLimitBytes: 999, sampleCapture: minimal)
        let replaced = try global.replacingSampleCapture(.factory)
        XCTAssertEqual(try replaced.resolve().budget, try global.resolve().budget)
        XCTAssertEqual(replaced.sampleCapture, .factory)
        let reset = try global.resettingResources()
        XCTAssertEqual(reset.sampleCapture, minimal)
        XCTAssertEqual(try reset.resolve().budget, try ExecutionPreferences.factory.resolve().budget)
        XCTAssertNil(reset.managedStorageLimitBytes)
        let project = try ProjectExecutionOverrides(maximumRunSeconds: 44, cpuThreads: 2, memory: .unlimited, sampleCapture: minimal)
        let inherited = try project.replacingSampleCapture(nil)
        XCTAssertNil(inherited.sampleCapture); XCTAssertEqual(inherited.maximumRunSeconds, 44)
        XCTAssertEqual(inherited.cpuThreads, 2); XCTAssertEqual(inherited.memory, .unlimited)
        let projectReset = try project.resettingResources()
        XCTAssertEqual(projectReset.sampleCapture, minimal); XCTAssertNil(projectReset.maximumRunSeconds)
        XCTAssertNil(projectReset.cpuThreads); XCTAssertEqual(projectReset.memory, .inherit)
    }
}
