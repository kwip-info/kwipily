import XCTest
@testable import KwipMLCore

final class ExecutionWorkloadPolicyTests: XCTestCase {
    func testSharedPresetsRoundTripAndPreserveHistoricalAllowances() throws {
        let legacy = try ExecutionWorkloadPolicy()
        XCTAssertEqual(legacy.limit(.scratchBytes), 100_000_000)
        XCTAssertEqual(legacy.limit(.retainedBytes), 200_000_000)
        XCTAssertEqual(legacy.limit(.parallelTasks), 1)
        for preset in ExecutionWorkloadPolicy.Preset.allCases where preset != .custom {
            let policy = try ExecutionWorkloadPolicy(preset: preset)
            XCTAssertEqual(try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONEncoder().encode(policy)), policy)
        }
    }
    func testCustomCaptureIsIndependentAndCannotImpersonatePreset() throws {
        let baseline = try ExecutionWorkloadPolicy(preset: .balanced)
        let changed = try baseline.overriding(["scratchBytes": 3_000_000_000])
        XCTAssertEqual(changed.preset, .custom)
        XCTAssertEqual(changed.limit(.scratchBytes), 3_000_000_000)
        XCTAssertEqual(baseline.limit(.scratchBytes), 2_147_483_648)
        XCTAssertThrowsError(try ExecutionWorkloadPolicy(preset: .balanced, limits: changed.limits))
        XCTAssertThrowsError(try baseline.overriding(["privatePath": 1]))
        XCTAssertThrowsError(try baseline.overriding(["scratchBytes": 0]))
        XCTAssertThrowsError(try baseline.overriding(["scratchBytes": Int64.max]))
    }
    func testMalformedCapturedFieldsAreRejected() throws {
        let policy = try ExecutionWorkloadPolicy(preset: .balanced)
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(policy)) as? [String: Any])
        for change in ["schema", "bool", "missing", "extra", "fractional"] {
            var value = original
            var limits = try XCTUnwrap(value["limits"] as? [String: Any])
            switch change {
            case "schema": value["schemaVersion"] = true
            case "bool": limits["tableRows"] = true
            case "missing": limits.removeValue(forKey: "tableRows")
            case "extra": limits["privatePath"] = "/private"
            default: limits["tableRows"] = 1.5
            }
            value["limits"] = limits
            XCTAssertThrowsError(try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONSerialization.data(withJSONObject: value)))
        }
    }
    func testApplicationProjectMigrationsAndLaunchCapturePreservePolicy() throws {
        let old = ExecutionPreferences.factory
        XCTAssertNil(old.workloadPolicy)
        XCTAssertEqual(try old.resolve().budget.resources?.workloadPolicy?.preset, .legacy, "New admissions capture unchanged legacy ceilings; stored preferences above remain nil")
        let selected = try ExecutionWorkloadPolicy(preset: .balanced)
        let application = try ExecutionPreferences(workloadPolicy: selected)
        XCTAssertEqual(application.schemaVersion, 5)
        XCTAssertEqual(try JSONDecoder().decode(ExecutionPreferences.self, from: JSONEncoder().encode(application)), application)
        let override = try ProjectExecutionOverrides(workloadPolicy: ExecutionWorkloadPolicy(preset: .light))
        XCTAssertEqual(override.schemaVersion, 4)
        XCTAssertEqual(try JSONDecoder().decode(ProjectExecutionOverrides.self, from: JSONEncoder().encode(override)), override)
        let resolved = try application.resolve(project: override)
        XCTAssertEqual(resolved.workloadPolicy?.preset, .light)
        XCTAssertEqual(resolved.budget.resources?.schemaVersion, 2)
        XCTAssertEqual(try JSONDecoder().decode(ResolvedExecutionBudget.self, from: JSONEncoder().encode(resolved.budget)), resolved.budget)
        XCTAssertEqual(try application.replacingSampleCapture(.factory).workloadPolicy, selected)
        XCTAssertEqual(try application.replacingDerivativeCapture(.factory).workloadPolicy, selected)
        XCTAssertNil(try application.resettingResources().workloadPolicy)
    }

}
