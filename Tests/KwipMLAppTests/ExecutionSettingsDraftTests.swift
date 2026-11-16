import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ExecutionSettingsDraftTests: XCTestCase {
    func testOpeningLegacyWorkloadDoesNotRewritePreferences() throws {
        let draft=ExecutionSettingsDraft(global:.factory)
        XCTAssertEqual(draft.workload.selection,"legacy")
        XCTAssertEqual(try draft.application(),.factory)
        XCTAssertNil(try draft.application().workloadPolicy)
    }
    func testProjectWorkloadIsExplicitAndDoesNotChangeApplicationOrCPU() throws {
        let global=try ExecutionPreferences(cpuThreads:3)
        var draft=ExecutionSettingsDraft(global:global,project:.factory)
        XCTAssertEqual(draft.workload.selection,"inherit")
        try draft.workload.choose("balanced")
        let project=try draft.project()
        XCTAssertEqual(project.workloadPolicy,try ExecutionWorkloadPolicy(preset:.balanced))
        XCTAssertNil(project.cpuThreads)
        XCTAssertNil(global.workloadPolicy)
        XCTAssertEqual(try global.resolve(project:project).workloadPolicy?.limit(.scratchBytes),2_147_483_648)
        try draft.workload.choose("inherit")
        draft.workload.values["scratchBytes"]="invalid inactive text"
        XCTAssertEqual(try draft.project(),.factory)
    }
    func testCustomWorkloadUsesExactBytesAndRejectsInvalidValues() throws {
        var draft=ExecutionSettingsDraft(global:.factory,project:.factory)
        try draft.workload.choose("custom")
        draft.workload.values["scratchBytes"]="123456789"
        let policy=try XCTUnwrap(draft.project().workloadPolicy)
        XCTAssertEqual(policy.preset,.custom)
        XCTAssertEqual(policy.limit(.scratchBytes),123456789)
        XCTAssertEqual(policy.limit(.retainedBytes),200000000)
        for invalid in ["0","-1","NaN","1.5","1e8","9007199254740992"] {
            draft.workload.values["scratchBytes"]=invalid
            XCTAssertThrowsError(try draft.project(),invalid)
        }
    }
    func testStoragePageMergeDoesNotReplaceWorkloadOrExecution() throws {
        let current=try ExecutionPreferences(cpuThreads:4,workloadPolicy:.init(preset:.balanced))
        var draft=ExecutionSettingsDraft(global:.factory)
        draft.storageLimited=true;draft.storageGiB="4"
        let merged=try draft.application(merging:current,scope:.storage)
        XCTAssertEqual(merged.workloadPolicy,current.workloadPolicy)
        XCTAssertEqual(merged.cpuThreads,4)
        XCTAssertEqual(merged.managedStorageLimitBytes,4_294_967_296)
    }
    func testPreparationCacheUsesExactBoundedWholeBytes() throws {
        var draft = ExecutionSettingsDraft(global: .factory)
        draft.preparationCacheBytes = "12345"
        XCTAssertEqual(try draft.application().preparationCacheLimitBytes, 12345)
        for value in ["0", "-1", "200000001", "1.5", "NaN", "1e2"] {
            draft.preparationCacheBytes = value
            XCTAssertThrowsError(try draft.application(), value)
        }
    }
    func testUnitsRoundTripWithoutLosingPersistedBytes() throws {
        let settings = try ExecutionPreferences(maximumRunSeconds: 19, cpuThreads: 3,
            maximumResidentBytes: 134_217_729, managedStorageLimitBytes: Int64.max)
        let draft = ExecutionSettingsDraft(global: settings)
        XCTAssertEqual(try draft.application(), settings)
    }
    func testLimitsRejectInvalidFractionalBytesAndOverflowWithoutClamping() throws {
        var draft = ExecutionSettingsDraft(global: .factory)
        draft.memoryMode = "limit"; draft.memoryGiB = ".125"
        draft.storageLimited = true; draft.storageGiB = "1.5"
        let accepted = try draft.application()
        XCTAssertEqual(accepted.maximumResidentBytes, 134_217_728)
        XCTAssertEqual(accepted.managedStorageLimitBytes, 1_610_612_736)
        for value in ["NaN", "inf", "-1", "0", "0.00000000001", "99999999999999999999", "1e2", "1,5"] {
            draft.storageGiB = value
            XCTAssertThrowsError(try draft.application(), value)
        }
        draft.storageLimited = false; draft.seconds = "1.5"
        XCTAssertThrowsError(try draft.application())
        draft.seconds = "86401"
        XCTAssertThrowsError(try draft.application())
        draft.seconds = "60"; draft.threads = "65"
        XCTAssertThrowsError(try draft.application())
    }
    func testProjectInheritanceAndExplicitUnlimitedRemainDistinct() throws {
        let global = try ExecutionPreferences(maximumRunSeconds: 120, cpuThreads: 4, maximumResidentBytes: 1_073_741_824)
        var draft = ExecutionSettingsDraft(global: global, project: .factory)
        XCTAssertEqual(try draft.project(), .factory)
        draft.seconds = "invalid"; draft.threads = "invalid"; draft.memoryGiB = "invalid"
        XCTAssertEqual(try draft.project(), .factory, "Inactive inherited fields do not block saving")
        draft.overrideSeconds = true
        XCTAssertThrowsError(try draft.project())
        draft.seconds = "30"; draft.overrideThreads = true; draft.threads = "2"; draft.memoryMode = "unlimited"
        let project = try draft.project()
        XCTAssertEqual(project.maximumRunSeconds, 30)
        XCTAssertEqual(project.cpuThreads, 2)
        XCTAssertEqual(project.memory, .unlimited)
        XCTAssertNil(try global.resolve(project: project).budget.resources?.maximumResidentBytes)
    }
}
