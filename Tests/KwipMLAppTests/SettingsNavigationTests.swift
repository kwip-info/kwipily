import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SettingsNavigationTests: XCTestCase {
    func testSearchRoutesControlsToOnePageInPriorityOrder() {
        XCTAssertEqual(SettingsPage.matches(""), SettingsPage.allCases)
        XCTAssertEqual(SettingsPage.matches("CPU threads"), [.execution])
        XCTAssertEqual(SettingsPage.matches("automatic update"), [.updates])
        XCTAssertEqual(SettingsPage.matches("cache"), [.storage])
        XCTAssertEqual(SettingsPage.matches("source access"), [.permissions])
        XCTAssertEqual(SettingsPage.matches(" LIGHT "), [.appearance])
        XCTAssertTrue(SettingsPage.matches("unknown feature").isEmpty)
    }
    func testApplyingOneResourcePagePreservesOtherPagesLatestValues() throws {
        let initial = ExecutionPreferences.factory
        var draft = ExecutionSettingsDraft(global: initial)
        draft.threads = "4"
        let latest = try ExecutionPreferences(cpuThreads: 2, managedStorageLimitBytes: 123456789, sampleCapture: .init(activationValues: false, predictionRows: false), preparationCacheLimitBytes: 12345)
        let saved = try draft.application(merging: latest, scope: .execution)
        XCTAssertEqual(saved.cpuThreads, 4)
        XCTAssertEqual(saved.managedStorageLimitBytes, latest.managedStorageLimitBytes)
        XCTAssertEqual(saved.preparationCacheLimitBytes, latest.preparationCacheLimitBytes)
        XCTAssertEqual(saved.sampleCapture, latest.sampleCapture)
        draft = ExecutionSettingsDraft(global: initial); draft.preparationCacheBytes = "9000"
        let storage = try draft.application(merging: saved, scope: .storage)
        XCTAssertEqual(storage.cpuThreads, 4)
        XCTAssertEqual(storage.preparationCacheLimitBytes, 9000)
        XCTAssertEqual(storage.sampleCapture, latest.sampleCapture)
    }
    func testScopedResetsPreserveCaptureAndOtherResourcePage() throws {
        let latest = try ExecutionPreferences(maximumRunSeconds: 99, cpuThreads: 7, maximumResidentBytes: 134217728,
            managedStorageLimitBytes: 87654321, sampleCapture: .init(activationValues: false, predictionRows: true), preparationCacheLimitBytes: 123)
        let defaults = ExecutionSettingsDraft(global: .factory)
        let execution = try defaults.application(merging: latest, scope: .execution)
        XCTAssertEqual(execution.maximumRunSeconds, 3600)
        XCTAssertEqual(execution.cpuThreads, 1)
        XCTAssertNil(execution.maximumResidentBytes)
        XCTAssertEqual(execution.managedStorageLimitBytes, 87654321)
        XCTAssertEqual(execution.preparationCacheLimitBytes, 123)
        XCTAssertFalse(execution.sampleCapture.activationValues)
        let storage = try defaults.application(merging: latest, scope: .storage)
        XCTAssertEqual(storage.maximumRunSeconds, 99)
        XCTAssertEqual(storage.cpuThreads, 7)
        XCTAssertEqual(storage.maximumResidentBytes, 134217728)
        XCTAssertNil(storage.managedStorageLimitBytes)
        XCTAssertEqual(storage.preparationCacheLimitBytes, 200000000)
        XCTAssertFalse(storage.sampleCapture.activationValues)
    }

}
