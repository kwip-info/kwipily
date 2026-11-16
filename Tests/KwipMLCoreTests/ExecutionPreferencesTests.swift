import XCTest
@testable import KwipMLCore

final class ExecutionPreferencesTests: XCTestCase {
    func testFactoryAndIndependentProjectResourceResolution() throws {
        let factory = try ExecutionPreferences.factory.resolve()
        XCTAssertEqual(factory.budget.maximumRunSeconds, 3600)
        XCTAssertEqual(factory.budget.resources?.cpuThreads, 1)
        XCTAssertNil(factory.budget.resources?.maximumResidentBytes)
        XCTAssertNil(factory.storageLimitBytes)
        let global = try ExecutionPreferences(maximumRunSeconds: 120, cpuThreads: 3,
            maximumResidentBytes: 536_870_912, managedStorageLimitBytes: 2_000_000_000)
        let project = try ProjectExecutionOverrides(maximumRunSeconds: 80, cpuThreads: 2, memory: .unlimited)
        let resolved = try global.resolve(project: project, capabilityMaximumSeconds: 10)
        XCTAssertEqual(resolved.budget.requestedSeconds, 80)
        XCTAssertEqual(resolved.budget.maximumRunSeconds, 10)
        XCTAssertEqual(resolved.budget.source, .project)
        XCTAssertEqual(resolved.budget.capabilityMaximumSeconds, 10)
        XCTAssertEqual(resolved.budget.resources?.cpuThreads, 2)
        XCTAssertNil(resolved.budget.resources?.maximumResidentBytes)
        XCTAssertEqual(resolved.cpuThreadsSource, .project)
        XCTAssertEqual(resolved.maximumResidentBytesSource, .project)
        XCTAssertEqual(resolved.storageLimitBytes, 2_000_000_000)
        let inherited = try global.resolve(project: .factory)
        XCTAssertEqual(inherited.budget.source, .application)
        XCTAssertEqual(inherited.budget.resources?.maximumResidentBytes, 536_870_912)
        XCTAssertEqual(inherited.cpuThreadsSource, .application)
        XCTAssertEqual(inherited.maximumResidentBytesSource, .application)
        let memoryOnly = try global.resolve(project: .init(memory: .limit(268_435_456)))
        XCTAssertEqual(memoryOnly.budget.source, .application)
        XCTAssertEqual(memoryOnly.budget.resources?.cpuThreads, 3)
        XCTAssertEqual(memoryOnly.budget.resources?.maximumResidentBytes, 268_435_456)
        XCTAssertEqual(memoryOnly.maximumResidentBytesSource, .project)
    }
    func testBoundsAreValidatedBeforePersistenceOrResolution() throws {
        for seconds in [0, 86_401] {
            XCTAssertThrowsError(try ExecutionPreferences(maximumRunSeconds: seconds))
            XCTAssertThrowsError(try ProjectExecutionOverrides(maximumRunSeconds: seconds))
        }
        for threads in [0, 65] {
            XCTAssertThrowsError(try ExecutionPreferences(cpuThreads: threads))
            XCTAssertThrowsError(try ProjectExecutionOverrides(cpuThreads: threads))
        }
        for memory in [0, 134_217_727, 1_099_511_627_777] {
            XCTAssertThrowsError(try ExecutionPreferences(maximumResidentBytes: memory))
            XCTAssertThrowsError(try ProjectExecutionOverrides(memory: .limit(memory)))
        }
        for bytes in [Int64(0), -1] { XCTAssertThrowsError(try ExecutionPreferences(managedStorageLimitBytes: bytes)) }
        XCTAssertNoThrow(try ExecutionPreferences(maximumRunSeconds: 86_400, cpuThreads: 64,
            maximumResidentBytes: 1_099_511_627_776, managedStorageLimitBytes: Int64.max))
        XCTAssertThrowsError(try ExecutionPreferences.factory.resolve(capabilityMaximumSeconds: 0))
    }
    func testStrictCodableAndExplicitMemoryModes() throws {
        let value = try ExecutionPreferences(maximumRunSeconds: 75, cpuThreads: 8, maximumResidentBytes: 268_435_456, managedStorageLimitBytes: 1_000_000_000)
        XCTAssertEqual(try JSONDecoder().decode(ExecutionPreferences.self, from: JSONEncoder().encode(value)), value)
        for memory in [ProjectExecutionOverrides.MemoryOverride.inherit, .unlimited, .limit(268_435_456)] {
            let project = try ProjectExecutionOverrides(cpuThreads: 4, memory: memory)
            XCTAssertEqual(try JSONDecoder().decode(ProjectExecutionOverrides.self, from: JSONEncoder().encode(project)), project)
        }
        for text in [
            #"{"schemaVersion":2,"maximumRunSeconds":3600,"cpuThreads":1}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":3600,"cpuThreads":1,"maximumResidentBytes":null}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":3600,"cpuThreads":1,"managedStorageLimitBytes":null}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":3600,"cpuThreads":1,"extra":1}"#,
            #"{"schemaVersion":1,"maximumRunSeconds":true,"cpuThreads":1}"#
        ] { XCTAssertThrowsError(try JSONDecoder().decode(ExecutionPreferences.self, from: Data(text.utf8))) }
        for text in [
            #"{"schemaVersion":1,"memory":{"mode":"inherit"},"managedStorageLimitBytes":5}"#,
            #"{"schemaVersion":1,"memory":{"mode":"inherit"},"cpuThreads":null}"#,
            #"{"schemaVersion":1,"memory":{"mode":"inherit","maximumResidentBytes":268435456}}"#,
            #"{"schemaVersion":1,"memory":{"mode":"unlimited","extra":true}}"#,
            #"{"schemaVersion":1,"memory":{"mode":"limit"}}"#,
            #"{"schemaVersion":1,"memory":{"mode":"limit","maximumResidentBytes":null}}"#,
            #"{"schemaVersion":1,"memory":null}"#
        ] { XCTAssertThrowsError(try JSONDecoder().decode(ProjectExecutionOverrides.self, from: Data(text.utf8))) }
    }
    func testSchemaMigrationAndExecutionResetPreserveAllOtherPreferences() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("preferences.json"), store = AppPreferencesStore(fileURL: file)
        for version in [1, 2] {
            var object: [String: Any] = ["schemaVersion": version,
                "canvasDefaults": ["flowDirection": "topToBottom", "nodeDetail": "small"],
                "shortcutOverrides": ["learningLibrary": ["key": "k", "command": true, "shift": true, "option": false, "control": false]]]
            if version == 2 { object["appearance"] = ["schemaVersion": 1, "mode": "light", "palette": "teal", "transparency": "subtle"] }
            let bytes = try JSONSerialization.data(withJSONObject: object)
            try bytes.write(to: file)
            var migrated = try store.load()
            XCTAssertEqual(migrated.schemaVersion, 3); XCTAssertEqual(migrated.execution, .factory)
            XCTAssertEqual(try Data(contentsOf: file), bytes)
            migrated.execution = try .init(maximumRunSeconds: 90, cpuThreads: 2, managedStorageLimitBytes: 800_000_000)
            try store.save(migrated)
            XCTAssertEqual(try store.load(), migrated)
            XCTAssertLessThanOrEqual(try Data(contentsOf: file).count, 4096)
            let reset = try store.resetExecution()
            XCTAssertEqual(reset.execution, .factory)
            XCTAssertEqual(reset.appearance, migrated.appearance)
            XCTAssertEqual(reset.canvasDefaults, migrated.canvasDefaults)
            XCTAssertEqual(reset.shortcutOverrides, migrated.shortcutOverrides)
            try store.save(migrated)
            XCTAssertEqual(try store.resetAppearance().execution, migrated.execution)
            XCTAssertEqual(try store.resetCanvasDefaults().execution, migrated.execution)
            XCTAssertEqual(try store.resetShortcuts().execution, migrated.execution)
        }
    }
    func testInvalidExecutionCannotBeSmuggledIntoOldSchemaOrResetAway() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("preferences.json"), store = AppPreferencesStore(fileURL: file)
        let encoded = try JSONEncoder().encode(AppPreferences.factory)
        let base = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var bad = base; bad["execution"] = ["schemaVersion": 1, "maximumRunSeconds": 0, "cpuThreads": 1]
        var legacy = base; legacy["schemaVersion"] = 2
        var missing = base; missing.removeValue(forKey: "execution")
        var null = base; null["execution"] = NSNull()
        for object in [bad, legacy, missing, null] {
            let bytes = try JSONSerialization.data(withJSONObject: object)
            try bytes.write(to: file)
            XCTAssertThrowsError(try store.load())
            XCTAssertThrowsError(try store.resetExecution())
            XCTAssertEqual(try Data(contentsOf: file), bytes)
        }
    }
}
