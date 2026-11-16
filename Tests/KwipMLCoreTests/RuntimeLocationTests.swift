import XCTest
@testable import KwipMLCore

final class RuntimeLocationTests: XCTestCase {
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let runtime = root.appendingPathComponent("KwipMLRuntime")
        try FileManager.default.createDirectory(at: runtime.appendingPathComponent("python/bin"), withIntermediateDirectories: true)
        try Data("#!/bin/sh\n".utf8).write(to: runtime.appendingPathComponent("python/bin/python3.13"))
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtime.appendingPathComponent("python/bin/python3.13").path)
        try Data("# bootstrap".utf8).write(to: runtime.appendingPathComponent("launch.py"))
        try Data("{}".utf8).write(to: runtime.appendingPathComponent("components.json"))
        try Data(#"{"schemaVersion":1,"pythonExecutable":"python/bin/python3.13","entryPoint":"launch.py","registry":"components.json","pythonVersion":"3.13.6","packages":{"torch":"2.14.1"}}"#.utf8).write(to: runtime.appendingPathComponent("runtime-manifest.json"))
        return root
    }
    func testBundledRuntimeResolvesAfterRelocationWithoutDevelopmentRoot() throws {
        let original = try fixture()
        let relocated = original.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + " moved resources")
        try FileManager.default.moveItem(at: original, to: relocated)
        defer { try? FileManager.default.removeItem(at: relocated) }
        let runtime = try RuntimeLocation.resolve(resources: relocated)
        XCTAssertEqual(runtime.kind, .bundled)
        XCTAssertEqual(runtime.packageVersions["torch"], "2.14.1")
        XCTAssertEqual(runtime.executableURL, relocated.appendingPathComponent("KwipMLRuntime/python/bin/python3.13"))
        XCTAssertEqual(runtime.argumentPrefix, ["-I", "-B", relocated.appendingPathComponent("KwipMLRuntime/launch.py").path])
        XCTAssertEqual(runtime.registryURL, relocated.appendingPathComponent("KwipMLRuntime/components.json"))
    }
    func testMalformedBundleNeverFallsBackToDevelopment() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        try Data("{}".utf8).write(to: root.appendingPathComponent("KwipMLRuntime/runtime-manifest.json"))
        XCTAssertThrowsError(try RuntimeLocation.resolve(resources: root, developmentRoot: root, allowDevelopment: true))
    }
    func testEscapingManifestAndSymlinkAreRejected() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let manifest = root.appendingPathComponent("KwipMLRuntime/runtime-manifest.json")
        let original = try String(contentsOf: manifest, encoding: .utf8)
        try original.replacingOccurrences(of: "launch.py", with: "../outside.py").write(to: manifest, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try RuntimeLocation.resolve(resources: root))
        try original.write(to: manifest, atomically: true, encoding: .utf8)
        let entry = root.appendingPathComponent("KwipMLRuntime/launch.py")
        try FileManager.default.removeItem(at: entry)
        try FileManager.default.createSymbolicLink(at: entry, withDestinationURL: URL(fileURLWithPath: "/usr/bin/true"))
        XCTAssertThrowsError(try RuntimeLocation.resolve(resources: root))
    }
    func testDevelopmentRequiresExplicitOptIn() throws {
        XCTAssertThrowsError(try RuntimeLocation.resolve(resources: nil, developmentRoot: URL(fileURLWithPath: "/tmp")))
    }
    func testSingleHelperRuntimeResolvesForHostAndHelperAfterRelocation() throws {
        let source = try fixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let app = source.appendingPathComponent("Nested.app")
        let resources = app.appendingPathComponent("Contents/Resources")
        let helperResources = app.appendingPathComponent("Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: helperResources, withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: source.appendingPathComponent("KwipMLRuntime"), to: helperResources.appendingPathComponent("KwipMLRuntime"))
        XCTAssertEqual(try RuntimeLocation.resolve(resources: resources), try RuntimeLocation.resolve(resources: helperResources))
        let moved = source.appendingPathComponent("Moved.app")
        try FileManager.default.moveItem(at: app, to: moved)
        let runtime = try RuntimeLocation.resolve(resources: moved.appendingPathComponent("Contents/Resources"))
        XCTAssertEqual(runtime.workingDirectoryURL, moved.appendingPathComponent("Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime", isDirectory: true))
    }
    func testDuplicateOrMalformedNestedRuntimeNeverFallsBack() throws {
        let source = try fixture()
        defer { try? FileManager.default.removeItem(at: source) }
        let app = source.appendingPathComponent("Nested.app")
        let resources = app.appendingPathComponent("Contents/Resources")
        let nested = app.appendingPathComponent("Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: nested.appendingPathComponent("runtime-manifest.json"))
        XCTAssertThrowsError(try RuntimeLocation.resolve(resources: resources, developmentRoot: source, allowDevelopment: true))
        try FileManager.default.copyItem(at: source.appendingPathComponent("KwipMLRuntime"), to: resources.appendingPathComponent("KwipMLRuntime"))
        XCTAssertThrowsError(try RuntimeLocation.resolve(resources: resources))
    }
    func testFixedBrokerProgramsResolveInBundledRunnerAndDevelopmentWithoutFallback() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let runtime = try RuntimeLocation.resolve(resources: root)
        let runner = runtime.workingDirectoryURL.appendingPathComponent("runner")
        try FileManager.default.createDirectory(at: runner, withIntermediateDirectories: true)
        for entry in RuntimeLocation.BrokerEntry.allCases {
            let expected = runner.appendingPathComponent(entry.rawValue)
            try Data("# fixed broker".utf8).write(to: expected)
            XCTAssertEqual(try runtime.brokerEntry(entry), expected)
        }
        let development = RuntimeLocation(kind: .development, packageVersions: [:], executableURL: runtime.executableURL,
            entryURL: runner.appendingPathComponent("main.py"), registryURL: runtime.registryURL, workingDirectoryURL: root)
        XCTAssertEqual(try development.brokerEntry(.apiTable), runner.appendingPathComponent("api_table_service.py"))
        let script = runner.appendingPathComponent("api_table_service.py")
        try FileManager.default.removeItem(at: script)
        try Data("# wrong directory must not become fallback".utf8).write(to: runtime.workingDirectoryURL.appendingPathComponent("api_table_service.py"))
        XCTAssertThrowsError(try runtime.brokerEntry(.apiTable))
        try FileManager.default.createSymbolicLink(at: script, withDestinationURL: runtime.entryURL)
        XCTAssertThrowsError(try runtime.brokerEntry(.apiTable))
        let retained = runtime.workingDirectoryURL.appendingPathComponent("retained-runner")
        try FileManager.default.moveItem(at: runner, to: retained)
        try FileManager.default.createSymbolicLink(at: runner, withDestinationURL: retained)
        XCTAssertThrowsError(try runtime.brokerEntry(.databaseOutput))
    }

}
