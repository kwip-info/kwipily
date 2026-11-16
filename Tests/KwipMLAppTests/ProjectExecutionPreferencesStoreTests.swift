import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ProjectExecutionPreferencesStoreTests: XCTestCase {
    private func root() -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func file(_ root: URL, _ project: String) -> URL {
        root.appendingPathComponent("project-execution-preferences/\(project).json")
    }

    func testMissingProjectInheritsWithoutCreatingPreferences() throws {
        let root = root(), project = UUID().uuidString
        let store = ProjectExecutionPreferencesStore(workspace: root)
        XCTAssertEqual(try store.load(projectID: project), .factory)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    func testRoundTripAndResetAffectOnlySelectedProject() throws {
        let root = root(), first = UUID().uuidString, second = UUID().uuidString
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let sentinel = root.appendingPathComponent("app-preferences.json")
        try Data("unchanged global preferences".utf8).write(to: sentinel)
        let store = ProjectExecutionPreferencesStore(workspace: root)
        let overridden = try ProjectExecutionOverrides(maximumRunSeconds: 90, cpuThreads: 4, memory: .limit(134_217_728))
        let unlimited = try ProjectExecutionOverrides(cpuThreads: 2, memory: .unlimited)
        try store.save(overridden, projectID: first)
        try store.save(unlimited, projectID: second)
        XCTAssertEqual(try store.load(projectID: first), overridden)
        XCTAssertEqual(try store.load(projectID: second), unlimited)
        let otherBytes = try Data(contentsOf: file(root, second))
        XCTAssertEqual(try store.reset(projectID: first), .factory)
        XCTAssertEqual(try store.load(projectID: first), .factory)
        XCTAssertEqual(try Data(contentsOf: file(root, second)), otherBytes)
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("unchanged global preferences".utf8))
    }

    func testCopiedOrExtendedEnvelopeIsRejectedInsteadOfApplyingAnotherProjectsOverrides() throws {
        let root = root(), first = UUID().uuidString, second = UUID().uuidString
        let store = ProjectExecutionPreferencesStore(workspace: root)
        try store.save(try .init(cpuThreads: 3), projectID: first)
        let firstBytes = try Data(contentsOf: file(root, first))
        try firstBytes.write(to: file(root, second))
        XCTAssertThrowsError(try store.load(projectID: second))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: firstBytes) as? [String: Any])
        object["unexpected"] = true
        try JSONSerialization.data(withJSONObject: object).write(to: file(root, first))
        XCTAssertThrowsError(try store.load(projectID: first))
        try Data(repeating: 32, count: 4097).write(to: file(root, first))
        XCTAssertThrowsError(try store.load(projectID: first))
    }

    func testUnsafeProjectIDsAndSymlinkedPreferenceTargetsArePreserved() throws {
        let root = root(), project = UUID().uuidString
        let store = ProjectExecutionPreferencesStore(workspace: root)
        for invalid in ["../outside", "not-an-id", ""] {
            XCTAssertThrowsError(try store.load(projectID: invalid))
            XCTAssertThrowsError(try store.save(.factory, projectID: invalid))
            XCTAssertThrowsError(try store.reset(projectID: invalid))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        try FileManager.default.createDirectory(at: file(root, project).deletingLastPathComponent(), withIntermediateDirectories: true)
        let sentinel = root.appendingPathComponent("outside.json"), bytes = Data("preserve".utf8)
        try bytes.write(to: sentinel)
        try FileManager.default.createSymbolicLink(at: file(root, project), withDestinationURL: sentinel)
        XCTAssertThrowsError(try store.load(projectID: project))
        XCTAssertThrowsError(try store.save(.factory, projectID: project))
        XCTAssertThrowsError(try store.reset(projectID: project))
        XCTAssertEqual(try Data(contentsOf: sentinel), bytes)
    }

    func testSymlinkedPreferenceNamespaceCannotRedirectSaving() throws {
        let root = root(), other = self.root(), project = UUID().uuidString
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("project-execution-preferences"), withDestinationURL: other)
        let store = ProjectExecutionPreferencesStore(workspace: root)
        XCTAssertThrowsError(try store.save(.factory, projectID: project))
        XCTAssertThrowsError(try store.load(projectID: project))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: other.path).isEmpty)
    }
}
