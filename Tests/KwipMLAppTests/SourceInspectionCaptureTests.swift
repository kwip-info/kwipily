import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SourceInspectionCaptureTests: XCTestCase {
    func testCapturePinsPrivateSourceAndRemovesItAfterUse() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("observations.csv"), bytes = Data("a,b\n1,2\n".utf8)
        try bytes.write(to: source)
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let captured = try SourceInspectionCapture.capture(path: source.path, workspace: root, sourcePermissions: root.appendingPathComponent("permissions"), window: window)
        XCTAssertEqual(captured.sourceURL.lastPathComponent, source.lastPathComponent)
        XCTAssertEqual(captured.sha256, WorkflowArchiveCodec.hash(bytes))
        try Data("changed".utf8).write(to: source)
        XCTAssertEqual(try Data(contentsOf: captured.sourceURL), bytes)
        captured.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: captured.sourceURL.path))
    }
    func testLargeSourceCapturesAndRevalidatesOnlyBoundedPrefix() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("large.csv")
        let prefix = Data(repeating: 65, count: SourceInspectionCapture.maximumSampleBytes)
        var bytes = prefix; bytes.append(Data(repeating: 66, count: 11_000_000))
        try bytes.write(to: source)
        let window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let permissions = root.appendingPathComponent("permissions")
        let capture = try SourceInspectionCapture.capture(path: source.path, workspace: root, sourcePermissions: permissions, window: window)
        defer { capture.remove() }
        XCTAssertEqual(try Data(contentsOf: capture.sourceURL), prefix)
        XCTAssertEqual(capture.sha256, WorkflowArchiveCodec.hash(prefix))
        try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions, window: window, expectedSHA256: capture.sha256)
        // A prefix sample deliberately makes no assertion about uninspected tail bytes.
        bytes[bytes.count - 1] = 67; try bytes.write(to: source)
        try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions, window: window, expectedSHA256: capture.sha256)
        bytes[0] = 67; try bytes.write(to: source)
        XCTAssertThrowsError(try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions, window: window, expectedSHA256: capture.sha256))
    }

    func testStructuredFilePinsTailAndRefusesOversize() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("table.parquet"), permissions = root.appendingPathComponent("permissions")
        var bytes = Data(repeating: 65, count: 1_100_000)
        try bytes.write(to: source)
        let window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let captured = try SourceInspectionCapture.capture(path: source.path, workspace: root, sourcePermissions: permissions, window: window, completeFile: true)
        defer { captured.remove() }
        XCTAssertEqual(try Data(contentsOf: captured.sourceURL), bytes)
        bytes[bytes.count - 1] = 66; try bytes.write(to: source)
        XCTAssertThrowsError(try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions, window: window, expectedSHA256: captured.sha256, completeFile: true))
        try Data(repeating: 65, count: 10_000_001).write(to: source)
        XCTAssertThrowsError(try SourceInspectionCapture.capture(path: source.path, workspace: root, sourcePermissions: permissions, window: window, completeFile: true))
    }

    func testExpiredBudgetRefusesCaptureWithoutLeavingStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 1)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime - 2)
        XCTAssertThrowsError(try SourceInspectionCapture.capture(path: "/missing.csv", workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), window: window)) { error in
            guard case ExecutionBudgetError.wallTimeLimitExceeded = error else { return XCTFail("Unexpected \(error)") }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }
    func testStorageAdmissionRejectsBeforeFirstSourceCopy() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("observations.csv")
        try Data("a,b\n1,2\n".utf8).write(to: source)
        let window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil),
                                               startedAtUptime: ProcessInfo.processInfo.systemUptime)
        XCTAssertThrowsError(try SourceInspectionCapture.capture(path: source.path, workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), window: window,
            projectID: UUID().uuidString, storageLimitBytes: 1))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("source-inspection-staging").path))
        XCTAssertEqual(try Data(contentsOf: source), Data("a,b\n1,2\n".utf8))
    }

    func testSmallSourceAdmissionChargesBothCopiesAndDeinitReleasesOwnedStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("observations.csv"), bytes = Data("a,b\n1,2\n".utf8)
        try bytes.write(to: source)
        let window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil),
                                               startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let projectID = UUID().uuidString, limit = Int64(1_000_000 + 2 * bytes.count)
        XCTAssertThrowsError(try SourceInspectionCapture.capture(path: source.path, workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), window: window,
            projectID: projectID, storageLimitBytes: limit - 1))
        var captured: SourceInspectionCapture? = try SourceInspectionCapture.capture(path: source.path, workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), window: window,
            projectID: projectID, storageLimitBytes: limit)
        XCTAssertEqual(captured?.storageReservation?.maximumCaptureBytes, bytes.count)
        let privateURL = try XCTUnwrap(captured?.sourceURL)
        XCTAssertEqual(try Data(contentsOf: privateURL), bytes)
        XCTAssertNotEqual(privateURL.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().lastPathComponent,
                          captured?.storageReservation?.stageID.uuidString)
        captured = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: privateURL.path))
        let retry = try SourceInspectionCapture.capture(path: source.path, workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), window: window,
            projectID: projectID, storageLimitBytes: limit)
        retry.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: retry.sourceURL.path))
    }

    func testStorageLimitWithoutProjectIdentityRefusesBeforeSourceStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("observations.csv")
        try Data("a\n1\n".utf8).write(to: source)
        let window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil),
                                               startedAtUptime: ProcessInfo.processInfo.systemUptime)
        XCTAssertThrowsError(try SourceInspectionCapture.capture(path: source.path, workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), window: window, storageLimitBytes: 20_000_000))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("source-inspection-staging").path))
    }

    func testCurrentSourceVerificationChecksAuthorizedBytesWithoutCreatingAnotherStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("observations.csv"), bytes = Data("a,b\n1,2\n".utf8)
        try bytes.write(to: source)
        let permissions = root.appendingPathComponent("permissions")
        let window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil),
                                               startedAtUptime: ProcessInfo.processInfo.systemUptime)
        try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions,
                                                   window: window, expectedSHA256: WorkflowArchiveCodec.hash(bytes))
        try Data("a,b\n9,2\n".utf8).write(to: source)
        XCTAssertThrowsError(try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions,
            window: window, expectedSHA256: WorkflowArchiveCodec.hash(bytes)))
        try bytes.write(to: source)
        try SourceAccess(directory: permissions).revoke(path: source.path)
        XCTAssertThrowsError(try SourceInspectionCapture.verifyCurrent(path: source.path, sourcePermissions: permissions,
            window: window, expectedSHA256: WorkflowArchiveCodec.hash(bytes)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("source-inspection-staging").path))
    }

}
