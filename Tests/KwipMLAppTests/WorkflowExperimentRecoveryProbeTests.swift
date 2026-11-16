import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentRecoveryProbeTests:XCTestCase {
    private func snapshot(_ root:URL)throws->[String:Data] {
        let enumerator=try XCTUnwrap(FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey]))
        var result:[String:Data]=[:]
        while let path=enumerator.nextObject() as? URL {
            if try path.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true {
                result[String(path.path.dropFirst(root.path.count))]=try Data(contentsOf:path)
            }
        }
        return result
    }
    private func assertReleased(_ directory:URL)throws {
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:directory.lastPathComponent)
        recovery.close()
    }
    @MainActor func testLiveOwnerAndSharedActivityAreInUseWithoutChangingFiles()throws {
        let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
        defer {f.owner.close();try? FileManager.default.removeItem(at:f.root)}
        let before=try snapshot(f.root)
        guard case .inUse=try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root) else {return XCTFail("Live owner must be in use")}
        XCTAssertEqual(try snapshot(f.root),before)
        let activity=try WorkflowFileLease(directory:f.directory,name:"activity.lock",mode:.shared)
        defer {activity.close()}
        f.owner.close()
        guard case .inUse=try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root) else {return XCTFail("Shared worker activity must be in use")}
        XCTAssertEqual(try snapshot(f.root),before)
        activity.close();try assertReleased(f.directory)
    }
    @MainActor func testInactivePendingIsRecoverableButProbeDoesNotCommit()throws {
        let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
        defer {try? FileManager.default.removeItem(at:f.root)}
        f.owner.close();let before=try snapshot(f.root)
        guard case .recoverable=try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root) else {return XCTFail("Inactive pending work is recoverable")}
        XCTAssertEqual(try snapshot(f.root),before)
        try assertReleased(f.directory)
    }
    @MainActor func testTerminalCancelledAndCompletedGroupsRemainUnchanged()throws {
        for completed in [false,true] {
            let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
            defer {f.owner.close();try? FileManager.default.removeItem(at:f.root)}
            if completed {
                for index in 0..<f.requests.count {
                    let run=UUID(),directory=UUID()
                    try f.store.begin(ordinal:index+1,runID:run,directoryID:directory)
                    try f.store.finish(ordinal:index+1,runID:run,directoryID:directory,requestIdentity:f.requests[index].requestIdentity,status:.completed,outcomeSHA256:String(repeating:"a",count:64))
                }
            } else {try f.store.cancelPending()}
            f.owner.close();let before=try snapshot(f.root)
            guard case .terminal=try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root) else {return XCTFail("Terminal ledger needs no reconciliation")}
            XCTAssertEqual(try snapshot(f.root),before)
            try assertReleased(f.directory)
        }
    }
    @MainActor func testLegacyAndMalformedProtocolAreUnavailableWithoutWrites()throws {
        for legacy in [false,true] {
            let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
            defer {f.owner.close();try? FileManager.default.removeItem(at:f.root)}
            if legacy {
                f.owner.close()
                for name in ["ownership.json","owner.lock","activity.lock"] {try FileManager.default.removeItem(at:f.directory.appendingPathComponent(name))}
            } else {
                let run=UUID();try f.store.begin(ordinal:1,runID:run,directoryID:UUID())
                try Data("{".utf8).write(to:f.directory.appendingPathComponent("phase-\(run.uuidString)-preparation.json"))
                f.owner.close()
            }
            let before=try snapshot(f.root)
            guard case .unavailable(let reason)=try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root) else {return XCTFail("Invalid recovery evidence must be unavailable")}
            XCTAssertFalse(reason.isEmpty)
            XCTAssertEqual(try snapshot(f.root),before)
            if !legacy {try assertReleased(f.directory)}
        }
    }
    @MainActor func testPrecancelledProbeThrowsCancellationWithoutWrites()async throws {
        let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
        defer {try? FileManager.default.removeItem(at:f.root)}
        f.owner.close();let before=try snapshot(f.root)
        let task=Task { @MainActor in
            withUnsafeCurrentTask {$0?.cancel()}
            return try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root)
        }
        do {_=try await task.value;XCTFail("Cancellation must propagate")}
        catch {XCTAssertTrue(error is CancellationError)}
        XCTAssertEqual(try snapshot(f.root),before)
        try assertReleased(f.directory)
    }
}
