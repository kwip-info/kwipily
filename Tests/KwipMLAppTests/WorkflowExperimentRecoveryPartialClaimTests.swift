import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentRecoveryPartialClaimTests:XCTestCase {
    @MainActor func testKilledClaimPublisherLeavesRecoveryUnavailableWithoutRepair()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath:python.path) else {throw XCTSkip("Repository Python environment is required")}
        let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
        defer {f.owner.close();try? FileManager.default.removeItem(at:f.root)}
        let run=UUID(),directoryID=UUID()
        try f.store.begin(ordinal:1,runID:run,directoryID:directoryID)
        let phase=f.root.appendingPathComponent("workflow-runs/\(f.store.record.projectID)/\(directoryID.uuidString)/preparation")
        let launch=try f.owner.publishLaunch(projectID:f.store.record.projectID,ordinal:1,runID:run.uuidString,directoryID:directoryID.uuidString,requestIdentity:f.requests[0].requestIdentity,phase:.preparation,arguments:["--execute-dataflow",phase.appendingPathComponent("request.json").path,"--source-bindings",phase.appendingPathComponent("bindings.json").path,"--output",phase.appendingPathComponent("artifacts").path])
        let claimPath=f.directory.appendingPathComponent("claim-\(launch.nonce).json")
        let partial=Data("{\"schemaVersion\":1,\"nonce\":".utf8)
        let process=Process(),output=Pipe(),ended=expectation(description:"Owned claim publisher exited")
        process.executableURL=python
        process.arguments=["-c", """
import fcntl, os, signal, sys
signal.alarm(8)
activity = os.open(sys.argv[1], os.O_RDONLY)
fcntl.flock(activity, fcntl.LOCK_SH)
claim = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(claim, b'{"schemaVersion":1,"nonce":')
os.fsync(claim)
print('READY', flush=True)
signal.pause()
""",f.directory.appendingPathComponent("activity.lock").path,claimPath.path]
        process.standardOutput=output
        process.terminationHandler={_ in ended.fulfill()}
        try process.run()
        defer {if process.isRunning {process.terminate()}}
        XCTAssertEqual(try output.fileHandleForReading.read(upToCount:6),Data("READY\n".utf8))
        XCTAssertEqual(try Data(contentsOf:claimPath),partial)
        f.owner.close()
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root),.inUse)
        // Signal only the still-owned live Process; the child is blocked after its
        // write barrier, so this models death before completing the claim record.
        guard process.isRunning else {return XCTFail("Publisher exited before simulated crash")}
        XCTAssertEqual(kill(process.processIdentifier,SIGKILL),0)
        wait(for:[ended],timeout:10)
        guard !process.isRunning else {return XCTFail("Publisher did not exit")}
        XCTAssertEqual(process.terminationReason,.uncaughtSignal)
        XCTAssertEqual(process.terminationStatus,SIGKILL)
        func snapshot()throws->[String:Data] {
            var files:[String:Data]=[:]
            guard let entries=FileManager.default.enumerator(at:f.root,includingPropertiesForKeys:[.isRegularFileKey]) else {throw DataFlowContractError("Cannot enumerate fixture")}
            for case let url as URL in entries {
                if try url.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true {files[url.path]=try Data(contentsOf:url)}
            }
            return files
        }
        let before=try snapshot()
        guard case .unavailable=try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root) else {return XCTFail("A partial claim must not become recoverable")}
        XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
        XCTAssertEqual(try snapshot(),before,"Recovery must preserve every ledger, captured input, and partial protocol byte")
        XCTAssertEqual(try Data(contentsOf:claimPath),partial)
        XCTAssertEqual(try WorkflowExperimentGroupStore.readRecord(directory:f.directory),f.store.record)
        XCTAssertFalse(FileManager.default.fileExists(atPath:f.root.appendingPathComponent("workflow-runs").path),"No execution or artifact repair was authorized")
    }
}
