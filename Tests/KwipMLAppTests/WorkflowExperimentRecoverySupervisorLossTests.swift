import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Real claim/supervisor/native-guard/recovery integration with a controlled
/// worker. This deliberately does not claim arbitrary ML phase coverage.
final class WorkflowExperimentRecoverySupervisorLossTests:XCTestCase {
    @MainActor func testStoppedWorkerRetainsAuthorityAfterSupervisorLossUntilGuardExit()throws {
        let repo=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"].map {URL(fileURLWithPath:$0)} ?? URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=repo.appendingPathComponent(".venv/bin/python")
        let watchdog=repo.appendingPathComponent(".local/native-watchdog/lifetime_watchdog.dylib")
        guard FileManager.default.isExecutableFile(atPath:python.path),FileManager.default.fileExists(atPath:watchdog.path) else {
            throw XCTSkip("Requires development Python and the staged native lifetime watchdog.")
        }
        let f=try WorkflowExperimentRecoveryCoordinatorTests().fixture()
        let process=Process(),input=Pipe(),output=Pipe()
        defer {
            try? input.fileHandleForWriting.close()
            let deadline=ProcessInfo.processInfo.systemUptime+4
            while process.isRunning && ProcessInfo.processInfo.systemUptime<deadline {Thread.sleep(forTimeInterval:0.01)}
            XCTAssertFalse(process.isRunning,"Broker must acknowledge EOF by cleaning its owned process group; its own alarm is the final bound.")
            f.owner.close()
            if !process.isRunning {try? FileManager.default.removeItem(at:f.root)}
        }
        let run=UUID(),directoryID=UUID(),project=f.store.record.projectID
        try f.store.begin(ordinal:1,runID:run,directoryID:directoryID)
        let phase=f.root.appendingPathComponent("workflow-runs/\(project)/\(directoryID.uuidString)/preparation")
        try FileManager.default.createDirectory(at:phase,withIntermediateDirectories:true)
        // Authorization checks regular exact-path inputs; no engine is invoked.
        try Data("{}".utf8).write(to:phase.appendingPathComponent("request.json"))
        try Data("{}".utf8).write(to:phase.appendingPathComponent("bindings.json"))
        let launch=try f.owner.publishLaunch(projectID:project,ordinal:1,runID:run.uuidString,directoryID:directoryID.uuidString,
            requestIdentity:f.requests[0].requestIdentity,phase:.preparation,arguments:["--execute-dataflow",phase.appendingPathComponent("request.json").path,"--source-bindings",phase.appendingPathComponent("bindings.json").path,"--output",phase.appendingPathComponent("artifacts").path])
        process.executableURL=python
        process.arguments=["-I",repo.appendingPathComponent("runner/tests/fixtures/recovery_supervisor_loss.py").path,"broker",repo.appendingPathComponent("runner").path,watchdog.path,f.directory.path,launch.nonce,launch.intentSHA256,launch.ownershipSHA256]
        process.standardInput=input;process.standardOutput=output
        // Inherit stderr: failures stay visible without an undrained error pipe.
        try process.run()
        try input.fileHandleForReading.close();try output.fileHandleForWriting.close()
        let stopped=try line(output.fileHandleForReading)
        guard stopped == "WORKER_STOPPED" else {
            XCTFail("Missing stopped-worker handshake: \(stopped)")
            return
        }
        let ledger=f.directory.appendingPathComponent("group.json")
        let claim=f.directory.appendingPathComponent("claim-\(launch.nonce).json")
        let before=try Data(contentsOf:ledger),claimed=try Data(contentsOf:claim)
        let requestBytes=try Data(contentsOf:phase.appendingPathComponent("request.json"))
        f.owner.close()
        try command("KILL_SUPERVISOR",input.fileHandleForWriting)
        XCTAssertEqual(try line(output.fileHandleForReading),"SUPERVISOR_EXITED")
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root),.inUse)
        XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
        XCTAssertEqual(try Data(contentsOf:ledger),before)
        XCTAssertEqual(try Data(contentsOf:claim),claimed)
        XCTAssertFalse(FileManager.default.fileExists(atPath:phase.appendingPathComponent("artifacts").path))
        try command("RESUME",input.fileHandleForWriting)
        XCTAssertEqual(try line(output.fileHandleForReading),"WORKER_RELEASED")
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.probe(directory:f.directory,workspaceDirectory:f.root),.recoverable)
        let recovered=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(recovered.variants.map(\.status),[.interrupted,.cancelled])
        XCTAssertNil(recovered.variants[0].outcomeSHA256)
        XCTAssertNotNil(recovered.recoveryReceipt)
        XCTAssertEqual(try Data(contentsOf:claim),claimed)
        XCTAssertEqual(try Data(contentsOf:phase.appendingPathComponent("request.json")),requestBytes)
        let final=try Data(contentsOf:ledger)
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root),recovered)
        XCTAssertEqual(try Data(contentsOf:ledger),final)
        try command("CLEANUP",input.fileHandleForWriting)
    }
    private func command(_ value:String,_ handle:FileHandle)throws {
        try handle.write(contentsOf:Data((value+"\n").utf8))
    }
    private func line(_ handle:FileHandle)throws->String {
        let deadline=ProcessInfo.processInfo.systemUptime+12
        var bytes:[UInt8]=[]
        while ProcessInfo.processInfo.systemUptime<deadline && bytes.count<4096 {
            var descriptor=pollfd(fd:handle.fileDescriptor,events:Int16(POLLIN),revents:0)
            let ready=Darwin.poll(&descriptor,1,100)
            if ready<0 {if errno == EINTR {continue};throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
            if ready == 0 {continue}
            var byte:UInt8=0
            let count=Darwin.read(handle.fileDescriptor,&byte,1)
            guard count == 1 else {throw DataFlowContractError("Supervisor-loss fixture closed before its handshake.")}
            if byte == 10 {return String(decoding:bytes,as:UTF8.self)}
            bytes.append(byte)
        }
        throw DataFlowContractError("Supervisor-loss fixture exceeded its bounded handshake.")
    }
}
