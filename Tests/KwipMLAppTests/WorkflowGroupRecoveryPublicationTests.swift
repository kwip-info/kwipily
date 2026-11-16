import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowGroupRecoveryPublicationTests:XCTestCase {
    private func fixture()throws->(URL,WorkflowGroupOwner) {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let directory=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let owner=try WorkflowGroupOwnership.initialize(directory:directory,groupID:directory.lastPathComponent)
        addTeardownBlock {owner.close()}
        return (directory,owner)
    }
    private func recover(_ directory:URL)throws->WorkflowGroupRecovery {
        try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:directory.lastPathComponent)
    }
    func testRecoveryPublishesExactBytesAndRejectsStaleOrClosedAuthority()throws {
        let (directory,owner)=try fixture(),first=Data("first".utf8),second=Data("second".utf8)
        try owner.publishGroupRecord(first)
        XCTAssertThrowsError(try recover(directory))
        owner.close()
        let recovery=try recover(directory)
        defer {recovery.close()}
        try recovery.validateDirectory(directory)
        XCTAssertThrowsError(try recovery.validateDirectory(directory.appendingPathComponent("other")))
        XCTAssertEqual(try recovery.readGroupRecord(),first)
        try recovery.replaceGroupRecord(expected:first,data:second)
        XCTAssertEqual(try recovery.readGroupRecord(),second)
        XCTAssertThrowsError(try recovery.replaceGroupRecord(expected:first,data:Data()))
        XCTAssertThrowsError(try recovery.replaceGroupRecord(expected:second,data:Data(repeating:0,count:256_001)))
        XCTAssertEqual(try recovery.readGroupRecord(),second)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:directory.path).contains {$0.hasPrefix(".ownership-")})
        recovery.close()
        XCTAssertThrowsError(try recovery.validateDirectory(directory))
        XCTAssertThrowsError(try recovery.readGroupRecord())
        XCTAssertThrowsError(try recovery.replaceGroupRecord(expected:second,data:Data()))
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("group.json")),second)
    }
    func testRecoveryInspectionIsAnchoredBoundedAndRejectsClosedAuthority()throws {
        let (directory,owner)=try fixture(),nonce=UUID().uuidString,run=UUID().uuidString
        owner.close()
        let recovery=try recover(directory)
        defer {recovery.close()}
        XCTAssertNil(try recovery.readGroupRecord())
        XCTAssertNil(try recovery.readPhaseSlot(runID:run,phase:"preparation"))
        XCTAssertNil(try recovery.readLaunch(nonce:nonce))
        XCTAssertNil(try recovery.readLaunchClaim(nonce:nonce))
        let files=[("phase-\(run)-preparation.json",Data("slot".utf8)),("launch-\(nonce).json",Data("intent".utf8)),("claim-\(nonce).json",Data("claim".utf8))]
        for (name,bytes) in files {try bytes.write(to:directory.appendingPathComponent(name))}
        XCTAssertEqual(try recovery.readPhaseSlot(runID:run,phase:"preparation"),files[0].1)
        XCTAssertEqual(try recovery.readLaunch(nonce:nonce),files[1].1)
        XCTAssertEqual(try recovery.readLaunchClaim(nonce:nonce),files[2].1)
        XCTAssertThrowsError(try recovery.readPhaseSlot(runID:run,phase:"../training"))
        XCTAssertThrowsError(try recovery.readLaunch(nonce:"../outside"))
        try Data(repeating:0,count:4097).write(to:directory.appendingPathComponent(files[2].0))
        XCTAssertThrowsError(try recovery.readLaunchClaim(nonce:nonce))
        recovery.close()
        XCTAssertThrowsError(try recovery.readPhaseSlot(runID:run,phase:"preparation"))
        XCTAssertThrowsError(try recovery.readLaunch(nonce:nonce))
        XCTAssertThrowsError(try recovery.readLaunchClaim(nonce:nonce))
    }
    func testRecoveryRefusesReplacedLockOrRedirectedLedger()throws {
        let (directory,owner)=try fixture(),bytes=Data("original".utf8)
        try owner.publishGroupRecord(bytes);owner.close()
        let recovery=try recover(directory)
        defer {recovery.close()}
        let ledger=directory.appendingPathComponent("group.json"),outside=directory.appendingPathComponent("sentinel")
        try bytes.write(to:outside)
        try FileManager.default.removeItem(at:ledger)
        try FileManager.default.createSymbolicLink(at:ledger,withDestinationURL:outside)
        XCTAssertThrowsError(try recovery.readGroupRecord())
        XCTAssertThrowsError(try recovery.replaceGroupRecord(expected:bytes,data:Data()))
        XCTAssertEqual(try Data(contentsOf:outside),bytes)
        try FileManager.default.moveItem(at:directory.appendingPathComponent("activity.lock"),to:directory.appendingPathComponent("old-activity"))
        try Data().write(to:directory.appendingPathComponent("activity.lock"))
        XCTAssertThrowsError(try recovery.validateDirectory(directory))
    }
    func testRealWorkerActivityPreventsRecoveryUntilProcessExit()throws {
        let (directory,owner)=try fixture();try owner.publishGroupRecord(Data("unchanged".utf8));owner.close()
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath:python.path) else {throw XCTSkip("Repository Python environment is required")}
        let process=Process(),input=Pipe(),output=Pipe(),ended=expectation(description:"owned activity child exited")
        process.executableURL=python
        process.arguments=["-c","import fcntl,os,signal,sys; signal.alarm(8); f=os.open(sys.argv[1],os.O_RDONLY); fcntl.flock(f,fcntl.LOCK_SH); print('READY',flush=True); sys.stdin.buffer.read(1)",directory.appendingPathComponent("activity.lock").path]
        process.standardInput=input;process.standardOutput=output
        process.terminationHandler={_ in ended.fulfill()}
        try process.run()
        defer {try? input.fileHandleForWriting.close();if process.isRunning {process.terminate()}}
        XCTAssertEqual(try output.fileHandleForReading.read(upToCount:6),Data("READY\n".utf8))
        XCTAssertThrowsError(try recover(directory))
        try input.fileHandleForWriting.close()
        wait(for:[ended],timeout:10)
        XCTAssertFalse(process.isRunning)
        let recovery=try recover(directory);defer {recovery.close()}
        XCTAssertEqual(try recovery.readGroupRecord(),Data("unchanged".utf8))
    }
    func testRepeatedRevocationPreservesImmutableClaim()throws {
        let (workspace,unusedOwner)=try fixture(); unusedOwner.close()
        let project=UUID().uuidString,nonce=UUID().uuidString,run=UUID().uuidString,runDirectory=UUID().uuidString
        let group=try WorkflowGroupOwnership.createGroup(workspace:workspace,projectID:project)
        let directory=group.directory,owner=group.owner
        defer { owner.close() }
        let phase=workspace.appendingPathComponent("workflow-runs/\(project)/\(runDirectory)/preparation")
        let arguments=["--execute-dataflow",phase.appendingPathComponent("request.json").path,"--source-bindings",phase.appendingPathComponent("bindings.json").path,"--output",phase.appendingPathComponent("artifacts").path]
        let intent=WorkflowLaunchIntent(schemaVersion:1,projectID:project,groupID:owner.metadata.groupID,generation:owner.metadata.generation,runID:run,directoryID:runDirectory,nonce:nonce,ordinal:1,requestIdentity:String(repeating:"a",count:64),ownershipSHA256:WorkflowPhaseArtifacts.hash(try Data(contentsOf:directory.appendingPathComponent("ownership.json"))),phase:.preparation,arguments:arguments)
        let bytes=try JSONEncoder().encode(intent),hash=WorkflowPhaseArtifacts.hash(bytes)
        try bytes.write(to:directory.appendingPathComponent("launch-\(nonce).json"))
        try bytes.write(to:directory.appendingPathComponent("phase-\(run)-preparation.json"))
        owner.close()
        let recovery=try recover(directory);defer {recovery.close()}
        try recovery.revokeLaunch(nonce:nonce,intentSHA256:hash)
        let claim=try XCTUnwrap(recovery.readLaunchClaim(nonce:nonce))
        XCTAssertEqual(try JSONDecoder().decode(WorkflowLaunchClaim.self,from:claim).state,"revoked")
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce:nonce,intentSHA256:hash))
        XCTAssertEqual(try recovery.readLaunchClaim(nonce:nonce),claim)
        XCTAssertEqual(try recovery.readLaunch(nonce:nonce),bytes)
    }
    func testPhaseRevocationHandlesSlotOnlyPublicationAndPreservesClaims()throws {
        for scenario in ["slotOnly","complete","differentIntent","claimed","wrongDigest","wrongGeneration","wrongPhase"] {
            let (workspace,unusedOwner)=try fixture();unusedOwner.close()
            let project=UUID().uuidString
            let created=try WorkflowGroupOwnership.createGroup(workspace:workspace,projectID:project)
            let directory=created.directory,owner=created.owner,run=UUID().uuidString,runDirectory=UUID().uuidString,nonce=UUID().uuidString
            defer {owner.close()}
            let phaseRoot=workspace.appendingPathComponent("workflow-runs/\(project)/\(runDirectory)/preparation")
            let arguments=["--execute-dataflow",phaseRoot.appendingPathComponent("request.json").path,"--source-bindings",phaseRoot.appendingPathComponent("bindings.json").path,"--output",phaseRoot.appendingPathComponent("artifacts").path]
            let intent=WorkflowLaunchIntent(schemaVersion:1,projectID:project,groupID:owner.metadata.groupID,
                generation:scenario == "wrongGeneration" ? UUID().uuidString:owner.metadata.generation,
                runID:run,directoryID:runDirectory,nonce:nonce,ordinal:1,requestIdentity:String(repeating:"a",count:64),
                ownershipSHA256:WorkflowPhaseArtifacts.hash(try Data(contentsOf:directory.appendingPathComponent("ownership.json"))),
                phase:.preparation,arguments:arguments)
            let bytes=try JSONEncoder().encode(intent),hash=WorkflowPhaseArtifacts.hash(bytes)
            let slot=directory.appendingPathComponent("phase-\(run)-preparation.json"),launch=directory.appendingPathComponent("launch-\(nonce).json"),claimURL=directory.appendingPathComponent("claim-\(nonce).json")
            try bytes.write(to:slot)
            if scenario == "complete" {try bytes.write(to:launch)}
            if scenario == "differentIntent" {try Data("different".utf8).write(to:launch)}
            var originalClaim:Data?
            if scenario == "claimed" {
                originalClaim=try JSONEncoder().encode(WorkflowLaunchClaim(schemaVersion:1,nonce:nonce,intentSHA256:hash,state:"claimed"))
                try originalClaim!.write(to:claimURL)
            }
            owner.close()
            let recovery=try recover(directory);defer {recovery.close()}
            if ["slotOnly","complete"].contains(scenario) {
                try recovery.revokePhase(runID:run,phase:"preparation",intentSHA256:hash)
                let claim=try XCTUnwrap(recovery.readLaunchClaim(nonce:nonce))
                XCTAssertEqual(try JSONDecoder().decode(WorkflowLaunchClaim.self,from:claim),WorkflowLaunchClaim(schemaVersion:1,nonce:nonce,intentSHA256:hash,state:"revoked"))
                XCTAssertThrowsError(try recovery.revokePhase(runID:run,phase:"preparation",intentSHA256:hash))
                XCTAssertEqual(try recovery.readLaunchClaim(nonce:nonce),claim)
                XCTAssertEqual(FileManager.default.fileExists(atPath:launch.path),scenario == "complete")
            } else {
                XCTAssertThrowsError(try recovery.revokePhase(runID:run,phase:scenario == "wrongPhase" ? "training":"preparation",intentSHA256:scenario == "wrongDigest" ? String(repeating:"b",count:64):hash))
                XCTAssertEqual(try recovery.readLaunchClaim(nonce:nonce),originalClaim)
            }
            XCTAssertEqual(try Data(contentsOf:slot),bytes)
        }
    }

}
