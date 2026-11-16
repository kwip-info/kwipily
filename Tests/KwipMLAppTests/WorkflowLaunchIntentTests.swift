import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowLaunchIntentTests:XCTestCase {
    private struct Fixture {
        let root:URL,group:URL,project:String,groupID:String,run:String,directoryID:String,owner:WorkflowGroupOwner
        var request:String {String(repeating:"a",count:64)}
        var arguments:[String] {
            let phase=root.appendingPathComponent("workflow-runs/\(project)/\(directoryID)/preparation")
            return ["--execute-dataflow",phase.appendingPathComponent("request.json").path,"--source-bindings",phase.appendingPathComponent("bindings.json").path,"--output",phase.appendingPathComponent("artifacts").path]
        }
        func ledger(stopped:Bool=false,runID:String?=nil)throws {
            let record:[String:Any] = ["schemaVersion":1,"groupID":groupID,"projectID":project,"train":["domain":"control","localID":"train"],"sourceRequestIdentity":request,"inputsManifestSHA256":request,"authoringSHA256":request,"revision":1,"stopRequested":stopped,"variants":[["ordinal":1,"requestIdentity":request,"status":"running","runID":runID ?? run,"directoryID":directoryID]]]
            try JSONSerialization.data(withJSONObject:record,options:.sortedKeys).write(to:group.appendingPathComponent("group.json"))
        }
        func publish(arguments:[String]?=nil,nonce:String=UUID().uuidString)throws->WorkflowPublishedLaunch {
            try owner.publishLaunch(projectID:project,ordinal:1,runID:run,directoryID:directoryID,requestIdentity:request,phase:.preparation,arguments:arguments ?? self.arguments,nonce:nonce)
        }
    }
    private func fixture()throws->Fixture {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        let project=UUID().uuidString,groupID=UUID().uuidString,run=UUID().uuidString,directoryID=UUID().uuidString
        let group=root.appendingPathComponent("workflow-experiments/\(project)/\(groupID)")
        try FileManager.default.createDirectory(at:group,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        for phase in ["preparation","training"] {
            let path=root.appendingPathComponent("workflow-runs/\(project)/\(directoryID)/\(phase)")
            try FileManager.default.createDirectory(at:path,withIntermediateDirectories:true)
            try Data("{}".utf8).write(to:path.appendingPathComponent("request.json"))
            if phase == "preparation" {try Data("{}".utf8).write(to:path.appendingPathComponent("bindings.json"))}
        }
        var filesystem=statfs()
        guard statfs(group.path,&filesystem) == 0 else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        let type=withUnsafePointer(to:&filesystem.f_fstypename) {$0.withMemoryRebound(to:CChar.self,capacity:16) {String(cString:$0)}}
        guard type == "apfs",filesystem.f_flags & UInt32(MNT_LOCAL) != 0 else {throw XCTSkip("Launch ownership requires local APFS.")}
        let owner=try WorkflowGroupOwnership.initialize(directory:group,groupID:groupID)
        let result=Fixture(root:root,group:group,project:project,groupID:groupID,run:run,directoryID:directoryID,owner:owner)
        try result.ledger();return result
    }
    func testBudgetIntentSchemaRoundTripAndMalformedPoliciesReject() throws {
        let f = try fixture()
        let budget = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        let published = try f.owner.publishLaunch(projectID: f.project, ordinal: 1, runID: f.run,
            directoryID: f.directoryID, requestIdentity: f.request, phase: .preparation,
            arguments: f.arguments, executionBudget: budget, remainingSeconds: 45.5)
        XCTAssertEqual(published.intent.schemaVersion, 2)
        XCTAssertEqual(published.intent.executionBudget, budget)
        XCTAssertEqual(published.intent.remainingSeconds, 45.5)
        let bytes = try Data(contentsOf: f.group.appendingPathComponent("launch-\(published.nonce).json"))
        XCTAssertEqual(try WorkflowLaunchIntent.decode(bytes), published.intent)
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0["schemaVersion"] = 1 }, { $0["schemaVersion"] = 3 },
            { $0.removeValue(forKey: "executionBudget") }, { $0.removeValue(forKey: "remainingSeconds") },
            { $0["remainingSeconds"] = NSNull() }, { $0["remainingSeconds"] = true },
            { $0["remainingSeconds"] = 0 }, { $0["remainingSeconds"] = 61 },
            { var policy = $0["executionBudget"] as! [String: Any]; policy["capabilityMaximumSeconds"] = NSNull(); $0["executionBudget"] = policy }
        ]
        for mutate in mutations {
            var object = original; mutate(&object)
            XCTAssertThrowsError(try WorkflowLaunchIntent.decode(JSONSerialization.data(withJSONObject: object)))
        }
    }
    func testPublicationRejectsUnpairedOrNonfiniteBudgetBeforeWriting() throws {
        let f = try fixture()
        let budget = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        for (policy, remaining) in [(budget, nil), (nil, 10.0), (budget, .infinity), (budget, .nan), (budget, 0), (budget, 61)] as [(ResolvedExecutionBudget?, Double?)] {
            XCTAssertThrowsError(try f.owner.publishLaunch(projectID: f.project, ordinal: 1, runID: f.run,
                directoryID: f.directoryID, requestIdentity: f.request, phase: .preparation,
                arguments: f.arguments, executionBudget: policy, remainingSeconds: remaining))
        }
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: f.group.path).contains { $0.hasPrefix("phase-") || $0.hasPrefix("launch-") })
    }
    func testTrainingCannotResetBudgetOrChangeSchemaOrPolicy() throws {
        let budget = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        let other = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 59)
        for legacy in [false, true] {
            let f = try fixture()
            let prep = try f.owner.publishLaunch(projectID: f.project, ordinal: 1, runID: f.run,
                directoryID: f.directoryID, requestIdentity: f.request, phase: .preparation,
                arguments: f.arguments, executionBudget: legacy ? nil : budget, remainingSeconds: legacy ? nil : 30)
            try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "nonce": prep.nonce, "intentSHA256": prep.intentSHA256, "state": "claimed"])
                .write(to: f.group.appendingPathComponent("claim-\(prep.nonce).json"))
            let phase = f.root.appendingPathComponent("workflow-runs/\(f.project)/\(f.directoryID)/training")
            let args = ["--graph", phase.appendingPathComponent("request.json").path, "--output", phase.appendingPathComponent("artifacts").path]
            let rejected: [(ResolvedExecutionBudget?, Double?)] = legacy ? [(budget, 20)] : [(nil, nil), (budget, 31), (other, 20)]
            for (policy, remaining) in rejected {
                XCTAssertThrowsError(try f.owner.publishLaunch(projectID: f.project, ordinal: 1, runID: f.run,
                    directoryID: f.directoryID, requestIdentity: f.request, phase: .training,
                    arguments: args, executionBudget: policy, remainingSeconds: remaining))
            }
            let training = try f.owner.publishLaunch(projectID: f.project, ordinal: 1, runID: f.run,
                directoryID: f.directoryID, requestIdentity: f.request, phase: .training,
                arguments: args, executionBudget: legacy ? nil : budget, remainingSeconds: legacy ? nil : 20)
            XCTAssertEqual(training.intent.schemaVersion, legacy ? 1 : 2)
        }
    }
    func testBudgetIntentCanBeRevokedByRecovery() throws {
        let f = try fixture()
        let budget = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        let published = try f.owner.publishLaunch(projectID: f.project, ordinal: 1, runID: f.run,
            directoryID: f.directoryID, requestIdentity: f.request, phase: .preparation,
            arguments: f.arguments, executionBudget: budget, remainingSeconds: 30)
        f.owner.close()
        let recovery = try WorkflowGroupOwnership.acquireRecovery(directory: f.group, groupID: f.groupID)
        defer { recovery.close() }
        try recovery.revokeLaunch(nonce: published.nonce, intentSHA256: published.intentSHA256)
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce: published.nonce, intentSHA256: published.intentSHA256))
    }
    func testPreparationRetentionFlagIsSealedAndCannotRepeat() throws {
        let f = try fixture()
        XCTAssertThrowsError(try f.publish(arguments: f.arguments + ["--retain-partial-dataflow", "--retain-partial-dataflow"]))
        let published = try f.publish(arguments: f.arguments + ["--retain-partial-dataflow"])
        XCTAssertEqual(published.intent.arguments.last, "--retain-partial-dataflow")
    }
    func testExactPublicationAndDuplicateAreImmutable()throws {
        let f=try fixture(),nonce=UUID().uuidString,published=try f.publish(nonce:nonce)
        XCTAssertEqual(published.nonce,nonce);XCTAssertEqual(published.groupDirectory,f.group)
        XCTAssertEqual(published.intent.arguments,f.arguments);XCTAssertEqual(published.intent.generation,f.owner.metadata.generation)
        let path=f.group.appendingPathComponent("launch-\(nonce).json"),bytes=try Data(contentsOf:path)
        XCTAssertEqual(published.intentSHA256,WorkflowPhaseArtifacts.hash(bytes))
        XCTAssertEqual(published.ownershipSHA256,WorkflowPhaseArtifacts.hash(try Data(contentsOf:f.group.appendingPathComponent("ownership.json"))))
        XCTAssertEqual(try Data(contentsOf:f.group.appendingPathComponent("phase-\(f.run)-preparation.json")),bytes)
        XCTAssertThrowsError(try f.publish())
        XCTAssertThrowsError(try f.publish(nonce:nonce))
        XCTAssertEqual(try Data(contentsOf:path),bytes)
    }
    func testLedgerAndClosedAuthorityRejectBeforePublication()throws {
        let f=try fixture()
        try f.ledger(stopped:true);XCTAssertThrowsError(try f.publish())
        try f.ledger(runID:UUID().uuidString);XCTAssertThrowsError(try f.publish())
        try f.ledger();f.owner.close();XCTAssertThrowsError(try f.publish())
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:f.group.path).contains {$0.hasPrefix("launch-")})
    }
    func testBoundedPhaseArgumentsAndExactPaths()throws {
        let f=try fixture()
        for args in [f.arguments+["--out","elsewhere"],f.arguments+["--unknown","x"],f.arguments+["--kwip-test"],f.arguments+["--output","elsewhere"],f.arguments+["--graph","x"],f.arguments+["--output=x"],f.arguments+[""],f.arguments+["x\0y"],f.arguments+[String(repeating:"a",count:65537)],f.arguments+Array(repeating:"x",count:129)] {
            XCTAssertThrowsError(try f.publish(arguments:args))
        }
        var args=f.arguments;args[1]="/tmp/request.json"
        XCTAssertThrowsError(try f.publish(arguments:args))
        XCTAssertThrowsError(try f.publish(nonce:"../escape"))
        let phase=f.root.appendingPathComponent("workflow-runs/\(f.project)/\(f.directoryID)/training")
        let training=["--graph",phase.appendingPathComponent("request.json").path,"--output",phase.appendingPathComponent("artifacts").path]
        XCTAssertThrowsError(try f.owner.publishLaunch(projectID:f.project,ordinal:1,runID:f.run,directoryID:f.directoryID,requestIdentity:f.request,phase:.training,arguments:training))
        let prep=try f.publish()
        XCTAssertThrowsError(try f.owner.publishLaunch(projectID:f.project,ordinal:1,runID:f.run,directoryID:f.directoryID,requestIdentity:f.request,phase:.training,arguments:training))
        try JSONSerialization.data(withJSONObject:["schemaVersion":1,"nonce":prep.nonce,"intentSHA256":prep.intentSHA256,"state":"claimed"]).write(to:f.group.appendingPathComponent("claim-\(prep.nonce).json"))
        let activity=try WorkflowFileLease(directory:f.group,name:"activity.lock",mode:.shared)
        XCTAssertThrowsError(try f.owner.publishLaunch(projectID:f.project,ordinal:1,runID:f.run,directoryID:f.directoryID,requestIdentity:f.request,phase:.training,arguments:training))
        activity.close()
        XCTAssertNoThrow(try f.owner.publishLaunch(projectID:f.project,ordinal:1,runID:f.run,directoryID:f.directoryID,requestIdentity:f.request,phase:.training,arguments:training))
    }
    func testExtraIntentFieldsRejectPriorAndRecoveryAuthority()throws {
        let f=try fixture(),published=try f.publish()
        let launch=f.group.appendingPathComponent("launch-\(published.nonce).json"),slot=f.group.appendingPathComponent("phase-\(f.run)-preparation.json")
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:launch)) as? [String:Any])
        object["extraAuthority"]="ignored?"
        let altered=try JSONSerialization.data(withJSONObject:object,options:.sortedKeys),hash=WorkflowPhaseArtifacts.hash(altered)
        try altered.write(to:launch);try altered.write(to:slot)
        try JSONSerialization.data(withJSONObject:["schemaVersion":1,"nonce":published.nonce,"intentSHA256":hash,"state":"claimed"]).write(to:f.group.appendingPathComponent("claim-\(published.nonce).json"))
        let phase=f.root.appendingPathComponent("workflow-runs/\(f.project)/\(f.directoryID)/training")
        XCTAssertThrowsError(try f.owner.publishLaunch(projectID:f.project,ordinal:1,runID:f.run,directoryID:f.directoryID,requestIdentity:f.request,phase:.training,arguments:["--graph",phase.appendingPathComponent("request.json").path,"--output",phase.appendingPathComponent("artifacts").path]))
        try FileManager.default.removeItem(at:f.group.appendingPathComponent("claim-\(published.nonce).json"))
        f.owner.close()
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:f.group,groupID:f.groupID)
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:hash))
    }
    func testDuplicateLedgerOrdinalsRejectPublication()throws {
        let f=try fixture(),path=f.group.appendingPathComponent("group.json")
        var record=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:path)) as? [String:Any])
        var variants=try XCTUnwrap(record["variants"] as? [[String:Any]])
        variants.append(["ordinal":1,"requestIdentity":String(repeating:"b",count:64),"status":"pending"])
        record["variants"]=variants
        try JSONSerialization.data(withJSONObject:record).write(to:path)
        XCTAssertThrowsError(try f.publish())
    }
    func testPartialPhaseSlotIsNeverReplaced()throws {
        let f=try fixture(),path=f.group.appendingPathComponent("phase-\(f.run)-preparation.json"),partial=Data("{".utf8)
        try partial.write(to:path)
        XCTAssertThrowsError(try f.publish())
        XCTAssertEqual(try Data(contentsOf:path),partial)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:f.group.path).contains {$0.hasPrefix("launch-")})
    }
    func testRecoveryRevocationAndExistingClaimPreserved()throws {
        let f=try fixture(),published=try f.publish()
        f.owner.close()
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:f.group,groupID:f.groupID)
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:String(repeating:"b",count:64)))
        try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:published.intentSHA256)
        let path=f.group.appendingPathComponent("claim-\(published.nonce).json"),bytes=try Data(contentsOf:path)
        let record=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        XCTAssertEqual(record["state"] as? String,"revoked")
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:published.intentSHA256))
        XCTAssertEqual(try Data(contentsOf:path),bytes)
        recovery.close();XCTAssertThrowsError(try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:published.intentSHA256))
    }
    func testPythonConsumesExactSwiftPublishedIntent()throws {
        let root=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"].map {URL(fileURLWithPath:$0)} ?? URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath:python.path) else {throw XCTSkip("Development Python unavailable for cross-language launch protocol test.")}
        let f=try fixture(),published=try f.publish(),process=Process(),output=Pipe(),errors=Pipe()
        process.executableURL=python
        process.arguments=["-c","import sys; sys.path.insert(0,sys.argv[1]); from worker_authorization import claim_launch; a=claim_launch(*sys.argv[2:]); print(a.intent['nonce']); a.check(); a.close()",root.appendingPathComponent("runner").path,f.group.path,published.nonce,published.intentSHA256,published.ownershipSHA256]
        process.standardOutput=output;process.standardError=errors
        try process.run()
        let deadline=ProcessInfo.processInfo.systemUptime+10
        while process.isRunning && ProcessInfo.processInfo.systemUptime < deadline {Thread.sleep(forTimeInterval:0.01)}
        if process.isRunning {
            process.terminate()
            let grace=ProcessInfo.processInfo.systemUptime+1
            while process.isRunning && ProcessInfo.processInfo.systemUptime < grace {Thread.sleep(forTimeInterval:0.01)}
            if process.isRunning {_=Darwin.kill(process.processIdentifier,SIGKILL)}
            XCTFail("Independent launch bootstrap exceeded its deadline.")
        }
        let exitDeadline=ProcessInfo.processInfo.systemUptime+2
        while process.isRunning && ProcessInfo.processInfo.systemUptime < exitDeadline {Thread.sleep(forTimeInterval:0.01)}
        guard !process.isRunning else {throw NSError(domain:"WorkflowLaunchIntentTests",code:1,userInfo:[NSLocalizedDescriptionKey:"Owned bootstrap did not exit after termination."])}
        let stdout=String(decoding:output.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self),stderr=String(decoding:errors.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self)
        XCTAssertEqual(process.terminationStatus,0,stderr)
        XCTAssertEqual(stdout.trimmingCharacters(in:.whitespacesAndNewlines),published.nonce)
        let claim=f.group.appendingPathComponent("claim-\(published.nonce).json"),bytes=try Data(contentsOf:claim)
        f.owner.close()
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:f.group,groupID:f.groupID)
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:published.intentSHA256))
        XCTAssertEqual(try Data(contentsOf:claim),bytes)
    }
    func testReplacedGroupAndPreexistingPartialClaimFailClosed()throws {
        let f=try fixture(),published=try f.publish()
        try Data("{".utf8).write(to:f.group.appendingPathComponent("claim-\(published.nonce).json"))
        f.owner.close()
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:f.group,groupID:f.groupID)
        XCTAssertThrowsError(try recovery.revokeLaunch(nonce:published.nonce,intentSHA256:published.intentSHA256))
        recovery.close()
        let other=try fixture(),moved=other.group.appendingPathExtension("old")
        try FileManager.default.moveItem(at:other.group,to:moved)
        try FileManager.default.createDirectory(at:other.group,withIntermediateDirectories:false)
        XCTAssertThrowsError(try other.publish())
    }
}
