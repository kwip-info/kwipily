import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentRecoveryCoordinatorTests:XCTestCase {
    struct Fixture {
        let root:URL
        let directory:URL
        let owner:WorkflowGroupOwner
        let store:WorkflowExperimentGroupStore
        let requests:[WorkflowExecutionRequest]
    }
    @MainActor func fixture()throws->Fixture {
        let helper=WorkflowExecutionRequestTests()
        let(root,authoring,runtime)=try helper.setup()
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(authoring.setExperimentSpecification(spec,for:.init(domain:.control,localID:"setup"),lease:authoring.lease()))
        let base=try helper.capture(authoring,runtime)
        let group=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:base.lease.projectID)
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:group.directory) {_ in Data("a,b,label\n1,2,0\n".utf8)}
        let requests=try base.experimentVariants(inputs:inputs,owner:authoring)
        let store=try WorkflowExperimentGroupStore.create(requests:requests,directory:group.directory,groupOwner:group.owner)
        return .init(root:group.directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),directory:group.directory,owner:group.owner,store:store,requests:requests)
    }
    @MainActor func testInactiveRunningAndPendingRecoverOnceWithoutInventingOutcomes()throws {
        let f=try fixture(),run=UUID(),directoryID=UUID()
        try f.store.begin(ordinal:1,runID:run,directoryID:directoryID)
        let before=f.store.record
        f.owner.close()
        let recovered=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(recovered.variants.map(\.status),[.interrupted,.cancelled])
        XCTAssertTrue(recovered.stopRequested)
        XCTAssertEqual(recovered.revision,before.revision+1)
        XCTAssertEqual(recovered.variants[0].runID,run.uuidString)
        XCTAssertNil(recovered.variants[0].outcomeSHA256)
        let bytes=try Data(contentsOf:f.directory.appendingPathComponent("group.json"))
        let again=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(again,recovered)
        XCTAssertEqual(try Data(contentsOf:f.directory.appendingPathComponent("group.json")),bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath:f.root.appendingPathComponent("workflow-runs").path))
    }
    @MainActor func testLiveOwnerAndWorkerPreventAnyRecoveryMutation()throws {
        let f=try fixture(),path=f.directory.appendingPathComponent("group.json"),before=try Data(contentsOf:path)
        XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
        XCTAssertEqual(try Data(contentsOf:path),before)
        let activity=try WorkflowFileLease(directory:f.directory,name:"activity.lock",mode:.shared)
        f.owner.close()
        XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
        XCTAssertEqual(try Data(contentsOf:path),before)
        activity.close()
        let recovered=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(recovered.variants.map(\.status),[.cancelled,.cancelled])
    }
    @MainActor func testUnclaimedLaunchIsRevokedBeforeInterruptedLedgerAndRepeatIsStable()throws {
        let f=try fixture(),run=UUID(),directoryID=UUID()
        try f.store.begin(ordinal:1,runID:run,directoryID:directoryID)
        let phase=f.root.appendingPathComponent("workflow-runs/\(f.store.record.projectID)/\(directoryID.uuidString)/preparation")
        let launch=try f.owner.publishLaunch(projectID:f.store.record.projectID,ordinal:1,runID:run.uuidString,directoryID:directoryID.uuidString,requestIdentity:f.requests[0].requestIdentity,phase:.preparation,arguments:["--execute-dataflow",phase.appendingPathComponent("request.json").path,"--source-bindings",phase.appendingPathComponent("bindings.json").path,"--output",phase.appendingPathComponent("artifacts").path])
        f.owner.close()
        _=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        let claimPath=f.directory.appendingPathComponent("claim-\(launch.nonce).json"),bytes=try Data(contentsOf:claimPath)
        let claim=try JSONDecoder().decode(WorkflowLaunchClaim.self,from:bytes)
        XCTAssertEqual(claim.state,"revoked");XCTAssertEqual(claim.intentSHA256,launch.intentSHA256)
        _=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(try Data(contentsOf:claimPath),bytes)
    }
    @MainActor func testSlotOnlyPublicationCrashCanRecoverWithoutRecreatingLaunch()throws {
        let f=try fixture(),run=UUID(),directoryID=UUID()
        try f.store.begin(ordinal:1,runID:run,directoryID:directoryID)
        let phase=f.root.appendingPathComponent("workflow-runs/\(f.store.record.projectID)/\(directoryID.uuidString)/preparation")
        let launch=try f.owner.publishLaunch(projectID:f.store.record.projectID,ordinal:1,runID:run.uuidString,directoryID:directoryID.uuidString,requestIdentity:f.requests[0].requestIdentity,phase:.preparation,arguments:["--execute-dataflow",phase.appendingPathComponent("request.json").path,"--source-bindings",phase.appendingPathComponent("bindings.json").path,"--output",phase.appendingPathComponent("artifacts").path])
        let launchPath=f.directory.appendingPathComponent("launch-\(launch.nonce).json")
        try FileManager.default.removeItem(at:launchPath)
        f.owner.close()
        let result=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(result.variants.map(\.status),[.interrupted,.cancelled])
        XCTAssertFalse(FileManager.default.fileExists(atPath:launchPath.path))
        let claim=try JSONDecoder().decode(WorkflowLaunchClaim.self,from:Data(contentsOf:f.directory.appendingPathComponent("claim-\(launch.nonce).json")))
        XCTAssertEqual(claim.state,"revoked")
        XCTAssertEqual(claim.intentSHA256,launch.intentSHA256)
        XCTAssertNoThrow(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
    }
    @MainActor func testMalformedProtocolAndChangedAuthoringFailWithoutLedgerMutation()throws {
        for malformedSlot in [false,true] {
            let f=try fixture(),run=UUID(),directoryID=UUID()
            try f.store.begin(ordinal:1,runID:run,directoryID:directoryID)
            let path=f.directory.appendingPathComponent("group.json"),before=try Data(contentsOf:path)
            if malformedSlot {try Data("{".utf8).write(to:f.directory.appendingPathComponent("phase-\(run.uuidString)-preparation.json"))}
            else {try Data("{}".utf8).write(to:f.directory.appendingPathComponent("authoring.json"))}
            f.owner.close()
            XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
            XCTAssertEqual(try Data(contentsOf:path),before)
        }
    }
}
