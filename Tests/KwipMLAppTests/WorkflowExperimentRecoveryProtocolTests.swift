import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentRecoveryProtocolTests:XCTestCase {
    @MainActor private func fixture()throws->WorkflowExperimentRecoveryCoordinatorTests.Fixture {
        let helper=WorkflowExperimentRecoveryCoordinatorTests()
        let f=try helper.fixture()
        addTeardownBlock {f.owner.close();try? FileManager.default.removeItem(at:f.root)}
        try f.store.begin(ordinal:1,runID:UUID(),directoryID:UUID())
        return f
    }
    @MainActor private func publish(_ f:WorkflowExperimentRecoveryCoordinatorTests.Fixture,phase:WorkflowLaunchIntent.Phase)throws->WorkflowPublishedLaunch {
        let variant=f.store.record.variants[0]
        let path=f.root.appendingPathComponent("workflow-runs/\(f.store.record.projectID)/\(variant.directoryID!)/\(phase.rawValue)")
        let args=phase == .preparation ? ["--execute-dataflow",path.appendingPathComponent("request.json").path,"--source-bindings",path.appendingPathComponent("bindings.json").path,"--output",path.appendingPathComponent("artifacts").path]:["--graph",path.appendingPathComponent("request.json").path,"--output",path.appendingPathComponent("artifacts").path]
        return try f.owner.publishLaunch(projectID:f.store.record.projectID,ordinal:1,runID:variant.runID!,directoryID:variant.directoryID!,requestIdentity:variant.requestIdentity,phase:phase,arguments:args)
    }
    private func claim(_ launch:WorkflowPublishedLaunch,state:String="claimed",hash:String?=nil)throws->Data {
        try JSONEncoder().encode(WorkflowLaunchClaim(schemaVersion:1,nonce:launch.nonce,intentSHA256:hash ?? launch.intentSHA256,state:state))
    }
    @MainActor private func refuseUnchanged(_ f:WorkflowExperimentRecoveryCoordinatorTests.Fixture)throws {
        f.owner.close()
        let before=try snapshot(f.directory)
        XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root))
        XCTAssertEqual(try snapshot(f.directory),before,"Refusal must not publish queued revocations or alter ledger bytes")
    }
    private func snapshot(_ directory:URL)throws->[String:Data] {
        var result:[String:Data]=[:]
        for name in try FileManager.default.contentsOfDirectory(atPath:directory.path) where name == "group.json" || name.hasPrefix("claim-") || name.hasPrefix("phase-") || name.hasPrefix("launch-") {
            result[name]=try Data(contentsOf:directory.appendingPathComponent(name))
        }
        return result
    }
    @MainActor func testPartialAndMismatchedClaimsRefuseWithoutMutation()throws {
        for partial in [true,false] {
            let f=try fixture(),prep=try publish(f,phase:.preparation)
            let bytes:Data
            if partial {bytes=Data("{".utf8)}
            else {bytes=try claim(prep,hash:String(repeating:"f",count:64))}
            try bytes.write(to:f.directory.appendingPathComponent("claim-\(prep.nonce).json"))
            try refuseUnchanged(f)
        }
    }
    @MainActor func testTrainingRequiresClaimedPreparationBeforeAnyRevocation()throws {
        for condition in ["missing","unclaimed","revoked"] {
            let f=try fixture(),prep=try publish(f,phase:.preparation)
            let prepClaim=f.directory.appendingPathComponent("claim-\(prep.nonce).json")
            try claim(prep).write(to:prepClaim)
            _=try publish(f,phase:.training)
            if condition == "revoked" {try claim(prep,state:"revoked").write(to:prepClaim)}
            else {
                try FileManager.default.removeItem(at:prepClaim)
                if condition == "missing" {
                    try FileManager.default.removeItem(at:f.directory.appendingPathComponent("phase-\(prep.intent.runID)-preparation.json"))
                }
            }
            try refuseUnchanged(f)
        }
    }
    @MainActor func testClaimedPreparationIsPreservedWhileTrainingIsRevoked()throws {
        let f=try fixture(),prep=try publish(f,phase:.preparation),prepBytes=try claim(prep)
        let prepPath=f.directory.appendingPathComponent("claim-\(prep.nonce).json")
        try prepBytes.write(to:prepPath)
        let training=try publish(f,phase:.training)
        f.owner.close()
        let result=try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root)
        XCTAssertEqual(result.variants.map(\.status),[.interrupted,.cancelled])
        XCTAssertEqual(try Data(contentsOf:prepPath),prepBytes)
        let actual=try JSONDecoder().decode(WorkflowLaunchClaim.self,from:Data(contentsOf:f.directory.appendingPathComponent("claim-\(training.nonce).json")))
        XCTAssertEqual(actual,WorkflowLaunchClaim(schemaVersion:1,nonce:training.nonce,intentSHA256:training.intentSHA256,state:"revoked"))
        let before=try snapshot(f.directory)
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory:f.directory,workspaceDirectory:f.root),result)
        XCTAssertEqual(try snapshot(f.directory),before)
    }
    @MainActor func testClaimedSlotWithoutNonceIntentAndUnequalNonceBytesRefuse()throws {
        for missing in [true,false] {
            let f=try fixture(),prep=try publish(f,phase:.preparation)
            let path=f.directory.appendingPathComponent("launch-\(prep.nonce).json")
            if missing {
                try claim(prep).write(to:f.directory.appendingPathComponent("claim-\(prep.nonce).json"))
                try FileManager.default.removeItem(at:path)
            } else {try Data("{}".utf8).write(to:path)}
            try refuseUnchanged(f)
        }
    }
}
