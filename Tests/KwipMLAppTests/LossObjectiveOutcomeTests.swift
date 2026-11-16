import XCTest
@testable import KwipMLApp

final class LossObjectiveOutcomeTests:XCTestCase {
    func testCompletedObjectiveHelperInventoryIsBoundedAndInterruptedRunsRefuseIt()throws {
        let preparation=WorkflowArchiveFile(path:"preparation/artifacts/manifest.json",bytes:100,sha256:String(repeating:"a",count:64))
        let model=WorkflowArchiveFile(path:"training/artifacts/manifest.json",bytes:200,sha256:String(repeating:"b",count:64))
        let helper=WorkflowArchiveFile(path:"training/artifacts/objective_recipe.py",bytes:131072,sha256:String(repeating:"c",count:64))
        let capture=WorkflowArchiveFile(path:"capture.json",bytes:300,sha256:String(repeating:"d",count:64))
        let time=WorkflowArchiveCodec.timestamp(Date(timeIntervalSince1970:1_000_000))
        let outcome=WorkflowArchiveOutcome(projectID:UUID().uuidString,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:String(repeating:"e",count:64),status:.completed,terminalPhase:.training,startedAt:time,finishedAt:time,artifacts:[preparation,model,helper],capture:capture,preparationManifestSHA256:preparation.sha256,modelManifestSHA256:model.sha256,workerRunID:UUID().uuidString)
        XCTAssertNoThrow(try WorkflowArchiveCodec.encodeOutcome(outcome))
        var oversized=outcome
        oversized=WorkflowArchiveOutcome(projectID:outcome.projectID,directoryID:outcome.directoryID,runID:outcome.runID,requestIdentity:outcome.requestIdentity,status:.completed,terminalPhase:.training,startedAt:time,finishedAt:time,artifacts:[preparation,model,.init(path:helper.path,bytes:131073,sha256:helper.sha256)],capture:capture,preparationManifestSHA256:preparation.sha256,modelManifestSHA256:model.sha256,workerRunID:outcome.workerRunID)
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeOutcome(oversized))
        let interrupted=WorkflowArchiveOutcome(projectID:outcome.projectID,directoryID:outcome.directoryID,runID:outcome.runID,requestIdentity:outcome.requestIdentity,status:.failed,terminalPhase:.training,startedAt:time,finishedAt:time,artifacts:[helper],error:"Actual training did not complete")
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeOutcome(interrupted))
    }
}
