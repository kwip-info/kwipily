import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExperimentRecoveryTransactionTests:XCTestCase {
    func testCapturedIdentityRevisionAndInvalidTransitionsPreserveLedger()throws {
        for mutation in ["project","revision","pendingCompleted","failedNoHash","cancelledNoHash","interruptedHash","stale"] {
            let f=try WorkflowExperimentRecoveryCoordinatorTests().fixture()
            defer {try? FileManager.default.removeItem(at:f.root)}
            try f.store.begin(ordinal:1,runID:UUID(),directoryID:UUID())
            let path=f.directory.appendingPathComponent("group.json"),expected=try Data(contentsOf:path)
            var next=f.store.record;next.stopRequested=true;next.variants[0].status = .interrupted;next.variants[1].status = .cancelled
            switch mutation {
            case "project":
                var object=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(next)) as? [String:Any])
                object["projectID"]=UUID().uuidString
                next=try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:JSONSerialization.data(withJSONObject:object))
            case "revision":next.revision+=1
            case "pendingCompleted":next.variants[1].status = .completed;next.variants[1].outcomeSHA256=String(repeating:"a",count:64)
            case "failedNoHash":next.variants[0].status = .failed
            case "cancelledNoHash":next.variants[0].status = .cancelled
            case "interruptedHash":next.variants[0].outcomeSHA256=String(repeating:"a",count:64)
            default:break
            }
            f.owner.close()
            let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:f.directory,groupID:f.store.record.groupID)
            defer {recovery.close()}
            if mutation == "stale" {try (expected+Data(" ".utf8)).write(to:path)}
            let before=try Data(contentsOf:path)
            XCTAssertThrowsError(try WorkflowExperimentGroupStore.commitRecovery(expected:expected,proposed:next,directory:f.directory,recovery:recovery),mutation)
            XCTAssertEqual(try Data(contentsOf:path),before,mutation)
        }
    }
    func testTerminalSiblingCannotBeRewritten()throws {
        let f=try WorkflowExperimentRecoveryCoordinatorTests().fixture()
        defer {try? FileManager.default.removeItem(at:f.root)}
        let run=UUID(),directory=UUID()
        try f.store.begin(ordinal:1,runID:run,directoryID:directory)
        try f.store.finish(ordinal:1,runID:run,directoryID:directory,requestIdentity:f.requests[0].requestIdentity,status:.completed,outcomeSHA256:String(repeating:"a",count:64))
        try f.store.begin(ordinal:2,runID:UUID(),directoryID:UUID())
        let path=f.directory.appendingPathComponent("group.json"),before=try Data(contentsOf:path)
        var next=f.store.record;next.stopRequested=true;next.variants[1].status = .interrupted;next.variants[0].outcomeSHA256=String(repeating:"b",count:64)
        f.owner.close()
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:f.directory,groupID:f.store.record.groupID)
        defer {recovery.close()}
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.commitRecovery(expected:before,proposed:next,directory:f.directory,recovery:recovery))
        XCTAssertEqual(try Data(contentsOf:path),before)
    }
}
