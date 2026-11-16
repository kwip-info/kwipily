import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentRecoveryReceiptTests:XCTestCase {
    private func prior()->WorkflowExperimentGroupRecord {
        let hash=String(repeating:"a",count:64)
        return .init(schemaVersion:1,groupID:UUID().uuidString,projectID:UUID().uuidString,train:.init(domain:.control,localID:"train"),sourceRequestIdentity:hash,inputsManifestSHA256:hash,authoringSHA256:hash,revision:3,stopRequested:false,variants:[
            .init(ordinal:1,requestIdentity:hash,status:.running,runID:UUID().uuidString,directoryID:UUID().uuidString),
            .init(ordinal:2,requestIdentity:String(repeating:"b",count:64),status:.pending)])
    }
    private func notes()->[WorkflowExperimentRecoveryReceipt.Variant] {[
        .init(ordinal:1,priorStatus:.running,status:.interrupted,reason:.terminalEvidenceUnavailable,detail:"Terminal outcome is missing."),
        .init(ordinal:2,priorStatus:.pending,status:.cancelled,reason:.pendingCancelled,detail:"Pending work was not launched.")
    ]}
    private func recovered(_ before:WorkflowExperimentGroupRecord)->WorkflowExperimentGroupRecord {
        var next=before;next.stopRequested=true;next.revision+=1
        next.variants[0].status = .interrupted;next.variants[1].status = .cancelled
        return next
    }
    func testLegacyDecodeAndReceiptRoundTrip()throws {
        let before=prior(),bytes=try JSONEncoder().encode(before)
        XCTAssertNil(try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:bytes).recoveryReceipt)
        XCTAssertNil((try JSONSerialization.jsonObject(with:bytes) as? [String:Any])?["recoveryReceipt"])
        var next=recovered(before)
        let receipt=WorkflowExperimentRecoveryReceipt(schemaVersion:1,timestampUnixSeconds:1234,priorRevision:before.revision,variants:notes())
        next.recoveryReceipt=receipt
        try receipt.validate(record:next)
        try receipt.validateTransition(prior:before,next:next)
        XCTAssertEqual(try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:JSONEncoder().encode(next)),next)
    }
    func testReceiptRejectsInvalidBoundsOrTransitionEvidence()throws {
        let before=prior(),next=recovered(before),valid=notes()
        let cases:[WorkflowExperimentRecoveryReceipt]=[
            .init(schemaVersion:2,timestampUnixSeconds:1,priorRevision:3,variants:valid),
            .init(schemaVersion:1,timestampUnixSeconds:.infinity,priorRevision:3,variants:valid),
            .init(schemaVersion:1,timestampUnixSeconds:-1,priorRevision:3,variants:valid),
            .init(schemaVersion:1,timestampUnixSeconds:253402300800,priorRevision:3,variants:valid),
            .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:4,variants:valid),
            .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:[]),
            .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:[valid[0],valid[0]]),
            .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:Array(repeating:valid[0],count:33)),
            .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:[.init(ordinal:1,priorStatus:.running,status:.interrupted,reason:.verifiedOutcomeRecovered,detail:"Wrong reason"),valid[1]]),
            .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:[.init(ordinal:1,priorStatus:.running,status:.interrupted,reason:.terminalEvidenceUnavailable,detail:String(repeating:"😀",count:513)),valid[1]])]
        for receipt in cases {XCTAssertThrowsError(try receipt.validate(record:next))}
        let incomplete=WorkflowExperimentRecoveryReceipt(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:[valid[0]])
        XCTAssertThrowsError(try incomplete.validateTransition(prior:before,next:next),"Every changed ordinal needs exactly one note")
        let incorrectPrior=WorkflowExperimentRecoveryReceipt(schemaVersion:1,timestampUnixSeconds:1,priorRevision:3,variants:[.init(ordinal:1,priorStatus:.pending,status:.cancelled,reason:.pendingCancelled,detail:"Wrong original status"),valid[1]])
        XCTAssertThrowsError(try incorrectPrior.validateTransition(prior:before,next:next))
    }
    func testBoundedDetailPreservesUnicodeAndUTF8Budget() {
        let text=String(repeating:"é😀a",count:1000)
        let result=WorkflowExperimentRecoveryReceipt.boundedDetail(text)
        XCTAssertLessThanOrEqual(result.utf8.count,2048)
        XCTAssertTrue(text.hasPrefix(result))
        XCTAssertGreaterThan(result.utf8.count,2040)
        XCTAssertEqual(WorkflowExperimentRecoveryReceipt.boundedDetail("short"),"short")
    }
    @MainActor func testRecoveryTransactionRequiresReceiptAndRepeatedRecoveryPreservesIt()throws {
        let helper=WorkflowExperimentRecoveryCoordinatorTests(),f=try helper.fixture()
        defer {f.owner.close();try? FileManager.default.removeItem(at:f.root)}
        try f.store.begin(ordinal:1,runID:UUID(),directoryID:UUID())
        let before=f.store.record,path=f.directory.appendingPathComponent("group.json"),bytes=try Data(contentsOf:path)
        f.owner.close()
        let authority=try WorkflowGroupOwnership.acquireRecovery(directory:f.directory,groupID:before.groupID)
        defer {authority.close()}
        var next=before;next.stopRequested=true;next.variants[0].status = .interrupted;next.variants[1].status = .cancelled
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.commitRecovery(expected:bytes,proposed:next,directory:f.directory,recovery:authority))
        XCTAssertEqual(try Data(contentsOf:path),bytes)
        next.recoveryReceipt = .init(schemaVersion:1,timestampUnixSeconds:1,priorRevision:before.revision,variants:notes())
        let saved=try WorkflowExperimentGroupStore.commitRecovery(expected:bytes,proposed:next,directory:f.directory,recovery:authority)
        let savedBytes=try Data(contentsOf:path)
        XCTAssertEqual(saved.recoveryReceipt,next.recoveryReceipt)
        XCTAssertEqual(try WorkflowExperimentGroupStore.commitRecovery(expected:savedBytes,proposed:saved,directory:f.directory,recovery:authority),saved)
        XCTAssertEqual(try Data(contentsOf:path),savedBytes)
        var forged=saved;forged.recoveryReceipt=nil
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.commitRecovery(expected:savedBytes,proposed:forged,directory:f.directory,recovery:authority))
        XCTAssertEqual(try Data(contentsOf:path),savedBytes)
    }
}
