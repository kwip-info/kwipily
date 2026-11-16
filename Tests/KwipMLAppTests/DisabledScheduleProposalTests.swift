import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DisabledScheduleProposalTests:XCTestCase {
    private func fixture()throws->(URL,ScheduleRule) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("disabled-schedule-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let doc=WorkflowDocument(projectID:UUID().uuidString,name:"Pinned source",nodes:[.init(address:.init(domain:.data,localID:"output"),name:"Table export",payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:[:]))])
        let version=try CanvasVersionJournal(root:root,projectID:doc.projectID).save(doc)
        return(root,.init(projectID:doc.projectID,projectName:"Untrusted supplied title",versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Disabled proposal",timing:.init(kind:.daily)))
    }
    func testPreviewHasNoLedgerSideEffectsAndSaveIsDisabledWithoutConsent()async throws {
        let (root,rule)=try fixture()
        let proposal=try DisabledScheduleProposal.preview(rule,root:root)
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("scheduled-runs").path))
        XCTAssertEqual(proposal.rule.projectName,"Pinned source");XCTAssertTrue(proposal.rule.hasEffects)
        XCTAssertEqual(proposal.effects.count,1);XCTAssertFalse(proposal.rule.allowEffects)
        let ledger=try ScheduledRunLedger(root:root)
        let saved=try await ledger.saveDisabled(proposal,authorize:{})
        XCTAssertEqual(saved.revision,1);XCTAssertFalse(saved.enabled);XCTAssertFalse(saved.allowEffects);XCTAssertNil(saved.outputAccessHash)
        let state=try await ledger.snapshot();XCTAssertFalse(state.enabled);XCTAssertTrue(state.jobs.isEmpty)
        await ledger.close()
        let persisted=try XCTUnwrap(ScheduledRunLedger.readOnlySnapshot(root:root));XCTAssertEqual(persisted.rules,[saved])
    }
    func testAtomicRuleRaceGrantRefusalAndNativeSaveRemainAuthoritative()async throws {
        let (root,rule)=try fixture(),proposal=try DisabledScheduleProposal.preview(rule,root:root)
        let ledger=try ScheduledRunLedger(root:root)
        do {_=try await ledger.saveDisabled(proposal,authorize:{throw MCPSettings.Refusal(code:"not_permitted")});XCTFail("Revoked policy admitted")}catch{}
        let before=try await ledger.snapshot();XCTAssertTrue(before.rules.isEmpty)
        var native=rule;native.name="Native winner";try await ledger.save(native)
        do {_=try await ledger.saveDisabled(proposal,authorize:{});XCTFail("Stale proposal overwrote native rule")}catch{}
        let after=try await ledger.snapshot();XCTAssertEqual(after.rules.first?.name,"Native winner");XCTAssertTrue(after.jobs.isEmpty)
        await ledger.close()
    }
    func testEnableConsentWrongPinAndChangedDestinationRefuseWithoutWrites()async throws {
        let (root,rule)=try fixture()
        var invalid=rule;invalid.enabled=true
        XCTAssertThrowsError(try DisabledScheduleProposal.preview(invalid,root:root))
        invalid=rule;invalid.allowEffects=true
        XCTAssertThrowsError(try DisabledScheduleProposal.preview(invalid,root:root))
        invalid=rule;invalid.versionHash=String(repeating:"0",count:64)
        XCTAssertThrowsError(try DisabledScheduleProposal.preview(invalid,root:root))
        let proposal=try DisabledScheduleProposal.preview(rule,root:root),ledger=try ScheduledRunLedger(root:root)
        try DiagnosticFileIO.writeAtomic(Data("{}".utf8),root:root,parts:["workflow-output-destinations",rule.projectID,"bindings.json"])
        do {_=try await ledger.saveDisabled(proposal,authorize:{});XCTFail("Destination change admitted")}catch{}
        let state=try await ledger.snapshot();XCTAssertTrue(state.rules.isEmpty);XCTAssertTrue(state.jobs.isEmpty)
        await ledger.close()
    }
}

extension DisabledScheduleProposalTests {
    func testDurableRequestReplayAfterRestartAndForeignOrChangedRequestsRefuse()async throws {
        let (root,rule)=try fixture(),proposal=try DisabledScheduleProposal.preview(rule,root:root)
        let request=DisabledScheduleRequest(client:"legacy-client",project:rule.projectID,id:UUID().uuidString,digest:String(repeating:"a",count:64))
        let ledger=try ScheduledRunLedger(root:root)
        let first=try await ledger.saveDisabled(proposal,request:request,authorize:{_ in})
        await ledger.close()
        let reopened=try ScheduledRunLedger(root:root)
        let replay=try await reopened.saveDisabled(nil,request:request,authorize:{_ in})
        XCTAssertEqual(first,replay)
        for bad in [DisabledScheduleRequest(client:request.client,project:rule.projectID,id:request.id,digest:String(repeating:"b",count:64)),
                    .init(client:"foreign",project:rule.projectID,id:request.id,digest:request.digest),
                    .init(client:request.client,project:UUID().uuidString,id:request.id,digest:request.digest)] {
            do {_=try await reopened.saveDisabled(nil,request:bad,authorize:{_ in});XCTFail("Cross-scope/conflicting request accepted")}catch{}
        }
        do {_=try await reopened.saveDisabled(nil,request:request,authorize:{_ in throw MCPSettings.Refusal(code:"not_permitted")});XCTFail("Replay bypassed revocation")}catch{}
        let state=try await reopened.snapshot();XCTAssertEqual(state.rules.count,1);XCTAssertEqual(state.rules.first?.revision,1);XCTAssertEqual(state.disabledRequests?.count,1)
        await reopened.close()
    }
    func testFailedPersistenceDoesNotLeaveRuleOrReceiptInMemory()async throws {
        let (root,rule)=try fixture(),proposal=try DisabledScheduleProposal.preview(rule,root:root),ledger=try ScheduledRunLedger(root:root)
        let folder=root.appendingPathComponent("scheduled-runs")
        try FileManager.default.setAttributes([.posixPermissions:0o500],ofItemAtPath:folder.path)
        defer {try? FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:folder.path)}
        do {_=try await ledger.saveDisabled(proposal,request:.init(client:"test",project:rule.projectID,id:UUID().uuidString,digest:String(repeating:"a",count:64)),authorize:{_ in});XCTFail("Unwritable ledger committed")}catch{}
        let state=try await ledger.snapshot();XCTAssertTrue(state.rules.isEmpty);XCTAssertNil(state.disabledRequests)
        XCTAssertTrue(try XCTUnwrap(ScheduledRunLedger.readOnlySnapshot(root:root)).rules.isEmpty)
        await ledger.close()
    }
}
