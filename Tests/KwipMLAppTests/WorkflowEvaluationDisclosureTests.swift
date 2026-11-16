import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Synthetic verified values exercise disclosure sequencing only. Real archive
/// authentication and live-session accounting are covered by integration tests.
final class WorkflowEvaluationDisclosureTests:XCTestCase {
    private let pin=String(repeating:"a",count:64)
    private func root()throws->URL {
        guard let path=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw DataFlowContractError("No temporary directory")}
        defer {free(path)}
        let root=URL(fileURLWithPath:String(cString:path)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        return root
    }
    func fixture(membershipOverride:String?=nil)->(WorkflowEvaluationHistoryReference,VerifiedWorkflowEvaluationRecord) {
        let reference=WorkflowEvaluationHistoryReference(projectID:UUID().uuidString,evaluationID:UUID())
        let parent=WorkflowFrozenModelReference(projectID:reference.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,outcomeSHA256:pin,modelManifestSHA256:pin,preparationManifestSHA256:pin,checkpointSHA256:pin)
        let prediction=WorkflowEvaluationPrediction(observationID:"held-out-row",logits:[1,0],predicted:0,target:0)
        let summary=WorkflowEvaluationSummary(schemaVersion:1,evaluatedCount:1,device:"cpu",batchSize:256,lossSum:0.3,lossMean:0.3,accuracy:1,confusionMatrix:[[1,0],[0,0]],perClass:[],preview:[prediction])
        let membership=try! WorkflowEvaluationExposureStore.Population(checkpointSHA256:pin,sourceSHA256:pin,observationIDs:[prediction.observationID]).membershipSHA256
        let manifest=WorkflowEvaluationManifest(schemaVersion:1,kind:"frozenEvaluation",status:"completed",evaluationID:reference.evaluationID.uuidString,projectID:reference.projectID,requestIdentity:pin,requestSHA256:pin,snapshotReceiptSHA256:pin,parent:parent,testEndpoint:.init(node:"split",port:"test"),settings:.init(),testProvenance:["sourceHash":.string(pin),"preparation":.object(["membershipDigest":.string(membershipOverride ?? membership)])],frozenLineage:[:],runtime:["device":"cpu"],artifacts:[])
        let value=WorkflowEvaluationArtifacts.Verified(manifest:manifest,manifestSHA256:pin,summary:summary,predictions:[prediction],files:[])
        let outcome=WorkflowEvaluationOutcome(evaluationID:reference.evaluationID.uuidString,projectID:reference.projectID,requestIdentity:pin,status:.completed,startedAt:Date(),finishedAt:Date(),capture:nil,manifestSHA256:pin,artifacts:[],error:nil)
        return (reference,.init(outcome:outcome,result:value,captureIncomplete:false))
    }
    private func report(_ root:URL)throws->WorkflowEvaluationExposureStore.Report {
        let population=try WorkflowEvaluationExposureStore.Population(checkpointSHA256:pin,sourceSHA256:pin,observationIDs:["held-out-row"])
        guard case .tracked(let report)=WorkflowEvaluationExposureStore(root:root).history(for:population) else {throw DataFlowContractError("Expected durable history")}
        return report
    }
    func testEvaluationIsRecordedWithoutInspectionThenDisclosureIsDurableAndIdempotent()throws {
        let root=try root(),(reference,record)=fixture(),value=try XCTUnwrap(record.result)
        try WorkflowEvaluationDisclosure.recordEvaluation(value,root:root)
        XCTAssertEqual(try report(root).evaluatedCount,1)
        XCTAssertEqual(try report(root).inspectionCount,0)
        var loads=0
        let loader:(WorkflowEvaluationHistoryReference,URL)->WorkflowEvaluationHistoryLoad={ requested,requestedRoot in
            XCTAssertEqual(requested,reference);XCTAssertEqual(requestedRoot,root);loads+=1
            return .terminal(record)
        }
        let disclosed=try WorkflowEvaluationDisclosure.load(reference,root:root,verifiedLoader:loader)
        XCTAssertEqual(disclosed.value.manifestSHA256,pin)
        XCTAssertEqual(disclosed.priorExposure.evaluatedCount,1)
        XCTAssertEqual(disclosed.priorExposure.inspectionCount,0)
        XCTAssertTrue(disclosed.priorExposure.priorHistoryUnknown)
        XCTAssertEqual(try report(root).inspectionCount,1,"Disclosure must be durable before returning metrics")
        let replay=try WorkflowEvaluationDisclosure.load(reference,root:root,verifiedLoader:loader)
        XCTAssertEqual(replay.priorExposure.inspectionCount,1)
        XCTAssertEqual(try report(root).inspectionCount,1)
        XCTAssertEqual(loads,2,"Each disclosure independently reloads saved authority")
    }
    func testUnavailableHistoryCannotInitializeAccountingOrDisclose()throws {
        let root=try root(),(reference,record)=fixture()
        let failures:[WorkflowEvaluationHistoryLoad]=[.inUse,.incomplete("No outcome"),.unavailable("Changed evidence"),.terminal(.init(outcome:record.outcome,result:nil,captureIncomplete:true))]
        for failure in failures {
            XCTAssertThrowsError(try WorkflowEvaluationDisclosure.load(reference,root:root,verifiedLoader:{_,_ in failure}))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("workflow-evaluation-exposure").path))
    }
    func testWrongReferenceDoesNotRecordOrDisclose()throws {
        let root=try root(),(_,record)=fixture(),(wrong,_)=fixture()
        XCTAssertThrowsError(try WorkflowEvaluationDisclosure.load(wrong,root:root,verifiedLoader:{_,_ in .terminal(record)}))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("workflow-evaluation-exposure").path))
    }
    func testMismatchedProtectedMembershipCannotRecordOrDisclose()throws {
        let root=try root(),(reference,record)=fixture(membershipOverride:String(repeating:"b",count:64))
        XCTAssertThrowsError(try WorkflowEvaluationDisclosure.recordEvaluation(XCTUnwrap(record.result),root:root))
        XCTAssertThrowsError(try WorkflowEvaluationDisclosure.load(reference,root:root,verifiedLoader:{_,_ in .terminal(record)}))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("workflow-evaluation-exposure").path))
    }
    func testDamagedJournalBlocksMetricsAndIsNotReset()throws {
        let root=try root(),(reference,record)=fixture()
        try WorkflowEvaluationDisclosure.recordEvaluation(XCTUnwrap(record.result),root:root)
        let index=root.appendingPathComponent("workflow-evaluation-exposure/index.json"),damaged=Data("broken".utf8)
        try damaged.write(to:index)
        XCTAssertThrowsError(try WorkflowEvaluationDisclosure.load(reference,root:root,verifiedLoader:{_,_ in .terminal(record)}))
        XCTAssertEqual(try Data(contentsOf:index),damaged)
    }
    func testExternalCopiesDiscloseAgainstSharedExposureButReadTheirOwnHistory() throws {
        let app = try root(), a = app.appendingPathComponent("external-projects/" + UUID().uuidString)
        let b = app.appendingPathComponent("external-projects/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: a, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: b, withIntermediateDirectories: true)
        let (first, recordA) = fixture(), (second, recordB) = fixture()
        _ = try WorkflowEvaluationDisclosure.load(first, root: a, exposureRoot: app, verifiedLoader: { _, local in
            XCTAssertEqual(local, a); return .terminal(recordA)
        })
        let disclosed = try WorkflowEvaluationDisclosure.load(second, root: b, exposureRoot: app, verifiedLoader: { _, local in
            XCTAssertEqual(local, b); return .terminal(recordB)
        })
        XCTAssertEqual(disclosed.priorExposure.inspectionCount, 1)
        XCTAssertEqual(disclosed.priorExposure.sameSourceInspectedOverlap, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: b.appendingPathComponent("workflow-evaluation-exposure").path))
    }

    func testSeparateExistingExposureJournalRequiresReconciliationWithoutReset() throws {
        let app = try root(), local = app.appendingPathComponent("external-projects/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        let (reference, record) = fixture()
        _ = try WorkflowEvaluationDisclosure.load(reference, root: local, verifiedLoader: { _, _ in .terminal(record) })
        let journal = local.appendingPathComponent("workflow-evaluation-exposure/index.json")
        let before = try Data(contentsOf: journal)
        XCTAssertThrowsError(try WorkflowEvaluationDisclosure.load(reference, root: local, exposureRoot: app, verifiedLoader: { _, _ in .terminal(record) })) { error in
            XCTAssertTrue(error.localizedDescription.contains("reconcil"))
        }
        XCTAssertEqual(try Data(contentsOf: journal), before)
        XCTAssertFalse(FileManager.default.fileExists(atPath: app.appendingPathComponent("workflow-evaluation-exposure").path))
    }

}
