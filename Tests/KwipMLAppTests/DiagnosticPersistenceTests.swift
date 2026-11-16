import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class DiagnosticPersistenceTests:XCTestCase {
    @MainActor private func setup()throws->(WorkspaceStore,URL) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false);store.beginProject(.init(name: "Untitled flow"));return (store,root)
    }
    private func hash(_ data:Data)->String {SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()}
    @MainActor func testFailureAndCancellationPreserveOriginalInputsAndLegacyIsDistinct() throws {
        let (store,root)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let id=try XCTUnwrap(store.activeProjectID)
        store.metrics=[EpochMetric(epoch:50,trainLoss:0.5,validationLoss:0.6,accuracy:0.7)]
        store.latestReceivedMetric=EpochMetric(epoch:51,trainLoss:.nan,validationLoss:0.7,accuracy:0.6)
        let error=" \nMPS backend error\t "
        store.recordExperiment(status:"failed",directory:root.appendingPathComponent("untrusted-unused"),error:error)
        var record=try XCTUnwrap(store.experiments.first)
        XCTAssertNotNil(record.diagnosticReport,record.diagnosticError ?? "No save error");XCTAssertNil(record.trainingLoss)
        guard case .verified(let report)=store.loadExperimentDiagnostics(record) else {return XCTFail("Expected verified original report")}
        XCTAssertEqual(report.findings.first?.evidence,error)
        XCTAssertEqual(report.observedInputCount,2);XCTAssertEqual(report.context.sampling,.sampled)
        XCTAssertTrue(report.findings.contains {$0.id == "measurement.nonfinite.v1"})
        store.metrics=[];store.latestReceivedMetric=nil
        let reopened=WorkspaceStore(rootOverride:root,detectCapabilities:false);reopened.activeProjectID=id;reopened.refreshExperiments()
        guard case .verified(let restored)=reopened.loadExperimentDiagnostics(try XCTUnwrap(reopened.experiments.first)) else {return XCTFail("Original report unavailable after reopen")}
        XCTAssertEqual(restored,report)
        record.status="cancelled"
        guard case .unavailable=store.loadExperimentDiagnostics(record) else {return XCTFail("Altered terminal identity accepted")}
        record.diagnosticReport=nil;record.diagnosticError=nil
        guard case .legacy=store.loadExperimentDiagnostics(record) else {return XCTFail("Legacy must not invent an original report")}
        store.recordExperiment(status:"cancelled",directory:root.appendingPathComponent("cancelled"),error:nil)
        guard case .verified(let cancelled)=store.loadExperimentDiagnostics(try XCTUnwrap(store.experiments.first)) else {return XCTFail("Cancelled snapshot should persist without a completed manifest")}
        XCTAssertTrue(cancelled.findings.isEmpty)
    }
    @MainActor func testCompletedManifestIsPinnedAndExternalOrChangedManifestFails() throws {
        let (store,root)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let directory=store.workspaceDirectory.appendingPathComponent("runs/\(UUID().uuidString)/artifacts")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let graph=try JSONSerialization.jsonObject(with:JSONEncoder().encode(store.graph))
        let manifest=try JSONSerialization.data(withJSONObject:["schemaVersion":1,"status":"completed","graph":graph])
        let url=directory.appendingPathComponent("manifest.json");try manifest.write(to:url)
        store.recordExperiment(status:"completed",directory:directory,error:nil)
        let record=try XCTUnwrap(store.experiments.first)
        guard case .verified=store.loadExperimentDiagnostics(record) else {return XCTFail("Expected manifest-pinned report")}
        try Data("changed".utf8).write(to:url)
        guard case .unavailable=store.loadExperimentDiagnostics(record) else {return XCTFail("Changed manifest accepted")}
        store.recordExperiment(status:"completed",directory:root,error:nil)
        XCTAssertNotNil(store.experiments.first?.diagnosticError)
        guard case .unavailable=store.loadExperimentDiagnostics(try XCTUnwrap(store.experiments.first)) else {return XCTFail("Missing completed manifest became legacy")}
    }
    @MainActor func testOperationManifestRequiresExpectedProjectAndPreservesLegacyNamespace() throws {
        let (store,root)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let project=try XCTUnwrap(store.activeProjectID), operation=UUID().uuidString
        let directory=store.workspaceDirectory.appendingPathComponent("workflow-operations/\(project)/\(operation)/artifacts")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let graph=try JSONSerialization.jsonObject(with:JSONEncoder().encode(store.graph))
        let bytes=try JSONSerialization.data(withJSONObject:["schemaVersion":1,"status":"completed","graph":graph])
        try bytes.write(to:directory.appendingPathComponent("manifest.json"))
        store.recordExperiment(status:"completed",directory:directory,error:nil)
        let record=try XCTUnwrap(store.experiments.first)
        XCTAssertNil(record.diagnosticError)
        XCTAssertEqual(try DiagnosticPersistence.completedManifest(record:record,projectID:project,root:store.workspaceDirectory),bytes)
        guard case .verified=store.loadExperimentDiagnostics(record) else {return XCTFail("Operation report must reopen")}
        XCTAssertThrowsError(try DiagnosticPersistence.completedManifest(record:record,projectID:UUID().uuidString,root:store.workspaceDirectory))
        var changed=record
        changed.directory=store.workspaceDirectory.path + "/workflow-operations/" + project + "/" + operation + "/../" + operation + "/artifacts"
        XCTAssertThrowsError(try DiagnosticPersistence.completedManifest(record:changed,projectID:project,root:store.workspaceDirectory))
        let legacy=store.workspaceDirectory.appendingPathComponent("runs/\(UUID().uuidString)/artifacts")
        try FileManager.default.createDirectory(at:legacy,withIntermediateDirectories:true)
        try bytes.write(to:legacy.appendingPathComponent("manifest.json"));changed.directory=legacy.path
        XCTAssertEqual(try DiagnosticPersistence.completedManifest(record:changed,projectID:project,root:store.workspaceDirectory),bytes)
        let displaced=root.appendingPathComponent("original-operation")
        try FileManager.default.moveItem(at:directory.deletingLastPathComponent(),to:displaced)
        try FileManager.default.createSymbolicLink(at:directory.deletingLastPathComponent(),withDestinationURL:displaced)
        XCTAssertThrowsError(try DiagnosticPersistence.completedManifest(record:record,projectID:project,root:store.workspaceDirectory))
    }
    @MainActor func testPhysicalOperationPathMatchesConfiguredWorkspaceAlias() throws {
        let root=URL(fileURLWithPath:"/var/tmp").appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,storageOverride:root,detectCapabilities:false)
        store.beginProject(.init(name:"Alias test"))
        let project=try XCTUnwrap(store.activeProjectID)
        let configured=store.workspaceDirectory.appendingPathComponent("workflow-operations/\(project)/\(UUID().uuidString)/artifacts")
        try FileManager.default.createDirectory(at:configured,withIntermediateDirectories:true)
        let pointer=try XCTUnwrap(realpath(configured.path,nil));defer {free(pointer)}
        let physical=URL(fileURLWithPath:String(cString:pointer))
        XCTAssertNotEqual(configured.path,physical.path)
        XCTAssertTrue(configured.path.hasPrefix("/var/tmp/"))
        XCTAssertTrue(physical.path.hasPrefix("/private/var/tmp/"))
        let graph=try JSONSerialization.jsonObject(with:JSONEncoder().encode(store.graph))
        let bytes=try JSONSerialization.data(withJSONObject:["schemaVersion":1,"status":"completed","graph":graph])
        try bytes.write(to:physical.appendingPathComponent("manifest.json"))
        for directory in [physical,configured] {
            store.recordExperiment(status:"completed",directory:directory,error:nil)
            let record=try XCTUnwrap(store.experiments.first)
            XCTAssertNil(record.diagnosticError,"Configured root and its physical alias identify the same operation")
            XCTAssertEqual(try DiagnosticPersistence.completedManifest(record:record,projectID:project,root:store.workspaceDirectory),bytes)
            guard case .verified=store.loadExperimentDiagnostics(record) else {return XCTFail("Aliased operation diagnostic must reopen")}
        }
    }
    @MainActor func testArtifactTamperingRepinningAndSymlinkAreRejected() throws {
        let (store,root)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        store.recordExperiment(status:"failed",directory:root,error:"opaque failure")
        var record=try XCTUnwrap(store.experiments.first)
        let artifact=store.workspaceDirectory.appendingPathComponent("diagnostic-reports/\(try XCTUnwrap(store.activeProjectID))/\(record.id).json")
        let original=try Data(contentsOf:artifact)
        var json=try XCTUnwrap(JSONSerialization.jsonObject(with:original) as? [String:Any])
        var report=try XCTUnwrap(json["report"] as? [String:Any])
        var findings=try XCTUnwrap(report["findings"] as? [[String:Any]])
        findings[0]["evidence"]="invented error";report["findings"]=findings;json["report"]=report
        let altered=try JSONSerialization.data(withJSONObject:json);try altered.write(to:artifact)
        record.diagnosticReport?.sha256=hash(altered)
        guard case .unavailable=store.loadExperimentDiagnostics(record) else {return XCTFail("Repinned non-reproducible report accepted")}
        try FileManager.default.removeItem(at:artifact)
        let outside=root.appendingPathComponent("outside.json");try original.write(to:outside)
        try FileManager.default.createSymbolicLink(at:artifact,withDestinationURL:outside)
        record.diagnosticReport?.sha256=hash(original)
        guard case .unavailable=store.loadExperimentDiagnostics(record) else {return XCTFail("Symlink artifact accepted")}
    }
    @MainActor func testAncestorLinkRefusesReadWriteAndSaveFailurePreservesRecord() throws {
        let (store,root)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        store.recordExperiment(status:"failed",directory:root,error:"original failure")
        let record=try XCTUnwrap(store.experiments.first)
        let directory=store.workspaceDirectory.appendingPathComponent("diagnostic-reports")
        let displaced=root.appendingPathComponent("displaced-reports")
        try FileManager.default.moveItem(at:directory,to:displaced)
        try FileManager.default.createSymbolicLink(at:directory,withDestinationURL:displaced)
        guard case .unavailable=store.loadExperimentDiagnostics(record) else {return XCTFail("Ancestor link accepted for read")}
        let before=try FileManager.default.contentsOfDirectory(atPath:displaced.appendingPathComponent(try XCTUnwrap(store.activeProjectID)).path)
        store.recordExperiment(status:"cancelled",directory:root,error:nil)
        XCTAssertEqual(store.experiments.count,2)
        let cancelled=try XCTUnwrap(store.experiments.first)
        XCTAssertEqual(cancelled.status,"cancelled");XCTAssertNil(cancelled.diagnosticReport);XCTAssertNotNil(cancelled.diagnosticError)
        guard case .unavailable=store.loadExperimentDiagnostics(cancelled) else {return XCTFail("Save failure became legacy")}
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:displaced.appendingPathComponent(try XCTUnwrap(store.activeProjectID)).path),before)
    }
    @MainActor func testExtremeEpochsAndImmutableBoundedArtifact() throws {
        let (store,root)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        store.metrics=[Int.min,Int.max].map {EpochMetric(epoch:$0,trainLoss:0.5,validationLoss:0.6,accuracy:0.7)}
        store.recordExperiment(status:"failed",directory:root,error:nil)
        let record=try XCTUnwrap(store.experiments.first)
        guard case .verified(let report)=store.loadExperimentDiagnostics(record) else {return XCTFail("Extreme epochs should not trap")}
        XCTAssertEqual(report.context.sampling,.sampled)
        let id=try XCTUnwrap(store.activeProjectID)
        XCTAssertThrowsError(try DiagnosticPersistence.save(record:record,projectID:id,metrics:store.metrics,root:store.workspaceDirectory))
        guard case .verified=store.loadExperimentDiagnostics(record) else {return XCTFail("Exclusive write changed original")}
        let path=store.workspaceDirectory.appendingPathComponent("diagnostic-reports/\(id)/\(record.id).json")
        try Data(repeating:32,count:DiagnosticFileIO.maximumBytes+1).write(to:path)
        guard case .unavailable=store.loadExperimentDiagnostics(record) else {return XCTFail("Oversized artifact accepted")}
    }

}
