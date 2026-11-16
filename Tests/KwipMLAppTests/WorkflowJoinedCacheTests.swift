import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowJoinedCacheTests:XCTestCase {
    @MainActor private func workspace(_ root:URL)async throws->WorkspaceStore {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual worker")}
        let store=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(store)
        XCTAssertTrue(store.loadJoinedWorkflowExample(device:"cpu"),store.statusMessage)
        let document=try WorkflowCodec.migrateToVersion2(XCTUnwrap(store.workflowSession).owner.document)
        store.newProject()
        _ = try store.projectLibrary.saveSource(.workflow(document))
        XCTAssertTrue(store.beginWorkflowProject(id:document.projectID),store.statusMessage)
        return store
    }
    @MainActor private func policy(_ value:String,_ host:WorkflowWorkspaceSession)throws {
        let node=try XCTUnwrap(host.owner.document.nodes.first {if case .train=$0.payload {return true};return false})
        guard case .train(let trainConfig)=node.payload else {return XCTFail("Missing Train")}
        var config=try XCTUnwrap(WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:host.owner.document.schemaVersion))
        config["preparationCachePolicy"] = .string(value)
        XCTAssertNotNil(host.owner.apply(.configure(address:node.address,config:try WorkflowTrainRunSetup.setting(config,in:trainConfig,schemaVersion:host.owner.document.schemaVersion)),lease:host.owner.lease()),host.owner.error ?? "Configuration rejected")
    }
    @MainActor private func run(_ store:WorkspaceStore)async throws->VerifiedWorkflowOutcome {
        let host=try XCTUnwrap(store.workflowSession)
        try await awaitWorkflowIntegrationIdle(store,host:host)
        let count=host.outcomes.count
        let done=expectation(description:"Joined lifecycle terminal")
        let observation=host.$outcomes.filter {$0.count>count}.prefix(1).sink {_ in done.fulfill()}
        defer {observation.cancel()}
        guard host.run() else {throw WorkflowArchiveError(host.statusMessage)}
        await fulfillment(of:[done],timeout:60)
        try await awaitWorkflowIntegrationIdle(store,host:host)
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        XCTAssertFalse(store.isBusy);XCTAssertFalse(host.owner.isBusy)
        let reference=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let value)=WorkflowHistoryReader.load(reference,root:outcome.directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) else {throw NSError(domain:"JoinedCache",code:1)}
        return value
    }
    @MainActor func testActualReuseInvalidatesJoinForEitherChangedSource()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=try await workspace(root),host=try XCTUnwrap(store.workflowSession)
        try policy("materialize",host)
        let baseline=try await run(store)
        let reference=baseline.reference
        try policy("reuse",host)
        XCTAssertTrue(host.selectPreparationCache(reference:reference,for:WorkflowExamples.trainAddress,lease:host.owner.lease()),host.statusMessage)
        let reused=try await run(store)
        let sources:Set<String>=[WorkflowExamples.sourceAddress.localID,WorkflowExamples.joinContextAddress.localID]
        let descendants:Set<String>=["join","cast","split","fit","training","validation"]
        XCTAssertEqual(Set(try XCTUnwrap(reused.preparationCacheReport).executedNodeIDs),sources)
        XCTAssertEqual(Set(try XCTUnwrap(reused.preparationCacheReport).reusedNodeIDs),descendants)
        XCTAssertEqual(reused.capture.sources,baseline.capture.sources)
        for address in [WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress] {
            let path=try XCTUnwrap(host.bindings.path(for:address)),original=try Data(contentsOf:URL(fileURLWithPath:path))
            var rows=String(decoding:original,as:UTF8.self).split(whereSeparator:\.isNewline).map(String.init)
            var values=rows[1].split(separator:",").map(String.init)
            values[1]=String(try XCTUnwrap(Double(values[1]))+0.12345)
            rows[1]=values.joined(separator:",")
            try (rows.joined(separator:"\n")+"\n").write(toFile:path,atomically:true,encoding:.utf8)
            let changed=try await run(store),report=try XCTUnwrap(changed.preparationCacheReport)
            XCTAssertTrue(report.reusedNodeIDs.isEmpty)
            XCTAssertEqual(Set(report.executedNodeIDs),sources.union(descendants))
            for source in changed.capture.sources {
                let previous=try XCTUnwrap(baseline.capture.sources.first {$0.address == source.address})
                if source.address == address {XCTAssertNotEqual(source.sha256,previous.sha256)}
                else {XCTAssertEqual(source.sha256,previous.sha256)}
            }
            try original.write(to:URL(fileURLWithPath:path))
        }
    }
    @MainActor func testCancellationDuringJoinedSourceCaptureReleasesOwnershipWithoutWorkerLaunch()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=try await workspace(root),host=try XCTUnwrap(store.workflowSession)
        var launches=0
        host.executionLaunch={_,_,_,_,_ in launches+=1;return .init(cancel:{})}
        let done=expectation(description:"Preparation cancellation acknowledged")
        let observation=store.$isPreparingData.dropFirst().filter{!$0}.prefix(1).sink{_ in done.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        XCTAssertTrue(store.isBusy)
        host.cancel()
        await fulfillment(of:[done],timeout:10)
        XCTAssertEqual(launches,0)
        XCTAssertFalse(store.isBusy);XCTAssertFalse(host.owner.isBusy);XCTAssertNil(store.cancelDataPreparation)
        // Immediate cancellation occurs before run admission. No worker means
        // no prepared/model artifact or manufactured terminal History receipt.
        XCTAssertTrue(host.outcomes.isEmpty)
        XCTAssertNil(host.execution)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-runs/\(host.projectID)").path))
    }
}
