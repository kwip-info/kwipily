import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentWorkerDeathTests:XCTestCase {
    @MainActor private struct Fixture {
        let root: URL
        let csv: URL
        let workspace: WorkspaceStore
        var host: WorkflowWorkspaceSession { workspace.workflowSession! }
    }
    @MainActor private func fixture() async throws -> Fixture {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual worker-death execution.")
        }
        let canonical=try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path,nil))
        defer {free(canonical)}
        let root = URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        var document = try WorkflowCodec.migrateToVersion2(WorkflowExamples.preparedParallelClassifier(device: "cpu"))
        let index = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        guard case .runSetup(var config) = document.nodes[index].payload else { throw XCTSkip("Missing sample setup") }
        config["epochs"] = .integer(100)
        config["preparationCachePolicy"] = .string("materialize")
        document=try WorkflowCodec.migrateToVersion4(document)
        config["experiment"] = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8,9])])]),"budget":.object(["maximumRuns":3,"maximumTotalEpochs":300])])
        document.nodes[index].payload = .runSetup(config: config)
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let csv = root.appendingPathComponent("private-observations.csv")
        let resource = URL(fileURLWithPath: development).appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv")
        try FileManager.default.copyItem(at: resource, to: csv)
        let host = try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        return .init(root: root, csv: csv, workspace: workspace)
    }
    @MainActor func testSecondRealTrainingWorkerDiesAndPreservesCompletedSibling()async throws {
        let f=try await fixture()
        defer {if f.workspace.isBusy {f.host.cancel()};if !f.workspace.isBusy {try? FileManager.default.removeItem(at:f.root)}}
        let development=try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"])
        let script=URL(fileURLWithPath:development).appendingPathComponent("runner/tests/fixtures/training_worker_death.py")
        let proof=f.root.appendingPathComponent("worker-death-proof.json")
        var phases:[String]=[],snapshot:[String:Data]?,firstDirectory:URL?,targetNonce:String?
        f.host.executionLaunch={executable,arguments,directory,line,exit in
            let index=try XCTUnwrap(arguments.firstIndex(of:"--kwip-supervise"))
            let group=URL(fileURLWithPath:arguments[index+1]),nonce=arguments[index+2]
            let intent=try WorkflowLaunchIntent.decode(Data(contentsOf:group.appendingPathComponent("launch-\(nonce).json")))
            phases.append("\(intent.ordinal):\(intent.phase.rawValue)")
            if intent.ordinal == 2 && intent.phase == .preparation {
                let record=try WorkflowExperimentGroupStore.readRecord(directory:group)
                let first=try XCTUnwrap(record.variants.first)
                XCTAssertEqual(first.status,.completed)
                let context=try WorkflowExperimentHistoryReader.captureContext(directory:group)
                let verified=try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:1,workspaceDirectory:f.root,policy:.historyPinned)
                XCTAssertEqual(verified.run.outcome.status,.completed)
                let path=f.root.appendingPathComponent("workflow-runs/\(record.projectID)/\(try XCTUnwrap(first.directoryID))")
                firstDirectory=path;snapshot=try self.files(path)
            }
            if intent.ordinal == 2 && intent.phase == .training {
                targetNonce=nonce
                let encoded=String(decoding:try JSONEncoder().encode(arguments),as:UTF8.self)
                let handle=try ProcessPump.launchWithLifetime(executable:executable,arguments:["-B",script.path,encoded,proof.path],directory:directory,onLine:line,onExit:exit)
                return WorkflowWorkerHandle(cancel:{handle.cancel()})
            }
            return try WorkflowExecutionSession.launchWorker(executable,arguments,directory,line,exit)
        }
        let done=expectation(description:"Completed first variant and failed real second worker")
        let observation=f.host.$outcomes.filter {$0.count == 2}.prefix(1).sink {_ in done.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress),f.host.statusMessage)
        await fulfillment(of:[done],timeout:90)
        let runner=try XCTUnwrap(f.host.experimentRunner)
        XCTAssertFalse(runner.isRunning);XCTAssertFalse(f.workspace.isBusy)
        XCTAssertNil(f.workspace.cancelDataPreparation)
        let record=runner.group.record
        XCTAssertEqual(record.variants.map(\.status),[.completed,.failed,.cancelled])
        XCTAssertNil(record.variants[2].runID);XCTAssertNil(record.variants[2].directoryID)
        XCTAssertEqual(phases,["1:preparation","1:training","2:preparation","2:training"])
        let observed=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:proof)) as? [String:Any])
        XCTAssertEqual(observed["nonce"] as? String,targetNonce)
        XCTAssertEqual(observed["ordinal"] as? Int,2);XCTAssertEqual(observed["epoch"] as? Int,1)
        XCTAssertEqual(observed["returncode"] as? Int,-Int(SIGKILL))
        XCTAssertEqual(try files(XCTUnwrap(firstDirectory)),try XCTUnwrap(snapshot))
        let context=try WorkflowExperimentHistoryReader.captureContext(directory:runner.group.directory)
        XCTAssertEqual(try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:1,workspaceDirectory:f.root,policy:.historyPinned).run.outcome.status,.completed)
        let failed=try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:2,workspaceDirectory:f.root,policy:.historyPinned)
        XCTAssertEqual(failed.run.outcome.status,.failed);XCTAssertNotNil(failed.run.outcome.error)
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:runner.group.directory,groupID:record.groupID)
        recovery.close()
        let ledger=runner.group.directory.appendingPathComponent("group.json"),before=try Data(contentsOf:ledger)
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory:runner.group.directory,workspaceDirectory:f.root),record)
        XCTAssertEqual(try Data(contentsOf:ledger),before)
        XCTAssertEqual(try files(XCTUnwrap(firstDirectory)),try XCTUnwrap(snapshot))
    }
    private func files(_ root:URL)throws->[String:Data] {
        let enumerator=try XCTUnwrap(FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey]))
        var result:[String:Data]=[:]
        for case let file as URL in enumerator where try file.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true {
            result[String(file.path.dropFirst(root.path.count+1))]=try Data(contentsOf:file)
        }
        return result
    }
}
