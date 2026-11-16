import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowRequiredCacheReportTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    /// New session captures use current operators; archived legacy integrity has dedicated fixtures.
    private func fixture(block:Bool=false)->WorkflowDocument {CurrentExecutionWorkflowTestFixture.make(block:block)}

    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func dataRegistry() throws -> DataFlowOperatorRegistry {
        try .load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
    }


    private struct Run { let root:URL;let outcome:WorkflowExecutionOutcome;let reference:WorkflowHistoryReference }
    @MainActor private func run(version:Int) async throws -> Run {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual native host preparation-to-training execution.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)

        let workspace = WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let runtime = try XCTUnwrap(workspace.runtime)
        var document = fixture(block:true); document.schemaVersion = version
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let model = try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
        let data = try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:model,dataRegistry:data)
        let host = try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.activeProjectID = owner.projectID
        workspace.workflowSession = host
        let csv = root.appendingPathComponent("host-observations.csv")
        let rows = (0..<64).map {"\($0),\(($0 * 7) % 13),\($0 % 2)"}
        try Data((["a,b,label"]+rows).joined(separator:"\n").utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:address(.data,"input"),lease:owner.lease(),expectedRevision:0))
        let finished = expectation(description:"Actual host publishes captured terminal outcome")
        let subscription = host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in finished.fulfill()}
        defer {subscription.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        XCTAssertTrue(workspace.isBusy, "Run capture reserves the workspace before asynchronous queue admission")
        await fulfillment(of:[finished],timeout:45)
        if workspace.isBusy {host.cancel()}
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        XCTAssertNil(outcome.recordError)

        host.retire();workspace.workflowSession=nil
        return Run(root:workspace.workspaceDirectory,outcome:outcome,reference:.init(projectID:document.projectID,
            directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent))))
    }
    private func replacingCache(_ run:Run,withNull:Bool) throws {
        let directory=run.outcome.directory
        let prepURL=directory.appendingPathComponent("preparation/artifacts/manifest.json")
        var prep=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:prepURL)) as? [String:Any])
        if withNull {prep["cache"]=NSNull()} else {prep.removeValue(forKey:"cache")}
        let prepBytes=try JSONSerialization.data(withJSONObject:prep,options:[.sortedKeys,.withoutEscapingSlashes])
        try prepBytes.write(to:prepURL)
        let prepPin=WorkflowArchiveCodec.hash(prepBytes)
        // Re-pin every dependent manifest reference, so failure cannot be explained
        // by a stale outer checksum or training-preparation linkage mismatch.
        let modelURL=directory.appendingPathComponent("training/artifacts/manifest.json")
        var model=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:modelURL)) as? [String:Any])
        var provenance=try XCTUnwrap(model["provenance"] as? [String:Any])
        var selection=try XCTUnwrap(provenance["preparation"] as? [String:Any])
        selection["manifestSHA256"]=prepPin;provenance["preparation"]=selection;model["provenance"]=provenance
        let modelBytes=try JSONSerialization.data(withJSONObject:model,options:[.sortedKeys,.withoutEscapingSlashes])
        try modelBytes.write(to:modelURL)
        let modelPin=WorkflowArchiveCodec.hash(modelBytes)
        let outcomeURL=directory.appendingPathComponent("outcome.json")
        var outcome=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:outcomeURL)) as? [String:Any])
        outcome["preparationManifestSHA256"]=prepPin;outcome["modelManifestSHA256"]=modelPin
        var artifacts=try XCTUnwrap(outcome["artifacts"] as? [[String:Any]])
        for index in artifacts.indices {
            if artifacts[index]["path"] as? String == "preparation/artifacts/manifest.json" {
                artifacts[index]["bytes"]=prepBytes.count;artifacts[index]["sha256"]=prepPin
            }
            if artifacts[index]["path"] as? String == "training/artifacts/manifest.json" {
                artifacts[index]["bytes"]=modelBytes.count;artifacts[index]["sha256"]=modelPin
            }
        }
        outcome["artifacts"]=artifacts
        try JSONSerialization.data(withJSONObject:outcome,options:[.sortedKeys,.withoutEscapingSlashes]).write(to:outcomeURL)
    }
    @MainActor func testOptInCompilerTwoOffRequiresReportEvenWithConsistentRepinnedArtifacts() async throws {
        let run=try await run(version:2);defer{try? FileManager.default.removeItem(at:run.root)}
        guard case .verified = WorkflowHistoryReader.load(run.reference,root:run.root) else {return XCTFail("Original archive should verify")}
        for null in [false,true] {
            try replacingCache(run,withNull:null)
            guard case .unavailable(let reason)=WorkflowHistoryReader.load(run.reference,root:run.root) else {
                XCTFail("Compiler2 off must require complete cache evidence, including null=\(null)");continue
            }
            XCTAssertTrue(reason.localizedCaseInsensitiveContains("cache"),reason)
        }
    }
    @MainActor func testOptInHistoricalCompilerOneAllowsAbsentCacheReport() async throws {
        let run=try await run(version:1);defer{try? FileManager.default.removeItem(at:run.root)}
        try replacingCache(run,withNull:false)
        guard case .verified(let result)=WorkflowHistoryReader.load(run.reference,root:run.root) else {return XCTFail("Legacy compiler1 omission must remain readable")}
        XCTAssertEqual(result.capture.compilerContractVersion,1)
        XCTAssertEqual(result.outcome.status,.completed)
        XCTAssertNotNil(result.modelContent)
        let output=run.outcome.directory.appendingPathComponent("preparation/artifacts")
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("manifest.json"))) as? [String:Any])
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparation(request:run.outcome.request,root:run.outcome.directory,
            completion:["outputDir":output.path,"manifestPath":output.appendingPathComponent("manifest.json").path,"graphDigest":try XCTUnwrap(manifest["graphDigest"]) ])) { error in
            XCTAssertTrue(error.localizedDescription.localizedCaseInsensitiveContains("cache"),error.localizedDescription)
        }
    }
}
