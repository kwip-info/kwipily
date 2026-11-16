import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InitializationHistoryExportTests:XCTestCase {
    func testActualCPUColdHistoryPinsHelperAndPortableQualifiedPolicy()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable actual development worker for initialization History/export")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.initializationRuntimeCapabilities.contains("initialization"))
        var document=try WorkflowExamples.preparedStepLRClassifier(device:"cpu")
        let data=try DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:development).appendingPathComponent("runner/dataflow-operators.json"))
        let previous=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:workspace.registry,dataRegistry:data,contractVersion:12)
        let learned=try XCTUnwrap(previous.model.graph.nodes.first {$0.type == "linear"})
        let initialization=try InitializationConfiguration(values:["seed":13,"rules":.array([.object(["owner":.object(["node":.string(learned.id),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0.1,"std":0.2])])])])
        // An explicit retained inline policy preserves its independent seed exactly.
        let trainIndex=try XCTUnwrap(document.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        guard case .train(var trainConfig)=document.nodes[trainIndex].payload else {throw WorkflowPhaseError("Missing Train")}
        trainConfig["initialization"]=initialization.recipeValue
        document.nodes[trainIndex].payload = .train(config:trainConfig)
        let setup=try XCTUnwrap(document.nodes.firstIndex {if case .runSetup=$0.payload{return true};return false})
        if case .runSetup(var config)=document.nodes[setup].payload {config["epochs"]=2;document.nodes[setup].payload = .runSetup(config:config)}
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:workspace.registry,dataRegistry:data)
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.activeProjectID=document.projectID;workspace.workflowSession=host
        let csv=root.appendingPathComponent("private-observations.csv")
        let rows=(0..<48).map {"\($0),\(($0*7)%13),\($0%2)"}
        try Data((["signal,context,label"]+rows).joined(separator:"\n").utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:WorkflowExamples.sourceAddress,lease:owner.lease(),expectedRevision:0))
        let finished=expectation(description:"Actual initialized training completes")
        let subscription=host.$outcomes.dropFirst().filter {!$0.isEmpty}.prefix(1).sink {_ in finished.fulfill()};defer {subscription.cancel()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        await fulfillment(of:[finished],timeout:60)
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage);XCTAssertNil(outcome.recordError)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        host.retire();workspace.workflowSession=nil
        guard case .verified(let archived)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Initialized run must cold-reopen from its immutable source and artifacts")}
        XCTAssertEqual(archived.plan.initializationRecipe?.configuration,initialization)
        let path="training/artifacts/initialization_recipe.py"
        let pin=try XCTUnwrap(archived.outcome.artifacts.first {$0.path==path})
        let helperURL=outcome.directory.appendingPathComponent(path),original=try Data(contentsOf:helperURL)
        XCTAssertEqual(pin.bytes,original.count);XCTAssertEqual(pin.sha256,WorkflowPhaseArtifacts.hash(original))
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:root);defer {package.remove()}
        XCTAssertEqual(package.files.first {$0.path=="initialization_recipe.py"}?.sha256,pin.sha256)
        XCTAssertEqual(try Data(contentsOf:package.directory.appendingPathComponent("initialization_recipe.py")),original)
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:package.manifestBytes)
        XCTAssertEqual(manifest["initializationRecipe"],try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(try WorkflowArchiveInitializationRecipe(try XCTUnwrap(archived.plan.initializationRecipe)))))
        let readme=try String(contentsOf:package.directory.appendingPathComponent("README.md"),encoding:.utf8)
        XCTAssertTrue(readme.contains("initialization_recipe.py"));XCTAssertTrue(readme.contains("fresh"))
        for bad in [Data("# changed\n".utf8),Data(repeating:65,count:131073)] {
            try bad.write(to:helperURL)
            guard case .unavailable=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Changed helper must invalidate cold History")}
            XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(reference,root:root))
            try original.write(to:helperURL)
        }
    }
}
