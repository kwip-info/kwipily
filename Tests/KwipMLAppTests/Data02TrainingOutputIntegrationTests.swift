import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class Data02TrainingOutputIntegrationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func address(_ id:String)->WorkflowNodeAddress {.init(domain:.data,localID:id)}
    private func wire(_ from:String,_ port:String,_ to:String)->WorkflowEdge {
        .init(id:"data02-\(from)-\(to)",from:.init(address:address(from),port:port),to:.init(address:address(to),port:"table"),kind:.table)
    }
    func testRealTrainingIncludesRenamedFilteredOutputBranchAndColdHistory()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real isolated runner interpreter.")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        var rename=try ColumnsConfigurationDraft(configuration:ColumnsConfigurationDraft.defaults)
        for name in ["measurement","context","label"] {rename.add(source:name)}
        rename.update("measurement") {$0.outputName="signal"};rename.newColumns="drop"
        document.nodes.append(.init(address:address("rename"),payload:.dataOperator(operatorID:"kwip.data.columns",config:rename.configuration)))
        var filter=FilterRowsConfigurationDraft.defaults
        filter["mode"] = .string("expression");filter["conditions"] = .array([])
        filter["expression"] = .string("[signal] >= -4");filter["column"] = .string("signal")
        document.nodes.append(.init(address:address("expression"),payload:.dataOperator(operatorID:"kwip.data.filterRows",config:filter)))
        document.nodes.append(.init(address:address("save"),payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string("filtered.csv"),"separator":.string(","),"columns":.array([.string("signal"),.string("label")]),"overwrite":.bool(false)])))
        document.edges.removeAll {($0.from.address==WorkflowExamples.sourceAddress && $0.to.address==address("cast")) || ($0.from.address==address("cast") && $0.to.address==address("split"))}
        document.edges += [wire("observations","table","rename"),wire("rename","table","cast"),wire("cast","table","expression"),wire("expression","matching","split"),wire("expression","matching","save")]
        let setup=try XCTUnwrap(document.nodes.firstIndex {$0.address == .init(domain:.control,localID:"setup")})
        guard case .runSetup(var config)=document.nodes[setup].payload else {return XCTFail("Missing run setup")}
        config["epochs"] = .integer(2);document.nodes[setup].payload = .runSetup(config:config)
        let source=root.appendingPathComponent("observations.csv")
        let signals=Array(-8 ... -1)+Array(1 ... 8)
        let rows=signals.map {"\($0),\($0*2),\($0<0 ? 0:1)"}
        try Data((["measurement,context,label"]+rows).joined(separator:"\n").utf8).write(to:source)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(source.path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let destination=root.appendingPathComponent("saved");try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        host.outputDestinations.pickDirectory={destination}
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address("save")]),extending:false)
        XCTAssertTrue(host.outputDestinations.chooseDestination(for:address("save"),lease:host.owner.lease()),host.outputDestinations.statusMessage)
        host.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}
        guard host.run() else {return XCTFail(host.statusMessage)}
        let deadline=Date().addingTimeInterval(90)
        while (host.outcomes.isEmpty || host.owner.isBusy) && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        if host.owner.isBusy {host.cancel()}
        let session=try XCTUnwrap(host.execution)
        XCTAssertEqual(session.phase,.completed,session.statusMessage)
        XCTAssertEqual(session.savedOutputs.count,1)
        XCTAssertEqual(session.savedOutputs.first?["rowCount"],.integer(12))
        XCTAssertEqual(session.savedOutputs.first?["completeSelectedTable"],.bool(true))
        let saved=try String(contentsOf:destination.appendingPathComponent("filtered.csv"),encoding:.utf8).split(whereSeparator:{$0.isNewline}).map {String($0).split(separator:",").map(String.init)}
        XCTAssertEqual(saved.first,["signal","label"])
        XCTAssertEqual(saved.dropFirst().compactMap {$0.first.flatMap(Double.init)},(Array(-4 ... -1)+Array(1 ... 8)).map(Double.init))
        XCTAssertEqual(saved.dropFirst().compactMap {$0.last.flatMap(Int.init)},Array(repeating:0,count:4)+Array(repeating:1,count:8))
        XCTAssertFalse(host.owner.isBusy)
        let outcome=try XCTUnwrap(host.outcomes.first)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let history)=WorkflowHistoryReader.load(reference,root:workspace.workspaceDirectory) else {return XCTFail("Cold History must independently verify the completed training archive")}
        XCTAssertEqual(history.outcome.status,.completed)
        XCTAssertTrue(history.plan.dataflow.nodes.contains {$0.id=="save"})
        let manifestURL=outcome.directory.appendingPathComponent("preparation/artifacts/manifest.json")
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:manifestURL))
        guard case .array(let receipts)=manifest["outputs"] else {return XCTFail("Durable saved-table receipts missing")}
        XCTAssertEqual(receipts.count,1)
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:workspace.workspaceDirectory)
        defer {package.remove()}
        for file in package.files where !file.path.hasSuffix(".pt") {
            let bytes=try Data(contentsOf:package.directory.appendingPathComponent(file.path))
            XCTAssertNil(bytes.range(of:Data(destination.path.utf8)),"Portable trained export leaked a private table destination in \(file.path)")
            XCTAssertNil(bytes.range(of:Data("destinations.json".utf8)))
        }
    }
}
