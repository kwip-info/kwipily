import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowDataPreparationSessionTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func workflow(_ example:String="calculate")throws->WorkflowDocument {
        let graph=try DataFlowCodec.decodeGraph(Data(contentsOf:repository.appendingPathComponent("runner/dataflow_examples/data02-\(example).json")))
        return WorkflowDocument(projectID:UUID().uuidString,name:"Data preparation without Train",nodes:graph.nodes.enumerated().map {index,node in
            WorkflowNode(address:.init(domain:.data,localID:node.id),name:node.id,x:Double(index)*200,y:100,payload:.dataOperator(operatorID:node.operatorID,config:node.config))
        },edges:graph.edges.enumerated().map {index,edge in WorkflowEdge(id:"edge-\(index)",from:.init(address:.init(domain:.data,localID:edge.from.node),port:edge.from.port),to:.init(address:.init(domain:.data,localID:edge.to.node),port:edge.to.port),kind:.table)})
    }
    func testDataOnlyPlanRetainsEveryOperatorAndRefusesMissingRequiredInput()throws {
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        var document=try workflow()
        let graph=try WorkflowDataPreparationPlan.compile(document,registry:registry)
        XCTAssertEqual(graph.nodes.map(\.operatorID),["kwip.data.sourceCSV","kwip.data.columns","kwip.data.formula"])
        XCTAssertEqual(graph.edges.count,2)
        document.edges.removeLast()
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(document,registry:registry))
        document=try workflow();document.nodes.append(.init(address:.init(domain:.control,localID:"train"),payload:.train(config:[:])))
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(document,registry:registry))
    }
    func testScopedSourceNeverExecutesUnrelatedOutputModelOrAPI()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real isolated runner.")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var document=try workflow("output")
        document.nodes.append(.init(address:.init(domain:.control,localID:"unrelated-train"),payload:.train(config:[:])))
        let apiOptions=APITableSettings.defaults.merging(["url":JSONValue.string("https://api.example.com/orders"),"mapping":.array([.object(["source":.string("/id"),"target":.string("id"),"type":.string("int64")])])]){_,v in v}
        document.nodes.append(.init(address:.init(domain:.data,localID:"unrelated-api"),payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:["format":.string("api"),"options":.object(apiOptions)])))
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),source=WorkflowNodeAddress(domain:.data,localID:"source")
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}
        XCTAssertTrue(host.prepareData(at:source),host.statusMessage)
        let identity=host.operationIdentity,deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertEqual(host.scopedDataOperation?.identity,identity)
        XCTAssertEqual(host.scopedDataOperation?.state,"completed",host.statusMessage)
        let session=try XCTUnwrap(host.currentDataPreparation)
        XCTAssertEqual(session.phase,.completed,session.statusMessage)
        XCTAssertTrue(session.savedOutputs.isEmpty)
        XCTAssertNil(host.currentExecution)
        XCTAssertNil(host.currentBackendFit)
        XCTAssertNil(host.sourceProvider.apiFailureEvidence)
        XCTAssertEqual(host.owner.document,document)
        XCTAssertFalse(workspace.isBusy);XCTAssertFalse(host.owner.isBusy)
        let entry=try XCTUnwrap(host.dataPreparationHistory.entries.first)
        let verified=try await host.dataPreparationHistory.verify(entry)
        _=verified
        XCTAssertNotNil(session.snapshot(for:source))
        XCTAssertNil(session.snapshot(for:.init(domain:.data,localID:"tool")))
        // A later selected API request owns its failure, rather than inheriting
        // the completed source session. No private profile means no HTTP call.
        let api=WorkflowNodeAddress(domain:.data,localID:"unrelated-api")
        XCTAssertTrue(host.prepareData(at:api),host.statusMessage)
        let failedIdentity=host.operationIdentity
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertNotEqual(failedIdentity,identity)
        XCTAssertEqual(host.scopedDataOperation?.identity,failedIdentity)
        XCTAssertEqual(host.scopedDataOperation?.state,"failed")
        XCTAssertEqual(host.sourceProvider.apiFailureEvidence?.runID,failedIdentity)
        XCTAssertTrue(host.currentDataPreparation === session)
        XCTAssertTrue(host.prepareData(at:api),host.statusMessage)
        host.cancel()
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertEqual(host.scopedDataOperation?.state,"cancelled")
        XCTAssertFalse(workspace.isBusy);XCTAssertFalse(host.owner.isBusy)
    }
    func testRunPresentationEnablesDataOnlyCanvas()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try workflow()
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        XCTAssertTrue(workspace.canRunCurrentFlow,"Run Flow must be available on a data-only canvas; execution still validates inputs and permissions.")
    }
    func testRealDataOnlyManagedRunWithoutTrainAndSaveOutput()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real isolated runner interpreter.")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try workflow("output")
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:.init(domain:.data,localID:"source"),lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let destination=root.appendingPathComponent("saved-output")
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        host.outputDestinations.pickDirectory={destination}
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[.init(domain:.data,localID:"tool")]),extending:false)
        XCTAssertTrue(host.outputDestinations.chooseDestination(for:.init(domain:.data,localID:"tool"),lease:host.owner.lease()),host.outputDestinations.statusMessage)
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        XCTAssertTrue(host.owner.isBusy)
        let active=try XCTUnwrap(host.dataPreparation)
        XCTAssertFalse(host.run(),"A second submission must not replace an active preparation owner.")
        XCTAssertTrue(host.dataPreparation === active)
        let deadline=Date().addingTimeInterval(60)
        while host.dataPreparation?.isRunning==true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let session=try XCTUnwrap(host.dataPreparation)
        XCTAssertFalse(session.isRunning,session.statusMessage)
        XCTAssertEqual(session.phase,.completed,session.statusMessage+"\n"+session.messages.joined(separator:"\n"))
        XCTAssertNotNil(session.output)
        XCTAssertEqual(session.savedOutputs.count,1)
        XCTAssertEqual(session.savedOutputs.first?["rowCount"],.integer(8))
        XCTAssertTrue(FileManager.default.fileExists(atPath:destination.appendingPathComponent("prepared-orders.csv").path))
        let wrapped = try XCTUnwrap(host.snapshot(for:.init(domain:.data,localID:"tool")))
        let direct = try XCTUnwrap(session.snapshot(for:.init(domain:.data,localID:"tool")))
        XCTAssertEqual(wrapped.dataCompleteness, "Complete data")
        XCTAssertEqual(wrapped.dataCompleteness, direct.dataCompleteness)
        XCTAssertEqual(wrapped.capturedAt, direct.capturedAt)
        XCTAssertNotNil(wrapped.capturedAt, "The host must preserve the recorded capture time used for preview precedence.")
        XCTAssertFalse(workspace.isBusy);XCTAssertFalse(host.owner.isBusy)
        XCTAssertEqual(host.dataPreparationHistory.entries.count,1)
        let restored=WorkflowDataPreparationHistory(root:workspace.workspaceDirectory,projectID:document.projectID)
        await restored.reloadAndWait()
        XCTAssertEqual(restored.entries.count,1)
        _ = try await restored.verify(try XCTUnwrap(restored.entries.first))
    }
    func testWorkingDataSampleThenAllUsesFullPopulationAndInvalidatesPriorSample() async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real isolated runner interpreter.")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let sourceFile=root.appendingPathComponent("working-data.csv")
        let lines=try String(contentsOf:repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv"),encoding:.utf8).split(separator:"\n").map(String.init)
        try (lines[0]+"\n"+String(repeating:lines.dropFirst().joined(separator:"\n")+"\n",count:40)).write(to:sourceFile,atomically:true,encoding:.utf8)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try workflow("calculate")
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession)
        let source=WorkflowNodeAddress(domain:.data,localID:"source")
        XCTAssertTrue(host.bindings.setPath(sourceFile.path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        func run(_ policy:PreparationSampling) async throws -> WorkflowCapturedDataOutput {
            workspace.preparationSampling=policy
            XCTAssertTrue(host.run(),host.statusMessage)
            let deadline=Date().addingTimeInterval(60)
            while host.dataPreparation?.isRunning == true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
            XCTAssertEqual(host.dataPreparation?.phase,.completed,host.dataPreparation?.statusMessage ?? "Missing session")
            return try XCTUnwrap(host.snapshot(for:source))
        }
        func count(_ snapshot:WorkflowCapturedDataOutput) throws -> Int64 {
            guard case .object(let table)=snapshot.ports["table"],case .object(let preview)=table["preview"],case .integer(let count)=preview["rowCount"] else { XCTFail("Missing row count"); return -1 }
            return count
        }
        let sampled=try await run(.init(mode:.rowCount,rowCount:3))
        XCTAssertEqual(try count(sampled),3)
        XCTAssertTrue(sampled.matchesCurrentConstruction && sampled.matchesCurrentBindings)
        workspace.preparationSampling = .all
        let old=try XCTUnwrap(host.snapshot(for:source))
        XCTAssertFalse(old.matchesCurrentConstruction && old.matchesCurrentBindings,"A prior sample must not appear current when All data is now requested.")
        XCTAssertEqual(try count(old),3,"Changing the setting must preserve captured evidence, not invent a refreshed table.")
        let complete=try await run(.all)
        XCTAssertEqual(try count(complete),320,"All data must replace the earlier three-record population.")
        if case .object(let table)=complete.ports["table"],case .object(let preview)=table["preview"],case .array(let rows)=preview["rows"] {
            XCTAssertEqual(rows.count,8,"Transport preview stays bounded; it must not limit the browsable captured population.")
        } else {XCTFail("Missing transport preview")}
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[source]),extending:false)
        let viewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:source,port:"table")))
        func page(_ offset:Int, expected:Int) async throws {
            XCTAssertTrue(viewer.load(offset:offset))
            let deadline=Date().addingTimeInterval(60)
            while viewer.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
            XCTAssertEqual(viewer.page?.totalRows,320,viewer.statusMessage)
            XCTAssertEqual(viewer.page?.rows.count,expected,viewer.statusMessage)
        }
        try await page(0,expected:100)
        try await page(100,expected:100)
        try await page(300,expected:20)
        workspace.preparationSampling = .init(mode:.percentage,percentage:12.5)
        XCTAssertFalse(viewer.load(),"A changed working population invalidates the old query context.")
        let percentage=try await run(.init(mode:.percentage,percentage:12.5))
        XCTAssertEqual(try count(percentage),40)
        let sampledViewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:source,port:"table")))
        XCTAssertTrue(sampledViewer.load())
        let deadline=Date().addingTimeInterval(60)
        while sampledViewer.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertEqual(sampledViewer.page?.totalRows,40,sampledViewer.statusMessage)
        XCTAssertEqual(sampledViewer.page?.rows.count,40,sampledViewer.statusMessage)
        XCTAssertEqual(complete.dataCompleteness,"Complete data")
        XCTAssertTrue(complete.matchesCurrentConstruction && complete.matchesCurrentBindings)
    }
    func testWorkspaceOutputWitnessForwardsVerifiedCompleteTableViewAndRejectsStaleSelection() async throws {
        guard let python = ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {
            throw XCTSkip("Select the real isolated runner interpreter.")
        }
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("view.csv")
        try Data(("x,y\n" + String(repeating: "1,2\n", count: 250)).utf8).write(to: source)
        let sourceAddress = WorkflowNodeAddress(domain: .data, localID: "source")
        let viewAddress = WorkflowNodeAddress(domain: .data, localID: "view")
        let document = WorkflowDocument(projectID: UUID().uuidString, name: "Complete View forwarding", nodes: [
            .init(address: sourceAddress, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:])),
            .init(address: viewAddress, payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))
        ], edges: [.init(id: "source-view", from: .init(address: sourceAddress, port: "table"),
                        to: .init(address: viewAddress, port: "table"), kind: .table)])
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        workspace.preparationSampling = .init(mode: .rowCount, rowCount: 3)
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(source.path, for: sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        host.executionLaunch = { _, arguments, directory, onLine, onExit in
            try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath: python), arguments, directory, onLine, onExit)
        }
        XCTAssertTrue(host.run(), host.statusMessage)
        let deadline = Date().addingTimeInterval(60)
        while host.dataPreparation?.isRunning == true && Date() < deadline { try await Task.sleep(for: .milliseconds(50)) }
        let preparation = try XCTUnwrap(host.currentDataPreparation)
        XCTAssertEqual(preparation.phase, .completed, preparation.statusMessage + "\n" + preparation.messages.joined(separator: "\n"))
        let entry = try XCTUnwrap(host.dataPreparationHistory.entries.first)
        let artifact = try XCTUnwrap(entry.tableViews[viewAddress.localID])
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifact.indexURL.path), "Use a real admitted complete-table index, rather than a preview-only fake.")
        XCTAssertEqual(WorkflowArchiveCodec.hash(try Data(contentsOf: artifact.indexURL)), artifact.indexSHA256)

        workspace.preparationSampling = .init(mode:.rowCount,rowCount:5)
        XCTAssertTrue(try XCTUnwrap(host.snapshot(for:sourceAddress)).matchesCurrentConstruction,"View requires complete ancestry under either global sample size.")
        let outputs: any WorkflowDataOutputs = host
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[sourceAddress]), extending: false)
        XCTAssertNil(outputs.fullTableViewSession(for: viewAddress), "An unselected View must not acquire a UI query context.")
        let selection=host.owner.selection
        let authorized=try XCTUnwrap(host.automationTableViewSession(for:.init(address:viewAddress,port:"table")))
        XCTAssertEqual(host.owner.selection,selection,"An independently authorized MCP query must not select its node.")
        XCTAssertTrue(authorized.load(offset:100))
        let queryDeadline=Date().addingTimeInterval(30)
        while authorized.isBusy && Date()<queryDeadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertEqual(authorized.page?.totalRows,250,authorized.statusMessage)
        XCTAssertEqual(authorized.page?.rows.count,100,authorized.statusMessage)
        XCTAssertEqual(host.owner.selection,selection)
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[viewAddress]), extending: false)
        let context = try WorkflowDataNodeContext(owner: host.owner, address: viewAddress, sources: host.sourceProvider, outputs: outputs)
        let forwarded = try XCTUnwrap(outputs.fullTableViewSession(for: viewAddress))
        XCTAssertTrue(context.fullTableViewSession === forwarded, "The selected-node output existential must reach the full table viewer.")
        XCTAssertTrue(preparation.fullTableViewSession(for: viewAddress) === forwarded, "Forwarding must retain the existing session and its guarded reservation.")
        XCTAssertEqual(context.outputPopulationLabel, "Complete data")
        XCTAssertFalse(workspace.isBusy)
        XCTAssertFalse(host.owner.isBusy, "Retrieving the table session does not start a query or release a lease.")

        host.canvas.moveNodes(ids: [try XCTUnwrap(host.canvas.projection.displayIDByAddress[viewAddress])], dx: 75, dy: 20)
        XCTAssertTrue(try XCTUnwrap(host.snapshot(for: viewAddress)).matchesCurrentConstruction, "Moving View must preserve its captured table.")
        XCTAssertTrue(try XCTUnwrap(host.snapshot(for: sourceAddress)).matchesCurrentConstruction, "Moving a downstream block must preserve upstream captures.")
        XCTAssertTrue(outputs.fullTableViewSession(for: viewAddress) === forwarded, "Presentation edits must retain the admitted full index query context.")

        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[sourceAddress]), extending: false)
        XCTAssertNil(context.fullTableViewSession)
        XCTAssertNil(outputs.fullTableViewSession(for: viewAddress))
        XCTAssertFalse(forwarded.load(), "A retained query session must refuse after selection authority changes.")
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[viewAddress]), extending: false)
        XCTAssertTrue(outputs.fullTableViewSession(for: viewAddress) === forwarded)
        XCTAssertNotNil(host.owner.apply(.configure(address: sourceAddress, config: ["separator": .string(";")]), lease: host.owner.lease()))
        XCTAssertFalse(try XCTUnwrap(host.snapshot(for: viewAddress)).matchesCurrentConstruction, "Parsing changes must stale captured data.")
        XCTAssertNil(outputs.fullTableViewSession(for: viewAddress))
        XCTAssertFalse(forwarded.load(), "A saved query must refuse after a computational change.")
        host.retire()
        XCTAssertNil(outputs.fullTableViewSession(for: viewAddress), "The host witness must reject retired owners even when the old View remains selected.")
        XCTAssertFalse(forwarded.load())
        XCTAssertFalse(workspace.isBusy)
    }
    func testLargeTableManagedExecutionExportAndColdHistoryUseBoundedPreviews() async throws {
        try await verifyLargeTableManagedExecution(sourceConfig: [:])
    }
    func testPolarsManagedExecutionExportAndColdHistoryUseBoundedPreviews() async throws {
        try await verifyLargeTableManagedExecution(sourceConfig: CSVParsingConfiguration.newDefaults)
    }
    private func verifyLargeTableManagedExecution(sourceConfig: [String: JSONValue]) async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real isolated runner interpreter.")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("large.csv")
        try Data(("x,y\n"+String(repeating:"1,2\n",count:500_001)).utf8).write(to:source)
        let sourceAddress=WorkflowNodeAddress(domain:.data,localID:"source")
        let sinkAddress=WorkflowNodeAddress(domain:.data,localID:"sink")
        let document=WorkflowDocument(projectID:UUID().uuidString,name:"500k preparation",nodes:[
            .init(address:sourceAddress,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:sourceConfig)),
            .init(address:sinkAddress,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string("complete.csv"),"columns":.array([]),"separator":.string(","),"overwrite":.bool(false)]))
        ],edges:[.init(id:"source-output",from:.init(address:sourceAddress,port:"table"),to:.init(address:sinkAddress,port:"table"),kind:.table)])
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(source.path,for:sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let destination=root.appendingPathComponent("export")
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        host.outputDestinations.pickDirectory={destination}
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[sinkAddress]),extending:false)
        XCTAssertTrue(host.outputDestinations.chooseDestination(for:sinkAddress,lease:host.owner.lease()))
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(90)
        while host.dataPreparation?.isRunning==true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let session=try XCTUnwrap(host.dataPreparation)
        XCTAssertEqual(session.phase,.completed,session.statusMessage+"\n"+session.messages.joined(separator:"\n"))
        XCTAssertEqual(session.savedOutputs.first?["rowCount"],.integer(500_001))
        let exported=destination.appendingPathComponent("complete.csv")
        let exportHash=WorkflowArchiveCodec.hash(try Data(contentsOf:exported))
        try FileManager.default.removeItem(at:source)
        if !sourceConfig.isEmpty {
            host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[sourceAddress]),extending:false)
            let viewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:sourceAddress,port:"table")))
            let started=Date()
            XCTAssertTrue(viewer.load())
            let queryDeadline=Date().addingTimeInterval(60)
            while viewer.isBusy && Date()<queryDeadline {try await Task.sleep(for:.milliseconds(50))}
            XCTAssertEqual(viewer.page?.totalRows,500_001,viewer.statusMessage)
            XCTAssertEqual(viewer.page?.rows.count,100,viewer.statusMessage)
            print("500k captured-table first page seconds: \(Date().timeIntervalSince(started))")
            let nextStarted=Date()
            XCTAssertTrue(viewer.load(offset:500_000))
            let finalDeadline=Date().addingTimeInterval(60)
            while viewer.isBusy && Date()<finalDeadline {try await Task.sleep(for:.milliseconds(50))}
            XCTAssertEqual(viewer.page?.positions,[500_000],viewer.statusMessage)
            XCTAssertEqual(viewer.page?.rows.count,1,viewer.statusMessage)
            print("500k captured-table cached final page seconds: \(Date().timeIntervalSince(nextStarted))")
        }
        let history=WorkflowDataPreparationHistory(root:workspace.workspaceDirectory,projectID:document.projectID)
        await history.reloadAndWait()
        let entry=try XCTUnwrap(history.entries.first)
        XCTAssertTrue(FileManager.default.fileExists(atPath:entry.artifactDirectory.appendingPathComponent("previews.json").path))
        let outputs=try await history.outputs(for:entry)
        guard case .object(let table)=outputs[sourceAddress]?["table"],case .object(let preview)=table["preview"] else {return XCTFail("Missing captured table preview")}
        XCTAssertEqual(preview["rowCount"],.integer(500_001))
        guard case .array(let rows)=preview["rows"] else {return XCTFail("Missing captured rows")}
        XCTAssertEqual(rows.count,8)
        XCTAssertEqual(WorkflowArchiveCodec.hash(try Data(contentsOf:exported)),exportHash)
        XCTAssertFalse(workspace.isBusy)
    }

}
