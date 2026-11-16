import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPredictionDataSessionTests:XCTestCase {
    func testPredictionSchemaProjectionKeepsQualifiedSourceIdentity()throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let train=WorkflowNodeAddress(domain:.control,localID:"train"),dataTrain=WorkflowNodeAddress(domain:.data,localID:"train"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:train,payload:.train(config:["runSetup":.object([:])])),.init(address:dataTrain,payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:[:])),.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(id:"prediction-link",from:.init(address:train,port:"predictions"),to:.init(address:view,port:"table"),kind:.table)])
        document.schemaVersion=17
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let schema=[DataColumnSchema(name:"predicted",dtype:"int64",nullable:false)]
        host.sourceProvider.capturedPredictionSchema={ $0 == .init(address:train,port:"predictions") ? schema:nil }
        XCTAssertEqual(host.sourceProvider.recordedSchema(for:view,in:host.owner.document),schema)
        XCTAssertNotNil(host.owner.apply(.reconnect(edgeID:"prediction-link",from:.init(address:dataTrain,port:"table"),to:.init(address:view,port:"table")),lease:host.owner.lease()))
        XCTAssertTrue(host.sourceProvider.recordedSchema(for:view,in:host.owner.document).isEmpty,"A data source with the same local ID must never borrow the control prediction schema.")
    }
    func testManagedPredictionBranchHasCompleteViewAndDurableHistory()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real isolated runner.")}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let ids=(0..<250).map {"row-\($0)"}
        let columns:[[String:Any]]=["observation_id","partition","target","output"].map {["name":$0,"dtype":"string","nullable":false]}
        let rows=ids.map {[$0,"validation","target","value"]}
        let dataVersion=WorkflowArchiveCodec.hash(try JSONSerialization.data(withJSONObject:["columns":columns,"rows":rows,"ids":ids],options:[.sortedKeys,.withoutEscapingSlashes]))
        let membership=WorkflowArchiveCodec.hash(Data(("kwip.membership.v1\n"+ids.sorted().map {"\($0.utf8.count):\($0)"}.joined()).utf8))
        let table:[String:Any]=["schemaVersion":1,"kind":"table","columns":columns,"rows":rows,"lineage":["role":"validation","sourceDigest":String(repeating:"a",count:64),"membershipDigest":membership,"observationIDs":ids,"dataVersion":dataVersion,"partitionDigest":"","fittedStateDigests":[],"diagnostics":[]]]
        let bytes=try JSONSerialization.data(withJSONObject:table,options:[.sortedKeys,.withoutEscapingSlashes]),pin=WorkflowArchiveCodec.hash(bytes),source=root.appendingPathComponent("predictions.json")
        try bytes.write(to:source)
        let train=WorkflowNodeAddress(domain:.control,localID:"train"),view=WorkflowNodeAddress(domain:.data,localID:"inspect")
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:train,payload:.train(config:["runSetup":.object([:])])),.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(id:"prediction-inspection",from:.init(address:train,port:"predictions"),to:.init(address:view,port:"table"),kind:.table)])
        document.schemaVersion=17
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let before=try workspace.executionPolicy(for:document.projectID).sampleCapture
        try host.owner.setRunHistoryPolicy(.none)
        XCTAssertFalse(host.performanceCaptureEnabled)
        XCTAssertTrue(host.setPerformanceCaptureEnabled(true))
        XCTAssertTrue(host.owner.runHistoryPolicy.metricsAndDiagnostics)
        XCTAssertFalse(host.owner.runHistoryPolicy.componentOutputs,"Performance capture never widens component snapshots.")
        let after=try workspace.executionPolicy(for:document.projectID).sampleCapture
        XCTAssertTrue(after.performanceMeasurements);XCTAssertEqual(after.activationValues,before.activationValues);XCTAssertEqual(after.predictionRows,before.predictionRows)
        XCTAssertTrue(host.setPerformanceCaptureEnabled(false));XCTAssertFalse(host.performanceCaptureEnabled)
        let endpoint=WorkflowEndpoint(address:train,port:"predictions")
        let plan=try WorkflowPredictionPreparation(document:host.owner.document,pins:[endpoint:pin])
        let artifact=TableViewQuerySession.Artifact(indexURL:source,indexSHA256:pin,sourceSHA256:pin,inputKind:"table")
        let inputs=Dictionary(uniqueKeysWithValues:plan.sources.keys.map {($0,artifact)})
        workspace.preparationSampling = .init(mode:.rowCount,rowCount:3)
        let session=try WorkflowDataPreparationSession(workspace:workspace,owner:host.owner,bindings:host.bindings,destinations:host.outputDestinations,history:host.dataPreparationHistory,runtime:try XCTUnwrap(workspace.runtime),policy:workspace.executionPolicy(for:document.projectID),launcher:{_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)},predictionPlan:plan,predictionInputs:inputs)
        try session.start()
        XCTAssertFalse(host.setPerformanceCaptureEnabled(true),"An active capture owns its immutable policy.")
        let deadline=Date().addingTimeInterval(60)
        while session.isRunning && Date()<deadline {try await Task.sleep(for:.milliseconds(30))}
        XCTAssertEqual(session.phase,.completed,session.statusMessage+"\n"+session.messages.joined(separator:"\n"))
        XCTAssertEqual(session.executionCoverage?.isSampled,false,"View demands the complete prediction population despite working-data sampling.")
        let entry=try XCTUnwrap(host.dataPreparationHistory.entries.first)
        XCTAssertNotNil(entry.tableViews[view.localID]);_ = try await host.dataPreparationHistory.verify(entry)
        XCTAssertFalse(workspace.isBusy);XCTAssertFalse(host.owner.isBusy)
    }
}
