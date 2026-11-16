import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPreparationFailureTests: XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testLargeCapturedFailureRowCountIsAcceptedWithinTableBudget() throws {
        let graph=DataFlowDocument(nodes:[.init(id:"columns",operatorID:"kwip.data.columns")])
        let detail:[String:Any] = ["schemaVersion":1,"code":"nodeExecution","nodeID":"columns","nodeLabel":"Columns","operatorID":"kwip.data.columns","componentTitle":"Columns","inputRowCount":500_001,"cause":"Invalid numeric cast","suggestion":"Choose a supported column type."]
        let decoded=try DataFlowFailureDetail.decode(JSONSerialization.data(withJSONObject:detail),graph:graph,labels:["columns":"Columns"],titles:["kwip.data.columns":"Columns"])
        XCTAssertEqual(decoded.inputRowCount,500_001)
    }
    func testRealColumnsCastFailurePreservesWorkerMessage() async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select real runner interpreter")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var document=try ColumnsPracticeExample.document()
        let address=WorkflowNodeAddress(domain:.data,localID:"columns")
        let index=try XCTUnwrap(document.nodes.firstIndex(where:{$0.address==address}))
        var configuration=ColumnsPracticeExample.configuration
        guard case .array(var rows)=configuration["rows"],case .object(var score)=rows[0] else {return XCTFail("practice rows missing")}
        score["targetType"] = .string("int64");rows[0] = .object(score);configuration["rows"] = .array(rows)
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.columns",config:configuration)
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("Sources/KwipMLApp/Resources/columns-observations.csv").path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(60)
        while host.dataPreparation?.isRunning==true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let session=try XCTUnwrap(host.dataPreparation)
        XCTAssertEqual(session.phase,.failed)
        XCTAssertTrue(session.statusMessage.contains("Columns row 1"),session.statusMessage)
        let failure=try XCTUnwrap(session.preparationFailure)
        XCTAssertEqual(failure.nodeID,"columns")
        XCTAssertTrue(session.preparationFailureIsCurrent)
        let display=try XCTUnwrap(host.canvas.projection.displayIDByAddress[address])
        let upstream=try XCTUnwrap(host.canvas.projection.displayIDByAddress[WorkflowExamples.sourceAddress])
        XCTAssertEqual(host.canvas.previewStatus(for:display),"Failed")
        XCTAssertEqual(host.canvas.canvasNodeFeedback(for:display).executionFailure,failure.cause)
        XCTAssertNotEqual(host.canvas.previewStatus(for:upstream),"Failed")
        let renderFailure=host.canvas.canvasRenderEvidence(validation:host.canvas.validation)
        XCTAssertEqual(renderFailure.statuses[display],"Failed")
        XCTAssertEqual(renderFailure.feedback[display]?.executionFailure,failure.cause)
        XCTAssertNotEqual(renderFailure.statuses[upstream],"Failed")
        XCTAssertNotNil(host.owner.apply(.move(addresses:[address],offset:.init(x:20,y:10)),lease:host.owner.lease()))
        XCTAssertTrue(session.preparationFailureIsCurrent,"Moving a block cannot invalidate runtime evidence.")
        XCTAssertNotNil(host.owner.apply(.configure(address:address,config:ColumnsPracticeExample.configuration),lease:host.owner.lease()))
        XCTAssertFalse(session.preparationFailureIsCurrent)
        XCTAssertNotNil(session.preparationFailure,"Retain explicitly stale diagnostic evidence.")
        XCTAssertNil(host.canvas.canvasNodeFeedback(for:display).executionFailure)
        XCTAssertNil(host.canvas.canvasRenderEvidence(validation:host.canvas.validation).feedback[display]?.executionFailure)
        XCTAssertTrue(host.run(),host.statusMessage)
        let retry=try XCTUnwrap(host.dataPreparation)
        XCTAssertFalse(session.preparationFailureIsCurrent,"Replaced sessions never regain current badges.")
        XCTAssertNil(retry.preparationFailure)
        let retryDeadline=Date().addingTimeInterval(60)
        while retry.isRunning && Date()<retryDeadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertEqual(retry.phase,.completed,retry.statusMessage)
        XCTAssertNil(retry.preparationFailure)
        XCTAssertNil(host.canvas.canvasNodeFeedback(for:display).executionFailure)
        XCTAssertNil(host.canvas.canvasRenderEvidence(validation:host.canvas.validation).feedback[display]?.executionFailure)
    }
    func testStructuredMetadataRejectsSpoofedUnknownAndUnattributedEvents() throws {
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let graph=DataFlowDocument(nodes:[.init(id:"columns",operatorID:"kwip.data.columns")])
        let detail:[String:JSONValue]=["schemaVersion":1,"code":.string("nodeExecution"),"nodeID":.string("columns"),"nodeLabel":.string("Captured label"),"operatorID":.string("kwip.data.columns"),"componentTitle":.string("Columns"),"inputRowCount":500001,"cause":.string("Bad integer"),"suggestion":.string("Choose a supported type.")]
        let event:[String:JSONValue]=["event":.string("error"),"message":.string("Original worker failure"),"dataflowError":.object(detail)]
        XCTAssertNotNil(try WorkflowPreparationFailure.detail(event:event,graph:graph,labels:["columns":"Captured label"],registry:registry))
        XCTAssertNil(try WorkflowPreparationFailure.detail(event:["event":.string("error"),"message":.string("Unattributed")],graph:graph,labels:[:],registry:registry))
        for (key,value) in [("nodeID",JSONValue.string("upstream")),("operatorID",.string("kwip.data.cast")),("nodeLabel",.string("Spoof")),("componentTitle",.string("Invented")),("inputRowCount",.integer(5000001)),("unknown",.bool(true))] {
            var changed=detail;changed[key]=value
            XCTAssertThrowsError(try WorkflowPreparationFailure.detail(event:["event":.string("error"),"dataflowError":.object(changed)],graph:graph,labels:["columns":"Captured label"],registry:registry),key)
        }
    }
    func testTrainingPreparationRetainsOnlyValidatedTerminalFailure() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let owner=try XCTUnwrap(workspace.workflowSession?.owner)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
        let source=root.appendingPathComponent("source.csv");try Data("signal,context,label\n1,2,0\n".utf8).write(to:source)
        var line:((String)->Void)?,exit:((Int32,String)->Void)?
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:try .capture(source)],directory:root.appendingPathComponent(UUID().uuidString),reservation:.init(acquire:{UUID()},release:{_ in}),launch:{_,_,_,onLine,onExit in line=onLine;exit=onExit;return WorkflowWorkerHandle(cancel:{})},onTerminal:{_ in})
        run.preparationFailureContextIsCurrent={true}
        XCTAssertTrue(run.start(),run.statusMessage)
        let deadline=Date().addingTimeInterval(10)
        while line == nil && Date()<deadline {try await Task.sleep(for:.milliseconds(10))}
        let send=try XCTUnwrap(line)
        let event:[String:Any]=["event":"error","message":"Original cast execution failure","dataflowError":["schemaVersion":1,"code":"nodeExecution","nodeID":"cast","nodeLabel":NSNull(),"operatorID":"kwip.data.columns","componentTitle":"Columns","inputRowCount":500001,"cause":"Invalid cast","suggestion":"Correct the source value."]]
        send(String(decoding:try JSONSerialization.data(withJSONObject:event),as:UTF8.self))
        XCTAssertNil(run.preparationFailure,"A live event is not a terminal failure.")
        try XCTUnwrap(exit)(1,"")
        while [.idle,.preparing,.training].contains(run.phase) && Date()<deadline {try await Task.sleep(for:.milliseconds(10))}
        XCTAssertEqual(run.phase,.failed,run.statusMessage)
        XCTAssertEqual(run.preparationFailure?.nodeID,"cast")
        XCTAssertTrue(run.preparationFailureIsCurrent)
        XCTAssertEqual(run.statusMessage,"Original cast execution failure")
        run.preparationFailureContextIsCurrent={false}
        XCTAssertFalse(run.preparationFailureIsCurrent,"Host replacement or binding change must hide current attribution.")
        XCTAssertNotNil(run.preparationFailure,"Stale evidence remains available.")
    }

    func testFailurePresentationUsesQualifiedAddressAndSeparatesStaleEvidence() throws {
        let graph=DataFlowDocument(nodes:[.init(id:"columns",operatorID:"kwip.data.columns")])
        let payload:[String:Any]=["schemaVersion":1,"code":"nodeExecution","nodeID":"columns","nodeLabel":NSNull(),"operatorID":"kwip.data.columns","componentTitle":"Columns","inputRowCount":6,"cause":"Bad cast","suggestion":"Correct the type."]
        let detail=try DataFlowFailureDetail.decode(JSONSerialization.data(withJSONObject:payload),graph:graph,labels:[:],titles:["kwip.data.columns":"Columns"])
        let current=WorkflowPreparationFailureEvidence(detail:detail,runID:UUID(),capturedRevision:7,isCurrent:true)
        XCTAssertEqual(current.currentCause(for:.init(domain:.data,localID:"columns")),"Bad cast")
        XCTAssertNil(current.currentCause(for:.init(domain:.model,localID:"columns")),"A same-named model node cannot inherit a data failure.")
        let stale=WorkflowPreparationFailureEvidence(detail:detail,runID:current.runID,capturedRevision:7,isCurrent:false)
        XCTAssertTrue(stale.matches(.init(domain:.data,localID:"columns")))
        XCTAssertNil(stale.currentCause(for:.init(domain:.data,localID:"columns")))
        XCTAssertEqual(stale.detail,current.detail)
    }

}
