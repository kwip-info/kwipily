import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowDataOutputBatchTests:XCTestCase {
    func testDefaultBatchFallbackRequestsEachQualifiedAddressOnceAndOmitsMissingEvidence() {
        let source=BatchFallbackProvider()
        let data=WorkflowNodeAddress(domain:.data,localID:"same")
        let model=WorkflowNodeAddress(domain:.model,localID:"same")
        let missing=WorkflowNodeAddress(domain:.data,localID:"missing")
        let values=source.snapshots(for:[data,model,missing])
        XCTAssertEqual(Set(source.requests),[data,model,missing])
        XCTAssertEqual(source.requests.count,3)
        XCTAssertEqual(Set(values.keys),[data,model])
        XCTAssertEqual(values[data]?.ports,["table":.string("data")])
        XCTAssertEqual(values[model]?.ports,["tensor":.string("model")])
        XCTAssertTrue(source.snapshots(for:[]).isEmpty)
        XCTAssertEqual(source.requests.count,3)
        source.crossAddress=true
        XCTAssertTrue(source.snapshots(for:[data]).isEmpty,"Fallback must reject a provider's foreign qualified address.")
    }
    func testManagedPreparationBatchMatchesIndividualEvidenceAndRejectsRetiredHost() async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the isolated runner interpreter.")}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let graph=try DataFlowCodec.decodeGraph(Data(contentsOf:repo.appendingPathComponent("runner/dataflow_examples/data02-calculate.json")))
        let document=WorkflowDocument(projectID:UUID().uuidString,name:"Batch observations",nodes:graph.nodes.map {
            .init(address:.init(domain:.data,localID:$0.id),name:$0.id,payload:.dataOperator(operatorID:$0.operatorID,config:$0.config))
        },edges:graph.edges.enumerated().map { index,edge in
            .init(id:"edge-\(index)",from:.init(address:.init(domain:.data,localID:edge.from.node),port:edge.from.port),to:.init(address:.init(domain:.data,localID:edge.to.node),port:edge.to.port),kind:.table)
        })
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let source=WorkflowNodeAddress(domain:.data,localID:"source")
        XCTAssertTrue(host.bindings.setPath(repo.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(60)
        while host.dataPreparation?.isRunning == true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let session=try XCTUnwrap(host.currentDataPreparation)
        XCTAssertEqual(session.phase,.completed,session.statusMessage+"\n"+session.messages.joined(separator:"\n"))
        let addresses=Set(document.nodes.map(\.address))
        let requested=addresses.union([.init(domain:.model,localID:"source"),.init(domain:.data,localID:"missing")])
        let batch=host.snapshots(for:requested)
        XCTAssertEqual(Set(batch.keys),addresses)
        for address in addresses {
            XCTAssertEqual(batch[address],host.snapshot(for:address))
            XCTAssertEqual(session.snapshots(for:requested)[address],session.snapshot(for:address))
            XCTAssertEqual(batch[address]?.dataCompleteness,"Complete data")
            XCTAssertNotNil(batch[address]?.capturedAt)
        }
        XCTAssertNil(batch[.init(domain:.model,localID:"source")],"A model local ID must never alias a data observation.")
        let render=host.canvas.canvasRenderEvidence(validation:host.canvas.validation)
        for node in host.canvas.graph.nodes {XCTAssertEqual(render.statuses[node.id],host.canvas.previewStatus(for:node.id))}
        let before=try XCTUnwrap(batch[source])
        XCTAssertTrue(host.bindings.setPath(nil,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let stale=host.snapshots(for:requested)
        XCTAssertEqual(stale[source]?.ports,before.ports)
        XCTAssertEqual(stale[source]?.matchesCurrentBindings,false)
        host.retire()
        XCTAssertTrue(host.snapshots(for:requested).isEmpty)
        XCTAssertNil(host.snapshot(for:source))
    }
}

@MainActor private final class BatchFallbackProvider:WorkflowDataOutputs {
    let projectID="batch-fixture"
    let sessionID=UUID()
    let runID=UUID()
    var requests:[WorkflowNodeAddress]=[]
    var crossAddress=false
    var changes:AnyPublisher<Void,Never> {Empty().eraseToAnyPublisher()}
    func snapshot(for address:WorkflowNodeAddress)->WorkflowCapturedDataOutput? {
        requests.append(address)
        guard address.localID == "same" else {return nil}
        return .init(projectID:projectID,sessionID:sessionID,address:crossAddress ? .init(domain:.control,localID:address.localID):address,runID:runID,requestIdentity:"captured",capturedRevision:1,bindingRevision:1,phase:.cancelled,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:address.domain == .data ? ["table":.string("data")]:["tensor":.string("model")])
    }
}
