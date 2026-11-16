import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCanvasRenderEvidenceTests:XCTestCase {
    func testOneRenderBatchesEightyNodeCapturesAndRemainsImmutableUntilNextRender() throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:(0..<80).map {
            .init(address:.init(domain:.data,localID:"node-\($0)"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")]))
        })
        _=try library.saveSource(.workflow(document))
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        let adapter=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        let provider=RenderEvidenceProbe(projectID:owner.projectID,sessionID:owner.sessionID);adapter.outputProvider=provider
        let before=adapter.canvasRenderEvidence(validation:adapter.validation)
        XCTAssertEqual(provider.batchCalls,1);XCTAssertEqual(provider.individualCalls,0)
        XCTAssertEqual(before.statuses.count,80);XCTAssertTrue(before.statuses.values.allSatisfy {$0=="Captured"})
        for node in adapter.graph.nodes {XCTAssertEqual(before.badges[node.id]?["table"],"table · captured · 12 rows")}
        provider.current=false
        XCTAssertTrue(before.statuses.values.allSatisfy {$0=="Captured"},"A render is a value snapshot, never a mutable retained provider")
        let after=adapter.canvasRenderEvidence(validation:adapter.validation)
        XCTAssertEqual(provider.batchCalls,2);XCTAssertTrue(after.statuses.values.allSatisfy {$0=="Stale"})
        provider.foreign=true
        let foreign=adapter.canvasRenderEvidence(validation:adapter.validation)
        XCTAssertTrue(foreign.statuses.values.allSatisfy {$0=="Not evaluated"})
        adapter.invalidate()
        let calls=provider.batchCalls
        XCTAssertTrue(adapter.canvasRenderEvidence(validation:adapter.validation).statuses.values.allSatisfy {$0=="Not evaluated"})
        XCTAssertEqual(provider.batchCalls,calls,"Retired adapters cannot query captured evidence")
    }
}

@MainActor private final class RenderEvidenceProbe:WorkflowDataOutputs {
    let projectID:String;let sessionID:UUID;let runID=UUID()
    var batchCalls=0;var individualCalls=0;var current=true;var foreign=false
    init(projectID:String,sessionID:UUID) {self.projectID=projectID;self.sessionID=sessionID}
    var changes:AnyPublisher<Void,Never> {Empty().eraseToAnyPublisher()}
    func snapshot(for address:WorkflowNodeAddress)->WorkflowCapturedDataOutput? {individualCalls+=1;return nil}
    func snapshots(for addresses:Set<WorkflowNodeAddress>)->[WorkflowNodeAddress:WorkflowCapturedDataOutput] {
        batchCalls+=1
        return Dictionary(uniqueKeysWithValues:addresses.map {address in
            (address,.init(projectID:foreign ? "foreign":projectID,sessionID:sessionID,address:address,runID:runID,requestIdentity:"fixture",capturedRevision:0,bindingRevision:0,phase:.completed,matchesCurrentConstruction:current,matchesCurrentBindings:true,ports:["table":.object(["preview":.object(["rowCount":.integer(12)])])]))
        })
    }
}
