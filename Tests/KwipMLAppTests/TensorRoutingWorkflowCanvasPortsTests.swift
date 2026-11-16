import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorRoutingWorkflowCanvasPortsTests: XCTestCase {
    @MainActor private func adapter(document: WorkflowDocument) throws -> (URL, WorkflowAuthoringStore, WorkflowCanvasAdapter) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:root)
        _=try library.saveSource(.workflow(document))
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        return(root,owner,try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data))
    }
    @MainActor private func settled(_ adapter:WorkflowCanvasAdapter) async throws {
        for _ in 0..<500 {
            if !adapter.isInferringTensors {return}
            try await Task.sleep(nanoseconds:10_000_000)
        }
        XCTFail("Workflow tensor inference did not finish within five seconds")
    }
    @MainActor func testPreparedChunkUsesFiveResolvedChoicesAndKeepsMissingSavedPartAsRepairMetadata() async throws {
        var document=try WorkflowExamples.preparedChunkClassifier()
        let routing=WorkflowNodeAddress(domain:.model,localID:"routing")
        let orphan=WorkflowNodeAddress(domain:.model,localID:"repair-target")
        document.nodes.append(.init(address:orphan,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true])))
        document.edges.append(.init(id:"saved-part-seven",from:.init(address:routing,port:"part7"),to:.init(address:orphan,port:"in"),kind:.tensor))
        let (root,owner,canvas)=try adapter(document:document)
        defer {canvas.invalidate();try? FileManager.default.removeItem(at:root)}
        try await settled(canvas)
        let display=try XCTUnwrap(canvas.projection.displayIDByAddress[routing])
        let node=try XCTUnwrap(canvas.graph.nodes.first {$0.id == display})
        let definition=try XCTUnwrap(canvas.registry.component(type:node.type))
        XCTAssertEqual(definition.outputs,["part0","part1","part2","part3","part4"],"The universal canvas must expose native Chunk's actual five parts, not its declared ceiling")
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:display,port:"part4",direction:.output).kind,.tensor)
        XCTAssertTrue(canvas.canvasPortDescriptor(nodeID:display,port:"part4",direction:.output).declared.contains("[N, 2, 1]"))
        for missing in ["part5","part6","part7"] {
            XCTAssertFalse(definition.outputs.contains(missing),"Unavailable ports must not be enabled connection choices")
            XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:display,port:missing,direction:.output).kind,.unknown)
        }
        let repair=try XCTUnwrap(canvas.projection.portDescriptor(address:routing,port:"part7",direction:.output))
        XCTAssertNil(repair.contract)
        XCTAssertNotNil(repair.unresolvedReason)
        let saved=try XCTUnwrap(canvas.graph.edges.first {canvas.projection.edgeIDByDisplayID[$0.id] == "saved-part-seven"})
        XCTAssertEqual(saved.sourcePort,"part7")
        let geometry=CanvasGeometrySnapshot(registry:canvas.registry,presentation:.init(),graph:canvas.graph,validation:canvas.validation)
        XCTAssertNotEqual(geometry.portAnchor(nodeID:display,output:true,port:"part7"),geometry.portAnchor(nodeID:display,output:true,port:"part0"))
        XCTAssertEqual(owner.document,document,"Inference and presentation must preserve saved authoring bytes")
        let inferenceCount=canvas.tensorInferenceCount
        canvas.moveNodes(ids:[display],dx:7,dy:9)
        XCTAssertEqual(canvas.tensorInferenceCount,inferenceCount,"Layout-only edits must reuse configured port inference")
        let moved=try XCTUnwrap(canvas.graph.nodes.first {$0.id == display})
        XCTAssertEqual(canvas.registry.component(type:moved.type)?.outputs,["part0","part1","part2","part3","part4"])
        XCTAssertEqual(owner.document.edges,document.edges)
        let ports=canvas.projection.ports
        XCTAssertTrue(canvas.apply(.rename(address:routing,name:"My chunk"),lease:owner.lease()))
        XCTAssertEqual(canvas.projection.ports,ports)
        XCTAssertEqual(canvas.projection.portDescriptor(address:routing,port:"part7",direction:.output),repair)
        canvas.undo()
        XCTAssertEqual(canvas.projection.ports,ports)
        XCTAssertEqual(canvas.tensorInferenceCount,inferenceCount)
        XCTAssertEqual(canvas.projection.portDescriptor(address:routing,port:"part7",direction:.output),repair)
    }
}
