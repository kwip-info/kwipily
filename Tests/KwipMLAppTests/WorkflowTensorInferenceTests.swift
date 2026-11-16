import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowTensorInferenceTests:XCTestCase {
    @MainActor func testShapesRefreshForConfigurationButNotSelectionOrMovement()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        var document=try WorkflowExamples.preparedParallelClassifier()
        // Reproduce an older saved layout overlapping the expanded container.
        let classifier=try XCTUnwrap(document.nodes.firstIndex {$0.address == .init(domain:.model,localID:"classifier")})
        document.nodes[classifier].x=600
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        let adapter=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        defer {adapter.invalidate()}
        for _ in 0..<200 {if !adapter.isInferringTensors {break};try await Task.sleep(nanoseconds:10_000_000)}
        XCTAssertFalse(adapter.isInferringTensors)
        let id=try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain:.model,localID:"classifier")])
        XCTAssertTrue(adapter.canvasPortDescriptor(nodeID:id,port:"out",direction:.output).declared.contains("[N, 2]"))
        let block=try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain:.model,localID:"features")])
        XCTAssertTrue(adapter.canvasPortDescriptor(nodeID:block,port:"out",direction:.output).declared.contains("[N, 16]"))
        let count=adapter.tensorInferenceCount
        adapter.selectNode(id,extending:false)
        adapter.moveNodes(ids:[id],dx:5,dy:7)
        XCTAssertEqual(adapter.tensorInferenceCount,count)
        XCTAssertTrue(adapter.configure(nodeID:id,config:["in_features":16,"out_features":3,"bias":true],lease:owner.lease()))
        XCTAssertGreaterThan(adapter.tensorInferenceCount,count)
        XCTAssertTrue(adapter.isInferringTensors)
        adapter.moveNodes(ids:[id],dx:17,dy:-9)
        XCTAssertTrue(adapter.apply(.rename(address:.init(domain:.model,localID:"classifier"),name:"Moved while inferring"),lease:owner.lease()))
        let pendingPosition=try XCTUnwrap(owner.document.nodes.first {$0.address.localID == "classifier"})
        // Invalid architecture must not retain the previous inferred shape while rebuilding.
        XCTAssertFalse(adapter.canvasPortDescriptor(nodeID:id,port:"out",direction:.output).declared.contains("[N, 2]"))
        for _ in 0..<200 {if !adapter.isInferringTensors {break};try await Task.sleep(nanoseconds:10_000_000)}
        XCTAssertFalse(adapter.isInferringTensors)
        XCTAssertFalse(adapter.canvasPortDescriptor(nodeID:id,port:"out",direction:.output).declared.contains("[N, 2]"))
        let published=try XCTUnwrap(adapter.graph.nodes.first {$0.id == id})
        XCTAssertEqual(published.name,pendingPosition.name)
        XCTAssertEqual(published.x,pendingPosition.x);XCTAssertEqual(published.y,pendingPosition.y)
    }
}
