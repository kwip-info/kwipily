import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SchedulerLearningIntegrationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    @MainActor func testTypedTwoRecipeCanvasFixedSGDAndSchemaSevenUndoPendingText()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        for task in SchedulerLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue));let canvas=try XCTUnwrap(lesson.canvas)
            XCTAssertEqual(Set(canvas.registry.components.map(\.id)),[task.componentID,"optimizer_sgd","train"])
            for (i,type) in [task.componentID,"optimizer_sgd","train"].enumerated() {canvas.addNode(type:type,x:Double(i*100),y:0)}
            XCTAssertEqual(canvas.graph.schemaVersion,7)
            let scheduler=try XCTUnwrap(canvas.graph.nodes.first {$0.type == task.componentID}),optimizer=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "optimizer_sgd"}),train=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "train"})
            XCTAssertEqual(try OptimizerConfiguration(componentID:optimizer.type,values:optimizer.config),try OptimizerConfiguration(componentID:optimizer.type,values:SchedulerLearningTask.suppliedOptimizerValues))
            XCTAssertTrue(canvas.canConnect(source:scheduler.id,sourcePort:"out",target:train,port:"scheduler"));canvas.connect(source:scheduler.id,sourcePort:"out",target:train.id,targetPort:"scheduler")
            XCTAssertTrue(canvas.canConnect(source:optimizer.id,sourcePort:"out",target:train,port:"optimizer"));canvas.connect(source:optimizer.id,sourcePort:"out",target:train.id,targetPort:"optimizer")
            XCTAssertTrue(canvas.validation.isValid);XCTAssertTrue(canvas.validation.tensorLayouts.isEmpty)
            XCTAssertFalse(canvas.canConnect(source:scheduler.id,sourcePort:"out",target:train,port:"optimizer"))
            canvas.undo();XCTAssertEqual(canvas.graph.edges.count,1);XCTAssertEqual(canvas.graph.schemaVersion,7);canvas.redo();XCTAssertEqual(canvas.graph.edges.count,2);XCTAssertEqual(canvas.graph.schemaVersion,7)
            var altered=optimizer.config;altered["lr"] = .number(0.2);XCTAssertFalse(canvas.configure(nodeID:optimizer.id,values:altered));XCTAssertFalse(canvas.setPending(nodeID:optimizer.id,field:"lr",text:"0.2"))
            let key=task == .step ? "gamma":task == .cosine ? "eta_min":task == .linear ? "start_factor":task == .plateau ? "factor":"max_lr"
            XCTAssertTrue(canvas.setPending(nodeID:scheduler.id,field:key,text:"0.3"));XCTAssertThrowsError(try lesson.draft?.submissionData())
            XCTAssertEqual(try TensorLearningDraft.decode(try XCTUnwrap(lesson.draft).encoded()).pendingFields,lesson.draft?.pendingFields)
            canvas.discardPending(nodeID:scheduler.id,field:key);XCTAssertNil(canvas.pendingText(nodeID:scheduler.id,field:key))
            XCTAssertFalse(canvas.canAddType(task.componentID));XCTAssertFalse(canvas.canAddType("train"));XCTAssertFalse(canvas.canAddType("input"))
            XCTAssertTrue(canvas.statusMessage.contains("recipe") || canvas.statusMessage.contains("discarded"))
        }
    }
}
