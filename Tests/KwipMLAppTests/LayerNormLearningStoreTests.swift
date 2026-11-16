import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LayerNormLearningStoreTests:XCTestCase {
    @MainActor func testAffineLessonReadinessGradingStalenessAndRestore() async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable explicit real runtime")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:path),storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:LayerNormLearningTask.affine.rawValue))
        guard let canvas=lesson.canvas else {return XCTFail("Normalization lesson did not start")}
        for type in ["input","layer_norm","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        guard nodes.count == 3 else {return XCTFail("Missing normalization construction")}
        let id=nodes[1].id
        var config=nodes[1].config;config["normalized_shape"] = .array([3]);config["eps"] = .number(0.125)
        XCTAssertTrue(canvas.configure(nodeID:id,values:config))
        let task=LayerNormLearningTask.affine
        var params=["weight":try XCTUnwrap(task.scale),"bias":try XCTUnwrap(task.bias)]
        XCTAssertTrue(canvas.setNormalizationParameters(nodeID:id,values:params));XCTAssertTrue(canvas.setGradientTarget(id))
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:id,targetPort:"in")
        canvas.connect(source:id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertFalse(lesson.canSubmit);XCTAssertNotNil(lesson.runtimeReadinessError)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertNil(lesson.runtimeReadinessError);XCTAssertTrue(lesson.canSubmit)
        XCTAssertNotNil(lesson.publicTask)
        lesson.submit();try await finished(workspace)
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertEqual(lesson.grade?.outcomes.count,4)
        XCTAssertTrue(lesson.resultSaved);XCTAssertNotNil(lesson.formativeProjection)
        params["weight"]!.values[0]+=1
        XCTAssertTrue(canvas.setNormalizationParameters(nodeID:id,values:params))
        XCTAssertFalse(lesson.resultIsCurrent);XCTAssertNil(lesson.formativeProjection)
        canvas.undo();XCTAssertTrue(lesson.resultIsCurrent)
        let projectID=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        XCTAssertTrue(workspace.beginProject(.init(name:"Other")))
        XCTAssertTrue(workspace.beginProject(graph,id:projectID))
        XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultIsCurrent)
        XCTAssertTrue(lesson.setMode(.assessment));XCTAssertNil(lesson.formativeProjection)
        lesson.submit();try await finished(workspace)
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved);XCTAssertNil(lesson.formativeProjection)
    }
    @MainActor private func finished(_ workspace:WorkspaceStore) async throws {
        let deadline=Date().addingTimeInterval(30)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertFalse(workspace.isAssessingLearning)
    }
}
