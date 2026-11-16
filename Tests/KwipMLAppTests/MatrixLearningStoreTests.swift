import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MatrixLearningStoreTests:XCTestCase {
    @MainActor func testMatrixLaunchReadinessMeasurementAndReopen()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:MatrixLearningTask.broadcast.rawValue))
        let canvas=try XCTUnwrap(lesson.canvas)
        for type in ["input","input","matmul","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        guard nodes.count == 4 else {return XCTFail("Missing matrix construction")}
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[2].id,targetPort:"a")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"b")
        canvas.connect(source:nodes[2].id,sourcePort:"out",target:nodes[3].id,targetPort:"in")
        XCTAssertFalse(lesson.canSubmit)
        XCTAssertNotNil(lesson.runtimeReadinessError)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(lesson.canSubmit)
        XCTAssertNotNil(lesson.publicTask)
        lesson.submit()
        let deadline=Date().addingTimeInterval(30)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved)
        XCTAssertNotNil(lesson.formativeProjection)
        let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        XCTAssertTrue(workspace.beginProject(.init(name:"Other")))
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertEqual(lesson.grade?.status,.passed)
        XCTAssertTrue(lesson.resultIsCurrent)
        XCTAssertTrue(lesson.setMode(.assessment))
        XCTAssertNil(lesson.formativeProjection)
        lesson.submit()
        let assessmentDeadline=Date().addingTimeInterval(30)
        while workspace.isAssessingLearning && Date()<assessmentDeadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved)
        XCTAssertNil(lesson.formativeProjection)

    }

}
