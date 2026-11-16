import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorLearningStoreTests:XCTestCase {
    @MainActor func testReadableProjectTitleDoesNotOverwriteRenamedProjectOnRestore() throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:"softmax-first-sample-axis"))
        XCTAssertEqual(workspace.graph.name,"Learn · Softmax · normalize columns")
        let id=try XCTUnwrap(workspace.activeProjectID)
        var renamed=workspace.graph
        renamed.name="My probability experiment"
        XCTAssertTrue(workspace.beginProject(renamed,id:id))
        lesson.restoreEditor()
        XCTAssertEqual(workspace.graph.name,"My probability experiment")
        XCTAssertEqual(lesson.draft?.operation,"softmax-first-sample-axis")
        XCTAssertTrue(lesson.isActive)
    }
    @MainActor func testSoftmaxLaunchReadinessMeasurementAndReopen()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:"softmax-last-axis"))
        let canvas=try XCTUnwrap(lesson.canvas)
        for type in ["input","softmax","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        guard nodes.count == 3 else {return XCTFail("Missing Softmax construction")}
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
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

    @MainActor func testEmbeddingLaunchReadinessMeasurementAndReopen()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:"embedding-padding-frequency"))
        XCTAssertFalse(lesson.statusMessage.contains("input gradients"))
        XCTAssertTrue(lesson.statusMessage.contains("table"))
        let canvas=try XCTUnwrap(lesson.canvas)
        for type in ["input","embedding","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        guard nodes.count == 3 else {return XCTFail("Missing embedding construction")}
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertTrue(canvas.configure(nodeID:nodes[1].id,values:try EmbeddingLearningTask.paddingFrequency.configuration.values))
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

    @MainActor func testCreateEditReopenAndOrdinaryRunProtection() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:"transpose"))
        let id=try XCTUnwrap(workspace.activeProjectID)
        XCTAssertTrue(lesson.isActive)
        XCTAssertEqual(workspace.lessonContext?.restoreState,.ready)
        XCTAssertFalse(workspace.canRunCurrentFlow)
        XCTAssertFalse(lesson.canSubmit)
        let canvas=try XCTUnwrap(lesson.canvas)
        var busyUpdates=0
        let busyObservation=canvas.objectWillChange.sink {busyUpdates+=1}
        workspace.setLearningAssessmentActive(true)
        XCTAssertGreaterThan(busyUpdates,0)
        workspace.setLearningAssessmentActive(false)
        busyObservation.cancel()
        canvas.addNode(type:"input",x:0,y:0)
        XCTAssertTrue(lesson.setMode(.practice))
        XCTAssertTrue(workspace.beginProject(.init(name:"Ordinary")))
        XCTAssertFalse(lesson.isActive)
        XCTAssertTrue(workspace.beginProject(.init(name:"Tensor"),id:id))
        XCTAssertTrue(lesson.isActive)
        XCTAssertEqual(lesson.draft?.mode,.practice)
        XCTAssertEqual(lesson.canvas?.graph.nodes.count,1)
        let current=workspace.activeProjectID
        XCTAssertFalse(lesson.start(operation:"invalid"))
        XCTAssertEqual(workspace.activeProjectID,current)
    }
    @MainActor func testRealMeasuredSubmissionAndResultStaleness() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:"transpose"))
        let canvas=try XCTUnwrap(lesson.canvas)
        for type in ["input","transpose","output"] {canvas.addNode(type:type,x:0,y:0)}
        var canvasUpdates=0
        let subscription=canvas.objectWillChange.sink {canvasUpdates+=1}
        defer {subscription.cancel()}
        let nodes=canvas.graph.nodes
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertTrue(lesson.canSubmit)
        let beforeRun=canvasUpdates
        lesson.submit()
        let deadline=Date().addingTimeInterval(45)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertGreaterThan(canvasUpdates,beforeRun)
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultIsCurrent)
        XCTAssertEqual(canvas.previewStatus(for:nodes[0].id),"Fixture input")
        XCTAssertEqual(canvas.previewStatus(for:nodes[1].id),"Not captured")
        XCTAssertEqual(canvas.previewStatus(for:nodes[2].id),"Measured output")
        XCTAssertEqual(lesson.formativeProjection?.fixtures.count,4)
        XCTAssertTrue(canvas.configure(nodeID:nodes[1].id,values:["dim0":1,"dim1":1]))
        XCTAssertFalse(lesson.resultIsCurrent)
        XCTAssertEqual(canvas.previewStatus(for:nodes[2].id),"Stale")
        XCTAssertNil(lesson.formativeProjection)
        canvas.undo();XCTAssertTrue(lesson.resultIsCurrent)
        XCTAssertTrue(lesson.resultSaved)
        let projectID=try XCTUnwrap(workspace.activeProjectID),savedGrade=lesson.grade
        let graph=workspace.graph
        XCTAssertTrue(workspace.beginProject(.init(name:"Another project")))
        XCTAssertTrue(workspace.beginProject(graph,id:projectID))
        XCTAssertEqual(lesson.grade,savedGrade)
        XCTAssertTrue(lesson.resultSaved)
        XCTAssertTrue(lesson.resultIsCurrent)
        XCTAssertEqual(lesson.formativeProjection?.fixtures.count,4)
        let pointer=root.appendingPathComponent("learning/projects/\(projectID)/tensor-receipts/latest.json")
        try FileManager.default.removeItem(at:pointer)
        try FileManager.default.createDirectory(at:pointer,withIntermediateDirectories:false)
        lesson.submit()
        let secondDeadline=Date().addingTimeInterval(45)
        while workspace.isAssessingLearning && Date()<secondDeadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertEqual(lesson.grade?.status,.passed)
        XCTAssertFalse(lesson.resultSaved)
        XCTAssertTrue(lesson.statusMessage.contains("could not be saved"))
        XCTAssertTrue(workspace.beginProject(.init(name:"Another project")))
        XCTAssertTrue(workspace.beginProject(graph,id:projectID))
        XCTAssertTrue(lesson.isActive)
        XCTAssertNil(lesson.grade)
        XCTAssertFalse(lesson.resultSaved)
        XCTAssertTrue(lesson.statusMessage.contains("could not be verified"))
    }
}
