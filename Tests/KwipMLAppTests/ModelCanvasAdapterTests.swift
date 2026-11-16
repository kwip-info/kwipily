import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelCanvasAdapterTests: XCTestCase {
    @MainActor func testCapturedProviderStatusHonorsDraftAndInvalidation() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        let root=try ModelCanvasAdapter(owner:owner)
        let provider: ([String],String)->WorkflowCapturedDataOutput? = { path,node in
            .init(projectID:id,sessionID:owner.sessionID,address:.init(domain:.model,localID:node),runID:UUID(),
                requestIdentity:"capture",capturedRevision:0,bindingRevision:0,phase:.completed,
                matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:["out":.object([:])])
        }
        root.capturedOutput=provider
        XCTAssertEqual(root.previewStatus(for:"features"),"Captured")
        XCTAssertEqual(root.previewStatus(for:"missing"),"Not evaluated")
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        let child=try ModelCanvasAdapter(owner:owner,path:["features"])
        child.capturedOutput=provider
        XCTAssertEqual(child.previewStatus(for:"branch-a"),"Stale")
        XCTAssertTrue(child.canvasEvidenceLabel(nodeID:"branch-a").contains("applied model"))
        child.invalidate()
        XCTAssertEqual(child.previewStatus(for:"branch-a"),"Not evaluated")
        XCTAssertNil(child.capturedOutput)
    }
    @MainActor func testRootAndChildShareOwnerAndStaleEditorsCannotWrite() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        let root=try ModelCanvasAdapter(owner:owner)
        XCTAssertEqual(root.graph.nodes.count,4)
        XCTAssertTrue(root.canOpenBlock("features"))
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        let child=try ModelCanvasAdapter(owner:owner,path:["features"])
        XCTAssertTrue(root.isBusy)
        XCTAssertFalse(child.isBusy)
        let before=root.graph
        child.selectNode("branch-a",extending:false)
        child.moveNodes(ids:["branch-a"],dx:25,dy:10)
        XCTAssertEqual(child.scope?.nodes.first(where:{$0.id == "branch-a"})?.x,240)
        let boundaryContext=child.boundaryRevision
        child.updateBoundary(context:boundaryContext) { $0.outputs[0].id="renamed-output" }
        child.updateBoundary(context:boundaryContext) { $0.title="Stale callback" }
        XCTAssertNotEqual(child.activeDefinition?.title,"Stale callback")
        child.updateBoundary(context:child.boundaryRevision) { $0.outputs[0].id="out" }
        XCTAssertEqual(root.graph,before)
        XCTAssertTrue(owner.applyDraft(lease:owner.lease(path:["features"])))
        XCTAssertFalse(root.isBusy)
        XCTAssertTrue(child.isBusy)
        XCTAssertEqual(child.scope?.nodes.first(where:{$0.id == "branch-a"})?.x,240)
        child.moveNodes(ids:["branch-a"],dx:10,dy:0)
        XCTAssertEqual(child.scope?.nodes.first(where:{$0.id == "branch-a"})?.x,240)
        root.invalidate()
        let saved=owner.document
        root.moveNodes(ids:["features"],dx:20,dy:0)
        XCTAssertEqual(owner.document,saved)
        XCTAssertFalse(root.historyContextIsActive)
    }
    @MainActor func testBoundaryContextSurvivesTypingAndNonstructuralEditsButRejectsChangedInterface() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        let child=try ModelCanvasAdapter(owner:owner,path:["features"]),context=child.boundaryRevision
        for title in ["M","My","My Block"] { child.updateBoundary(context:context) { $0.title=title } }
        XCTAssertEqual(child.activeDefinition?.title,"My Block")
        XCTAssertEqual(child.boundaryRevision,context)
        child.updateBoundary(context:context) { definition in
            definition.body.nodes[0].name="Renamed primitive"
            definition.body.nodes[0].x+=10
            definition.body.nodes[0].config["out_features"] = .integer(8)
            definition.parameters[0].field.defaultValue = .integer(8)
        }
        XCTAssertEqual(child.boundaryRevision,context)
        owner.setExecutionBusy(true);owner.setExecutionBusy(false)
        XCTAssertEqual(child.boundaryRevision,context)
        child.updateBoundary(context:context) { $0.outputs[0].id="features-output" }
        XCTAssertNotEqual(child.boundaryRevision,context)
        child.updateBoundary(context:context) { $0.title="Obsolete callback" }
        XCTAssertEqual(child.activeDefinition?.title,"My Block")
        let latest=child.boundaryRevision
        child.updateBoundary(context:latest) { $0.body.nodes[0].id="new-identity" }
        XCTAssertNotEqual(child.boundaryRevision,latest)
    }
    @MainActor func testTypedConfigurationAndFailedSaveKeepDisplayedSource() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),record=try library.save(ExampleFlows.parallelClassifier)
        let owner=try ModelAuthoringStore(library:library,projectID:record.id)
        let canvas=try ModelCanvasAdapter(owner:owner)
        canvas.selectNode("branch-a",extending:false)
        let stale=owner.lease()
        XCTAssertTrue(canvas.setConfigValue("out_features",value:.integer(12),nodeID:"branch-a",lease:stale))
        XCTAssertFalse(canvas.setConfigValue("out_features",value:.integer(14),nodeID:"branch-a",lease:stale))
        XCTAssertFalse(canvas.setConfigValue("bias",value:.integer(1),nodeID:"branch-a",lease:owner.lease()))
        XCTAssertEqual(canvas.graph.nodes.first(where:{$0.id == "branch-a"})?.config["out_features"],.integer(12))
        let before=canvas.graph,url=try library.documentURL(id:record.id)
        try FileManager.default.removeItem(at:url)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        canvas.addNode(type:"relu",x:100,y:100)
        XCTAssertEqual(canvas.graph,before)
        XCTAssertNotNil(owner.error)
    }
}
