import XCTest
import SwiftUI
import AppKit
import Combine
import KwipMLCore
@testable import KwipMLApp

/// Full workbench, rather than an isolated two-node canvas: camera updates must
/// move the installed native surface without invalidating all surrounding panes.
@MainActor final class WorkflowCameraHostingTests:XCTestCase {
    func testEightyNodeWorkbenchCameraMovesWithoutWholeWorkbenchNotifications() async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let nodes:[WorkflowNode]=(0..<80).map {index in
            let configuration:[String:JSONValue]=index==0 ? ["separator":.string(",")]:[:]
            let payload=WorkflowNodePayload.dataOperator(operatorID:index==0 ? "kwip.data.sourceCSV":"kwip.data.view",config:configuration)
            return WorkflowNode(address:.init(domain:.data,localID:"node-\(index)"),name:"Node \(index)",x:Double(index%8)*280,y:Double(index/8)*180,payload:payload)
        }
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:nodes,edges:(1..<80).map {index in
            .init(id:"edge-\(index)",from:.init(address:nodes[index-1].address,port:"table"),to:.init(address:nodes[index].address,port:"table"),kind:.table)
        })
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let session=try XCTUnwrap(workspace.workflowSession)
        session.canvas.canvasViewState = .init(zoom:0.25)
        let observation=workspace.installWorkflowCanvasPersistence(session)
        defer {observation.cancel();session.retire()}
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:1200,height:800),styleMask:[.titled],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false
        let view=NSHostingView(rootView:WorkflowWorkbenchView(owner:session.owner,canvas:session.canvas,sources:session.sourceProvider,outputs:session,executionHost:session,workspace:workspace))
        window.contentView=view;window.orderFront(nil)
        defer {window.orderOut(nil);window.close()}
        func find(_ node:NSView)->CanvasScrollProbe? {
            if let probe=node as? CanvasScrollProbe {return probe}
            for child in node.subviews {if let probe=find(child) {return probe}}
            return nil
        }
        let deadline=Date().addingTimeInterval(10)
        while (find(view)?.controller?.origin == nil || session.canvas.isInferringTensors) && Date()<deadline {
            view.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(50))
        }
        view.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(100));view.layoutSubtreeIfNeeded()
        let probe=try XCTUnwrap(find(view)),camera=try XCTUnwrap(probe.controller)
        let start=try XCTUnwrap(camera.origin)
        var canvasChanges=0,workspaceChanges=0
        let a=session.canvas.objectWillChange.sink {_ in canvasChanges+=1}
        let b=workspace.objectWillChange.sink {_ in workspaceChanges+=1}
        defer {a.cancel();b.cancel()}
        var layoutMS:[Double]=[]
        InteractionProfiler.shared.start()
        for _ in 0..<24 {
            XCTAssertTrue(camera.input.routeWheel(point:CGPoint(x:300,y:220),inside:true,deltaX:8,deltaY:0,precise:true,momentum:false))
            let begin=Date()
            view.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(16));view.layoutSubtreeIfNeeded()
            layoutMS.append(Date().timeIntervalSince(begin)*1000)
        }
        let profile=InteractionProfiler.shared.stop()
        let sceneBuilds=profile.samples.filter {$0.metric == .canvasScene}.count
        XCTAssertLessThanOrEqual(sceneBuilds,4,"Local camera translation must not reconstruct the full 80-node scene for every packet")
        XCTAssertEqual(try XCTUnwrap(camera.origin).x,start.x-192,accuracy:0.001)
        XCTAssertEqual(camera.origin?.y,start.y)
        XCTAssertEqual(canvasChanges,0)
        XCTAssertEqual(workspaceChanges,0)
        InteractionProfiler.shared.start()
        var zoomMS:[Double]=[]
        for _ in 0..<24 {
            let pointer=CGPoint(x:300,y:220)
            let expected=CanvasCameraGeometry.zoom(origin:try XCTUnwrap(camera.origin),zoom:session.canvas.canvasViewState.zoom,pointer:pointer,delta:0.5,precise:true,reservedTop:camera.reservedTop)
            XCTAssertTrue(camera.input.routeWheel(point:pointer,inside:true,deltaX:0,deltaY:0.5,precise:true,momentum:false))
            XCTAssertEqual(session.canvas.canvasViewState.zoom,expected.zoom,accuracy:0.000001)
            XCTAssertEqual(try XCTUnwrap(camera.origin).x,expected.origin.x,accuracy:0.000001)
            let begin=Date();view.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(16));view.layoutSubtreeIfNeeded()
            zoomMS.append(Date().timeIntervalSince(begin)*1000)
        }
        let zoomProfile=InteractionProfiler.shared.stop()
        let zoomBuilds=zoomProfile.samples.filter {$0.metric == .canvasScene}.count
        XCTAssertLessThanOrEqual(zoomBuilds,4,"Scale transforms must not reconstruct the full scene for each small wheel packet")
        XCTAssertEqual(canvasChanges,0);XCTAssertEqual(workspaceChanges,0)
        let packetCanvasChanges=canvasChanges,packetWorkspaceChanges=workspaceChanges
        XCTAssertTrue(workspace.flushWorkflowCanvasBeforeTransition())
        let saved=try XCTUnwrap(workspace.arrangementLibrary.projectState(projectID:document.projectID))
        XCTAssertEqual(saved.modelCanvas.scrollX,try XCTUnwrap(camera.origin).x,accuracy:0.001)
        XCTAssertEqual(saved.modelCanvas.zoom,session.canvas.canvasViewState.zoom,accuracy:0.000001)
        XCTAssertEqual(session.owner.document,document)
        print("80-node zoom packets=24 sceneBuilds=\(zoomBuilds) layout+16ms-yield median=\(zoomMS.sorted()[12]) max=\(zoomMS.max() ?? 0) ms; not FPS")
        print("80-node workbench native horizontal camera packets=24 canvasNotifications=\(packetCanvasChanges) workspaceNotifications=\(packetWorkspaceChanges) sceneBuilds=\(sceneBuilds) layout+16ms-yield median=\(layoutMS.sorted()[12]) max=\(layoutMS.max() ?? 0) ms; not FPS")
    }
}
