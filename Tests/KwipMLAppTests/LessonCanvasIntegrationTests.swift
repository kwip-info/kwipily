import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LessonCanvasIntegrationTests:XCTestCase {
    func testFreshLessonCopyPreservesSavedBrokenAttemptAndBookmark() throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false))
        let original=try registry.openLesson(entryID:"model-output",source:source)
        defer {_=registry.close(original)}
        let workspace=original.session.workspace,host=try XCTUnwrap(workspace.workflowSession)
        var config=try ModelOutputConfiguration([:]);config.checkExpectedOutput=true;config.expectedSampleRank="exact";config.expectedSampleDimensions=[3]
        XCTAssertNotNil(host.owner.apply(.configure(address:ModelOutputPracticeExample.address,config:config.values),lease:host.owner.lease()))
        XCTAssertTrue(LessonSaveAction.save(workspace:workspace,entryID:"model-output",saveDocument:{original.session.save()}))
        let saved=try workspace.projectLibrary.loadSource(id:host.owner.projectID)
        let fresh=try registry.openLesson(entryID:"model-output",source:original.session,freshCopy:true)
        defer {_=registry.close(fresh)}
        let new=try XCTUnwrap(fresh.session.workspace.workflowSession)
        XCTAssertNotEqual(new.owner.projectID,host.owner.projectID)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:host.owner.projectID),saved)
        XCTAssertEqual(workspace.lessonReading.record("model-output").savedProjectID,host.owner.projectID)
        let node=try XCTUnwrap(new.owner.document.nodes.first {$0.address==ModelOutputPracticeExample.address})
        guard case .modelPrimitive(_,_,let values)=node.payload else {return XCTFail("Missing supplied output")}
        XCTAssertEqual(try ModelOutputConfiguration(values).expectedSampleDimensions,[2])
        XCTAssertEqual(fresh.session.workspace.lessonRailEntryID,"model-output")
    }
    func testPracticeOpensSeparateCanvasSavesAndReopensTheSameOwner() throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false))
        let original=source.workspace.graph
        let canvas=try registry.openLesson(entryID:"model-output",source:source)
        defer {_=registry.close(canvas)}
        XCTAssertEqual(source.workspace.graph,original)
        XCTAssertNil(source.workspace.activeProjectID)
        XCTAssertEqual(canvas.session.workspace.lessonRailEntryID,"model-output")
        let owner=try XCTUnwrap(canvas.session.workspace.workflowSession?.owner)
        XCTAssertFalse(owner.document.nodes.isEmpty)
        let documentBeforeFocus=owner.document
        let adapter=try XCTUnwrap(canvas.session.workspace.workflowSession?.canvas)
        let selectedBefore=adapter.selectedNodeIDs
        let follow=try XCTUnwrap(LessonContentCatalog.content(entryID:"model-output")?.steps.first {$0.title=="Follow the result"})
        canvas.session.workspace.lessonNodeFocus=follow.canvasFocus
        let highlighted=adapter.graph.nodes.filter {follow.canvasFocus.contains(nodeID:$0.id,type:adapter.canvasIconType(nodeID:$0.id))}
        XCTAssertEqual(highlighted.count,2)
        XCTAssertEqual(Set(highlighted.compactMap {try? WorkflowCanvasIDs.address(from:$0.id)}),Set([ModelOutputPracticeExample.address,.init(domain:.model,localID:"classifier")]))
        XCTAssertEqual(adapter.selectedNodeIDs,selectedBefore)
        XCTAssertEqual(owner.document,documentBeforeFocus)
        XCTAssertTrue(adapter.revealLessonNode(WorkflowExamples.sourceAddress))
        XCTAssertEqual(adapter.canvasRevealRequest?.nodeIDs,Set([try WorkflowCanvasIDs.nodeID(WorkflowExamples.sourceAddress)]))
        XCTAssertEqual(adapter.selectedNodeIDs,selectedBefore)
        XCTAssertEqual(owner.document,documentBeforeFocus)
        XCTAssertFalse(adapter.revealLessonNode(.init(domain:.data,localID:"missing")))
        canvas.session.workspace.showLesson(entryID:"model-output")
        XCTAssertEqual(canvas.session.workspace.lessonNodeFocus,follow.canvasFocus)
        canvas.session.workspace.showLesson(entryID:"columns")
        XCTAssertEqual(canvas.session.workspace.lessonNodeFocus,CanvasLessonFocus())
        canvas.session.workspace.showLesson(entryID:"model-output")

        XCTAssertTrue(LessonSaveAction.save(workspace:canvas.session.workspace,entryID:"model-output",saveDocument:{canvas.session.save()}))
        source.workspace.refreshProjects()
        XCTAssertTrue(source.workspace.lessonProjects.contains(where:{$0.id==owner.projectID}))
        let reopened=try registry.openLesson(entryID:"model-output",source:source)
        XCTAssertTrue(reopened===canvas)
        XCTAssertTrue(registry.focusLessonCanvas(preferred:canvas.session))
        XCTAssertTrue(canvas.session.workspace.showLessonRail)
        XCTAssertTrue(source.workspace.lessonReading.unsave(projectID:owner.projectID))
        source.workspace.refreshProjects()
        let preserved=try XCTUnwrap(source.workspace.projects.first {$0.id==owner.projectID})
        XCTAssertTrue(try registry.open(preserved,section:nil)===canvas)
        XCTAssertEqual(canvas.session.workspace.workflowSession?.owner.document,owner.document)
    }
    func testSaveRefusalCannotCreateASavedLessonAndWindowIconsDistinguishTypes() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(storageOverride:root,detectCapabilities:false)
        workspace.newProject();workspace.showLesson(entryID:"guided")
        XCTAssertFalse(LessonSaveAction.save(workspace:workspace,entryID:"guided",saveDocument:{false}))
        XCTAssertNil(workspace.lessonReading.record("guided").savedProjectID)
        let kinds=["Guided","Example","Practice","Construction"]
        let icons=kinds.map {DocumentWindowIdentity.project(url:nil,isLesson:true,lessonKind:$0,isExternal:false,isManaged:true).symbol}
        XCTAssertEqual(Set(icons).count,4)
        for icon in icons {XCTAssertNotNil(NSImage(systemSymbolName:icon,accessibilityDescription:nil))}
    }
    func testConstructionFamilyLaunchesItsRestrictedCheckerCanvas() throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(storageOverride:root,detectCapabilities:false))
        let controller=try registry.openLesson(entryID:"tensor-layout",source:source)
        defer {_=registry.close(controller)}
        XCTAssertTrue(controller.session.tensor.isActive)
        XCTAssertNil(controller.session.workspace.workflowSession)
        XCTAssertEqual(controller.session.workspace.lessonRailEntryID,"tensor-layout")
        XCTAssertNotNil(controller.session.tensor.canvas)
        XCTAssertEqual(controller.session.tensor.draft?.operation,"reshape")
        XCTAssertEqual(controller.session.tensor.draft?.mode,.assessment)
    }
    func testLibraryCanOpenSpecificAssessmentAndRejectCrossFamilyTask() throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(storageOverride:root,detectCapabilities:false))
        let controller=try registry.openLesson(entryID:"tensor-layout",operation:"transpose",source:source)
        defer {_=registry.close(controller)}
        XCTAssertEqual(controller.session.tensor.draft?.operation,"transpose")
        XCTAssertEqual(controller.session.tensor.draft?.mode,.assessment)
        XCTAssertNil(source.workspace.activeProjectID)
        XCTAssertThrowsError(try registry.openLesson(entryID:"tensor-layout",operation:"softmax-last-axis",source:source))
        XCTAssertEqual(controller.session.tensor.draft?.operation,"transpose")
    }
    func testAffineChildrenLaunchExactTasksAndFreshRecoveryPreservesItsObservations()throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(storageOverride:root,detectCapabilities:false))
        XCTAssertEqual(ConstructionLessonCatalog.groups["affine"]?.count,9)
        for choice in ConstructionLessonCatalog.affineChoices {
            let controller=try registry.openLesson(entryID:"affine",operation:choice.id,source:source)
            XCTAssertEqual(controller.session.affine.lessonTask?.identity.assessmentID,choice.id)
            XCTAssertEqual(controller.session.affine.mode,.assessment)
            XCTAssertFalse(controller.session.tensor.isActive)
            if choice.id == ModelConstructionLessonIdentity.recovery(taskSeed:0).assessmentID {
                let before=try XCTUnwrap(controller.session.affine.lessonTask)
                let fresh=try registry.startFreshLesson(entryID:"affine",workspace:controller.session.workspace)
                XCTAssertEqual(fresh.session.affine.lessonTask?.identity,before.identity)
                XCTAssertEqual(fresh.session.affine.lessonTask?.probes,before.probes)
                XCTAssertNotEqual(fresh.session.workspace.activeProjectID,controller.session.workspace.activeProjectID)
                _=registry.close(fresh)
            }
            _=registry.close(controller)
        }
        XCTAssertThrowsError(try registry.openLesson(entryID:"affine",operation:"transpose",source:source))
        XCTAssertNil(source.workspace.activeProjectID)
    }
    func testEveryDiscoverableEntryHasFiniteSharedContentAndStableHashes() {
        for item in LearningLibraryPresentation.items {
            guard let content=LessonContentCatalog.content(entryID:item.id) else {XCTFail("No content for \(item.id)");continue}
            XCTAssertFalse(content.steps.isEmpty,item.id)
            XCTAssertEqual(content.steps.count,Set(content.steps.map(\.id)).count,item.id)
            for step in content.steps {
                XCTAssertFalse(step.blocks.isEmpty,item.id)
                let expected=step.revision
                for _ in 0..<5 {XCTAssertEqual(step.revision,expected,item.id)}
            }
        }
    }
}
