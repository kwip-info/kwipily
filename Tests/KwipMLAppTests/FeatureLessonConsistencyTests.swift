import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FeatureLessonConsistencyTests:XCTestCase {
    func testEveryAuthoredFocusAndLinkResolvesInItsActualFactory()throws {
        let expression=try NSRegularExpression(pattern:"kwip-node://[^)\\s]+")
        for (id,targets) in LessonCanvasReferences.addresses.sorted(by:{$0.key<$1.key}) {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:id),id)
            let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
            let addresses=Set(document.nodes.map(\.address))
            XCTAssertFalse(document.edges.isEmpty,id)
            XCTAssertEqual(Set(content.steps.map(\.id)),Set(targets.keys),"Missing or stale per-step focus: "+id)
            for step in content.steps {
                XCTAssertFalse(step.nodeFocus?.addresses.isEmpty ?? true,id+" "+step.id)
                for address in step.nodeFocus?.addresses ?? [] {
                    XCTAssertTrue(addresses.contains(address),id+" "+step.id+" "+address.localID)
                }
                XCTAssertEqual(Set(step.blocks.map(\.id)).count,step.blocks.count,id+" "+step.id)
                for block in step.blocks {
                    let text=block.text as NSString
                    for match in expression.matches(in:block.text,range:NSRange(location:0,length:text.length)) {
                        let url=try XCTUnwrap(URL(string:text.substring(with:match.range)))
                        let address=try XCTUnwrap(LessonNodeLink.address(url),id)
                        XCTAssertTrue(addresses.contains(address),id+" "+step.id+" "+url.absoluteString)
                    }
                }
            }
        }
    }
    func testFocusUsesTheStepContextAndPreservesExplicitAuthorshipAndReadingRevision()throws {
        let media=try XCTUnwrap(LessonContentCatalog.content(entryID:"prepare-media"))
        let image=try XCTUnwrap(media.steps.first {$0.id=="try.1"}?.nodeFocus)
        let audio=try XCTUnwrap(media.steps.first {$0.id=="try.2"}?.nodeFocus)
        XCTAssertNotEqual(image.addresses,audio.addresses)
        XCTAssertTrue(image.addresses.contains(.init(domain:.data,localID:"image-view")))
        XCTAssertTrue(audio.addresses.contains(.init(domain:.data,localID:"audio-view")))
        var step=LessonContent.step("try.1","Focus","Read the image")
        let revision=step.revision
        step.nodeFocus = .init(addresses:[.init(domain:.data,localID:"authored-target")])
        let content=LessonContent(entryID:"prepare-media",title:"Media",kind:"Practice",steps:[step])
        let annotated=LessonCanvasReferences.annotate(content)
        XCTAssertEqual(annotated.steps[0].nodeFocus,step.nodeFocus)
        XCTAssertEqual(annotated.steps[0].revision,revision)
    }
    func testNewPracticesSeparateReflectionFromLimitsAndRemainUnscored()throws {
        for id in ["prepare-media","document-classification","extract-text","game-rules","game-episodes","transcribe-audio","classify-audio","prebuilt-models","typed-decisions"] {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:id),id)
            let inspection=try XCTUnwrap(content.steps.first {$0.id=="inspect"},id)
            XCTAssertEqual(inspection.blocks.filter {$0.kind == .question}.count,1,id)
            XCTAssertTrue(inspection.blocks.contains {$0.kind == .detail && !$0.text.isEmpty},id)
            XCTAssertTrue(content.steps.flatMap(\.blocks).map(\.text).joined().contains("unscored"),id)
            XCTAssertTrue(content.steps.flatMap(\.blocks).contains {$0.kind == .canvasAction},id)
        }
    }
    func testAllReviewedFamiliesHaveStructuredReadingAndCurrentPrerequisites()throws {
        let families=["prepare-media","document-classification","extract-text","transcribe-audio","game-rules","game-episodes","self-play","mixture-experts","classify-audio","prebuilt-models","typed-decisions","fine-tuning","save-model","file-input","model-input","summarize","load-model","text-encoding","iteration-transfer"]
        XCTAssertEqual(Set(families).count,19)
        for id in families {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:id),id)
            XCTAssertFalse(content.steps.isEmpty,id)
            XCTAssertEqual(Set(content.steps.map(\.id)).count,content.steps.count,id)
            XCTAssertFalse(content.steps[0].blocks.first?.text.isEmpty ?? true,id)
            XCTAssertTrue(content.steps.flatMap(\.blocks).contains {$0.kind == .canvasAction},id)
        }
        let rules=try XCTUnwrap(LessonContentCatalog.content(entryID:"game-rules"))
        XCTAssertFalse(rules.steps.flatMap(\.blocks).map(\.text).joined().contains("consumer tools follow later"))
    }
    func testNewPracticeSavedEditsSurviveFreshLaunch()throws {
        _=NSApplication.shared
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false))
        for id in ["prepare-media","prebuilt-models","typed-decisions"] {
            let controller=try registry.openLesson(entryID:id,source:source)
            let workspace=controller.session.workspace
            let owner=try XCTUnwrap(workspace.workflowSession?.owner)
            let address=try XCTUnwrap(owner.document.nodes.first?.address)
            XCTAssertNotNil(owner.apply(.rename(address:address,name:"My retained edit"),lease:owner.lease()))
            XCTAssertTrue(LessonSaveAction.save(workspace:workspace,entryID:id,saveDocument:{controller.session.save()}))
            let saved=owner.document
            source.workspace.refreshProjects()
            let reopened=try registry.openLesson(entryID:id,source:source)
            XCTAssertEqual(reopened.session.workspace.workflowSession?.owner.document,saved)
            let fresh=try registry.startFreshLesson(entryID:id,workspace:workspace)
            XCTAssertNotEqual(fresh.session.workspace.workflowSession?.owner.projectID,owner.projectID)
            XCTAssertEqual(owner.document,saved)
            XCTAssertFalse(fresh.session.workspace.workflowSession?.owner.document.nodes.contains {$0.name == "My retained edit"} ?? true)
            for controller in registry.openControllers {_=registry.close(controller)}
        }
    }

}
