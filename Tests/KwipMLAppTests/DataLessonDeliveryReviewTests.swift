import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DataLessonDeliveryReviewTests:XCTestCase {
    func testDataPracticesHaveSpecificRailDirectionsAndNoGenericFallback()throws {
        let ids=["union","summarize","sort","unique","reshapeTable","splitText","table-output","text-encoding","sequence-windows","columns","filter","file-input","partition","formula","preprocess","data02-expression","data02-regex-match","join","aggregate","data-preparation-course"]
        for id in ids {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:id),id)
            XCTAssertGreaterThan(content.steps.count,3,id)
            XCTAssertEqual(Set(content.steps.map(\.id)).count,content.steps.count,id)
            let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:"\n")
            XCTAssertFalse(text.contains("Use the lesson's exercise choices to open its supplied canvas."),id)
            XCTAssertFalse(text.contains("Run the example or use Check construction where the lesson supports it."),id)
            XCTAssertTrue(text.contains("unscored") || text.contains("not scored") || text.contains("not a scored") || text.contains("no scored"),id)
            XCTAssertTrue(text.contains("Run") || text.contains("run"),id)
            XCTAssertEqual(content.kind,LearningLibraryPresentation.items.first {$0.id==id}?.kind,id)
        }
        let encoding=try XCTUnwrap(DataPracticeLessonCopy.packets["text-encoding"])
        XCTAssertTrue(encoding.sections.map(\.text).joined().contains("ID 10"))
        XCTAssertFalse(encoding.sections.map(\.text).joined().contains("no longer covers lowercase"))
        XCTAssertTrue(DataPracticeLessonCopy.packets["table-output"]!.introduction.contains("choose the folder"))
        XCTAssertEqual(DataPreparationLessonContent.exampleTitle("preprocess"),"Open Preprocess practice")
        XCTAssertTrue(DataPreparationLessonContent.railSteps.first!.blocks.first!.text.contains("empty lesson canvas"))
    }

    func testAuthoredCanvasLinksResolveOnlyToActualBundledNodes()throws {
        let expression=try NSRegularExpression(pattern:"kwip-node://[^)\\s]+")
        for (id,packet) in DataPracticeLessonCopy.packets {
            let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
            let text=packet.introduction as NSString
            for match in expression.matches(in:packet.introduction,range:NSRange(location:0,length:text.length)) {
                let url=try XCTUnwrap(URL(string:text.substring(with:match.range)))
                let address=try XCTUnwrap(LessonNodeLink.address(url),id)
                XCTAssertTrue(document.nodes.contains {$0.address==address},id+" "+url.absoluteString)
            }
        }
    }

    func testFreshPracticeLaunchersOpenRealBoundGraphsAndPreserveSavedEdits()throws {
        _=NSApplication.shared
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false))
        let expected:[String:String]=["union":"kwip.data.union","summarize":"kwip.data.aggregate","sort":"kwip.data.sort","unique":"kwip.data.unique","reshapeTable":"kwip.data.reshapeTable","splitText":"kwip.data.splitText","table-output":"kwip.data.outputDelimited","text-encoding":"kwip.data.textEncode","sequence-windows":"kwip.data.sequenceWindows","data02-expression":"kwip.data.filterRows","data02-regex-match":"kwip.data.filterRows"]
        for (id,operatorID) in expected.sorted(by:{$0.key<$1.key}) {
            let controller=try registry.openLesson(entryID:id,source:source)
            let workspace=controller.session.workspace
            let document=try XCTUnwrap(workspace.workflowSession?.owner.document,id)
            XCTAssertFalse(document.edges.isEmpty,id)
            XCTAssertTrue(document.nodes.contains {if case .dataOperator(let actual,_)=($0.payload) {return actual==operatorID};return false},id)
            XCTAssertEqual(workspace.lessonRailEntryID,id)
            let sources=document.nodes.filter {if case .dataOperator(let type,_)=($0.payload) {return type=="kwip.data.sourceCSV"};return false}
            XCTAssertFalse(sources.isEmpty,id)
            for node in sources {XCTAssertNotNil(workspace.workflowSession?.bindings.path(for:node.address),id)}
            let edited=try XCTUnwrap(workspace.workflowSession?.owner.document.nodes.first?.address,id)
            let owner=try XCTUnwrap(workspace.workflowSession?.owner,id)
            XCTAssertNotNil(owner.apply(.rename(address:edited,name:"Reviewed "+id),lease:owner.lease()),id)
            let savedDocument=try XCTUnwrap(workspace.workflowSession?.owner.document,id)
            XCTAssertTrue(LessonSaveAction.save(workspace:workspace,entryID:id,saveDocument:{controller.session.save()}),id)
            source.workspace.refreshProjects()
            let reopened=try registry.openLesson(entryID:id,source:source)
            XCTAssertTrue(reopened===controller,id)
            XCTAssertEqual(reopened.session.workspace.workflowSession?.owner.document,savedDocument,id)
            _=registry.close(controller)
        }
    }
}
