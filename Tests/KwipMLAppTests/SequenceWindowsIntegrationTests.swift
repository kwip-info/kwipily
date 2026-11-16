import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SequenceWindowsIntegrationTests:XCTestCase {
    func testRealPreparationPinnedVocabularyAndCapturedPair()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the isolated runner interpreter")}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        var document=try SequenceWindowsPracticeExample.document()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==SequenceWindowsPracticeExample.address})
        guard case .dataOperator(let op,var c)=document.nodes[index].payload else {return XCTFail("Missing windows")}
        c["scope"] = .string("document");c["incomplete"] = .string("pad");document.nodes[index].payload = .dataOperator(operatorID:op,config:c)
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repo.appendingPathComponent("Sources/KwipMLApp/Resources/sequence-windows-observations.csv").path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(60)
        while host.owner.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        if host.owner.isBusy {host.cancel();return XCTFail("Preparation timed out")}
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[SequenceWindowsPracticeExample.address]),lease:host.owner.lease()))
        let viewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:SequenceWindowsPracticeExample.address,port:"table")),host.statusMessage)
        let resolver=try XCTUnwrap(viewer.tokenVocabulary),config=try XCTUnwrap(resolver.windowConfiguration())
        XCTAssertEqual(config,c)
        XCTAssertEqual(try resolver.resolve(column:"input_ids"),["<unk>","<pad>","T","o"," ","b","e","a","r","n","t"])
        XCTAssertNil(try resolver.resolve(column:"loss_mask"))
        let viewResolver=resolver.at(.init(node:"view",port:"table"));XCTAssertEqual(viewResolver.windowConfiguration(),c)
        XCTAssertTrue(viewer.load(),viewer.statusMessage)
        let pageDeadline=Date().addingTimeInterval(30)
        while viewer.isBusy && Date()<pageDeadline {try await Task.sleep(for:.milliseconds(50))}
        let page=try XCTUnwrap(viewer.page,viewer.statusMessage)
        XCTAssertEqual(page.totalRows,7)
        let columns=page.columns.enumerated().map {CapturedDataGrid.Column(id:$0.offset,name:$0.element["name"]!.stringValue!,label:$0.element["name"]!.stringValue!,dtype:$0.element["dtype"]!.stringValue!,nullability:"required")}
        let rows=zip(page.positions,page.rows).map {position,values in CapturedDataGrid.Row(index:position,cells:values.enumerated().map {i,v in CapturedDataGrid.cell(WorkflowCapturedDataOutput.render(v),column:page.columns[i].mapValues(WorkflowCapturedDataOutput.render))})}
        let first=try SequenceWindowSelection(row:XCTUnwrap(rows.first),columns:columns,configuration:config,vocabulary:resolver.resolve(column:"input_ids"))
        XCTAssertEqual(first.inputs.count,8)
        XCTAssertTrue(rows.contains {row in (try? SequenceWindowSelection(row:row,columns:columns,configuration:config,vocabulary:nil).mask.contains(0)) == true})
        // Captured authority remains frozen even if live names and context are changed.
        XCTAssertEqual(resolver.windowConfiguration()?["contextLength"],8)
    }
}
