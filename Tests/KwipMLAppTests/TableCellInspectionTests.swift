import XCTest
import AppKit
import SwiftUI
@testable import KwipMLApp

final class TableCellInspectionTests:XCTestCase {
    @MainActor func testOverflowDrawerIsReadOnlyClosableAndDoesNotOpenShortValues() {
        let long="First line\n"+String(repeating:"Complete captured words ",count:30)
        let rows=[CapturedDataGrid.Row(index:9,cells:[.init(text:long,isAbsent:false)]),.init(index:3,cells:[.init(text:"short",isAbsent:false)])]
        let model=CapturedNativeTable(columns:[.init(id:0,name:"note",label:"Note",dtype:"string",nullability:"required")],rows:rows,selectedRow:.constant(nil))
        let coordinator=CapturedNativeTable.Coordinator(model),container=CapturedTableContainerView(frame:NSRect(x:0,y:0,width:500,height:320))
        coordinator.container=container
        let table=container.scroll.documentView as! NSTableView
        for id in ["sample","0"] {let column=NSTableColumn(identifier:.init(id));column.width=150;table.addTableColumn(column)}
        coordinator.inspect(row:1,column:0,table:table)
        XCTAssertTrue(container.detail.isHidden)
        coordinator.inspect(row:0,column:0,table:table)
        XCTAssertFalse(container.detail.isHidden);XCTAssertEqual(container.text.string,long)
        let native=coordinator.tableView(table,viewFor:table.tableColumns[1],row:0) as? NSTableCellView
        XCTAssertEqual(native?.textField?.maximumNumberOfLines,1)
        XCTAssertFalse(native?.textField?.stringValue.contains("\n") ?? true)
        XCTAssertTrue(native?.textField?.stringValue.hasPrefix("First line") ?? false)
        XCTAssertTrue(native?.layer?.masksToBounds==true)
        container.layoutSubtreeIfNeeded()
        XCTAssertGreaterThan(container.scroll.frame.height,60)
        XCTAssertLessThanOrEqual(container.detail.frame.maxY,container.scroll.frame.minY)
        XCTAssertFalse(container.text.isEditable);XCTAssertTrue(container.text.isSelectable)
        XCTAssertTrue(container.title.stringValue.contains("Row 10"))
        container.closeDetail();XCTAssertTrue(container.detail.isHidden);XCTAssertEqual(container.text.string,"")
        coordinator.inspect(row:-1,column:0,table:table);XCTAssertTrue(container.detail.isHidden)
    }
    @MainActor func testNativeTableRemainsContainedWhenHistoryPaneShrinks() {
        let view=CapturedTableContainerView(frame:NSRect(x:0,y:0,width:1200,height:800))
        view.show(title:"Captured value",page:.slice(String(repeating:"text",count:100),start:0),focus:false)
        for size in [NSSize(width:400,height:160),NSSize(width:240,height:28),NSSize(width:800,height:500)] {
            view.setFrameSize(size);view.needsLayout=true;view.layoutSubtreeIfNeeded()
            XCTAssertTrue(view.bounds.contains(view.scroll.frame))
            XCTAssertTrue(view.bounds.contains(view.detail.frame))
            XCTAssertTrue(view.clipsToBounds)
            XCTAssertTrue(view.scroll.clipsToBounds)
            XCTAssertTrue(view.detail.clipsToBounds)
            XCTAssertEqual(view.layer?.masksToBounds,true)
        }
        view.closeDetail();view.needsLayout=true;view.layoutSubtreeIfNeeded()
        XCTAssertEqual(view.scroll.frame,view.bounds)
    }
    func testFullCellProtocolCannotRelaxOrdinaryPageLimitsOrAddressAnotherRow()throws {
        let pin=String(repeating:"a",count:64)
        var query=TableViewQuery(schemaVersion:5,offset:42,limit:1,search:"",sourceSHA256:pin,indexSHA256:pin,fullCellColumn:3)
        XCTAssertEqual(try TableViewQuery.decode(JSONEncoder().encode(query)),query)
        var event:[String:Any]=["event":"tableViewPage","schemaVersion":1,"columns":[["name":"text","dtype":"string","nullable":false]],"rows":[[String(repeating:"x",count:9000)]],"positions":[42],"offset":42,"nextOffset":NSNull(),"totalRows":100,"matchedRows":100,"cellTruncated":false,"sourceSHA256":pin,"cellOffset":0,"cellTotalCharacters":9000]
        XCTAssertNoThrow(try TableViewPage.decode(JSONSerialization.data(withJSONObject:event),query:query))
        event["positions"]=[43];XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:event),query:query))
        query.schemaVersion=2;XCTAssertThrowsError(try query.validate())
        query.schemaVersion=5;query.search="filtered";XCTAssertThrowsError(try query.validate())
    }
}

extension TableCellInspectionTests {
    @MainActor func testActualCapturedCellReadKeepsThePageAndRefusesChangedEvidence()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Explicit development worker required")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory();try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let long="Begin\n"+String(repeating:"Captured Unicode 🔎 words\n",count:2000)
        let values=root.appendingPathComponent("values.json"),index=root.appendingPathComponent("view.sqlite3")
        try JSONSerialization.data(withJSONObject:[[0,"short"],[1,long]]).write(to:values)
        let process=Process();process.executableURL=URL(fileURLWithPath:python);process.currentDirectoryURL=repository
        process.arguments=["-c","import json,sys;from pathlib import Path;from types import SimpleNamespace;from runner.table_view import build_index;rows=json.loads(Path(sys.argv[1]).read_text());build_index(SimpleNamespace(columns=[dict(name='id',dtype='int64',nullable=False),dict(name='text',dtype='string',nullable=False)],rows=rows,row_ids=['a','b']),Path(sys.argv[2]),source_sha256='a'*64)",values.path,index.path]
        try process.run();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let runtime=try XCTUnwrap(workspace.runtime),pin=WorkflowArchiveCodec.hash(try Data(contentsOf:index))
        var held=false
        let session=TableViewQuerySession(artifact:.init(indexURL:index,indexSHA256:pin,sourceSHA256:String(repeating:"a",count:64)),workspace:root,projectID:UUID().uuidString,runtime:runtime,reservation:.init(acquire:{if held{return nil};held=true;return UUID()},release:{_ in held=false}),accepts:{true})
        XCTAssertTrue(session.load(sort:.init(column:0,descending:true)))
        let deadline=Date().addingTimeInterval(45)
        while session.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        let page=try XCTUnwrap(session.page,session.statusMessage);XCTAssertTrue(page.cellTruncated)
        var inspected:CapturedCellPage?
        session.readFullCell(row:1,column:1){inspected=$0}
        while inspected==nil && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertEqual(inspected?.text,String(long.unicodeScalars.prefix(16_384)));XCTAssertEqual(inspected?.next,16_384);XCTAssertEqual(session.page,page)
        inspected=nil;session.readFullCell(row:1,column:1,start:16_384){inspected=$0}
        while inspected==nil && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertEqual(inspected?.text,String(String.UnicodeScalarView(long.unicodeScalars.dropFirst(16_384).prefix(16_384))))
        XCTAssertEqual(inspected?.start,16_384);XCTAssertEqual(session.page,page)
        var bytes=try Data(contentsOf:index);bytes[0]^=1;try bytes.write(to:index)
        inspected=nil;session.readFullCell(row:1,column:1){inspected=$0}
        while inspected==nil && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertTrue(inspected?.message?.hasPrefix("Text preview unavailable:")==true,inspected?.message ?? "no completion")
        XCTAssertEqual(session.page,page)
    }
}
