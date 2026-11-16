import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DatabaseOutputTests:XCTestCase {
    private var config:[String:JSONValue] {
        var c=DatabaseOutputSettings.defaults
        c["table"] = .string("orders");c["mode"] = .string("create")
        c["mapping"] = .array([.object(["source":.string("id"),"target":.string("order_id"),"type":.string("INTEGER"),"nullable":false])])
        return c
    }
    func testSettingsRejectDuplicateUnknownOrTruncatedIdentifiers()throws {
        XCTAssertNoThrow(try DatabaseOutputSettings.normalized(config))
        var c=config;c["query"] = .string("DELETE FROM orders")
        XCTAssertThrowsError(try DatabaseOutputSettings.normalized(c))
        c=config;c["keys"] = .array([.string("absent")])
        XCTAssertThrowsError(try DatabaseOutputSettings.normalized(c))
        c=config;c["mode"] = .string("upsert")
        XCTAssertThrowsError(try DatabaseOutputSettings.normalized(c))
        c=config;c["backend"] = .string("postgresql");c["table"] = .string(String(repeating:"x",count:64))
        XCTAssertThrowsError(try DatabaseOutputSettings.normalized(c))
        c=config;c["mapping"] = .array([.object(["source":.string("a"),"target":.string("ID"),"type":.string("INTEGER"),"nullable":false]),.object(["source":.string("b"),"target":.string("id"),"type":.string("INTEGER"),"nullable":false])])
        XCTAssertThrowsError(try DatabaseOutputSettings.normalized(c))
    }
    func testLedgerCrashIntentBlocksSameTargetUntilExplicitAcknowledgment()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let ledger=DatabaseOutputLedger(root:root,projectID:UUID().uuidString)
        func record()->DatabaseOutputLedger.Record {.init(id:UUID().uuidString,nodeID:"out",preparationID:UUID().uuidString,targetIdentity:String(repeating:"a",count:64),configuration:config,sourceSHA256:String(repeating:"b",count:64),createdAt:Date(),receipt:["status":.string("intent")])}
        let first=record();try ledger.begin(first)
        // A new store/process reading the persisted intent must fail closed.
        let reopened=DatabaseOutputLedger(root:root,projectID:ledger.projectID)
        XCTAssertEqual(try reopened.load().count,1)
        XCTAssertThrowsError(try reopened.begin(record()))
        try reopened.finish(id:first.id,receipt:["operationID":.string(first.id),"status":.string("unknown")])
        XCTAssertThrowsError(try reopened.begin(record()))
        try reopened.acknowledge(first.id)
        let next=record();try reopened.begin(next)
        try reopened.finish(id:next.id,receipt:["operationID":.string(next.id),"status":.string("committed")])
        XCTAssertThrowsError(try reopened.finish(id:next.id,receipt:["status":.string("rolledBack")]))
        XCTAssertThrowsError(try reopened.acknowledge(next.id))
        XCTAssertEqual(try reopened.load().first?.receipt["status"],.string("committed"))
    }
    func testDatabaseDestinationPreservesUnusedLegacyFileConfiguration()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let definition=try XCTUnwrap(DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")).operatorDefinition(id:"kwip.data.outputDelimited"))
        var c=definition.defaultConfig;c["destination"] = .string("database");c["database"] = .object(config)
        c["filename"] = .string("old.csv");c["columns"] = .array([.string("old_source")])
        XCTAssertEqual(try TableOutputConfiguration.validate(c,definition:definition,schema:[]),c)
        c["databaseFilename"] = .string("../db.sqlite")
        XCTAssertThrowsError(try TableOutputConfiguration.validate(c,definition:definition,schema:[]))
    }
    func testManagedPreparationHandsCompletePinnedTableToBrokerAndHistoryDoesNotWrite()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the isolated development interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var document=try TableOutputPracticeExample.document()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address.localID=="table-output"})
        var db=DatabaseOutputSettings.defaults;db["table"] = .string("practice_orders");db["mode"] = .string("create")
        db["mapping"] = .array([.object(["source":.string("order_id"),"target":.string("id"),"type":.string("TEXT"),"nullable":false])])
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.outputDelimited",config:["destination":.string("database"),"database":.object(db),"databaseFilename":.string("test.sqlite"),"filename":.string("old.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false])
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowNodeAddress(domain:.data,localID:"table-output")
        XCTAssertTrue(host.bindings.setPath(try TableOutputPracticeExample.resourceURL().path,for:.init(domain:.data,localID:"source"),lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        let database=root.appendingPathComponent("test.sqlite");try Data().write(to:database)
        host.outputDestinations.pickDatabaseFile={database};host.outputDestinations.pickDatabaseFolder={_ in root}
        XCTAssertEqual(try host.outputDestinations.chooseDatabase(for:address,lease:host.owner.lease()),"test.sqlite")
        try host.owner.setRunHistoryPolicy(.none)
        var writes=0
        host.outputDestinations.databaseOutputs.execute={action,settings,secret,id,handle,hash,bookmark,name in
            XCTAssertEqual(action,"write");XCTAssertEqual(name,"test.sqlite");XCTAssertNotNil(bookmark)
            XCTAssertEqual(secret.username,"")
            let bytes=try XCTUnwrap(handle).readToEnd() ?? Data()
            XCTAssertEqual(WorkflowArchiveCodec.hash(bytes),hash)
            let table=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            guard case .array(let rows)=table["rows"] else {throw WorkflowArchiveError("No complete input table")}
            XCTAssertEqual(rows.count,12);writes+=1
            return ["schemaVersion":1,"kind":.string("databaseWrite"),"operationID":.string(id),"status":.string("committed"),"sourceSHA256":.string(hash),"inputRows":12]
        }
        host.executionLaunch={_,args,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,onLine,onExit)}
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(40)
        while host.dataPreparation?.isRunning==true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertEqual(host.dataPreparation?.phase,.completed,host.dataPreparation?.statusMessage ?? host.statusMessage)
        XCTAssertEqual(writes,1)
        let entry=try XCTUnwrap(host.dataPreparationHistory.entries.first)
        _ = try await host.dataPreparationHistory.outputs(for:entry)
        let records=try await host.dataPreparationHistory.databaseRecords(for:entry)
        XCTAssertEqual(records.count,1);XCTAssertEqual(records.first?.receipt["status"],.string("committed"));XCTAssertEqual(writes,1)
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("old.csv").path))
    }

    func testReceiptViewportScrollsInsteadOfClippingWhenDockShrinks()async throws {
        let record=DatabaseOutputLedger.Record(id:UUID().uuidString,nodeID:"output",preparationID:UUID().uuidString,targetIdentity:String(repeating:"a",count:64),configuration:config,sourceSHA256:String(repeating:"b",count:64),createdAt:Date(),receipt:["status":.string("unknown"),"inputRows":12,"message":.string("Check the destination before submitting another write.")])
        let host=NSHostingView(rootView:DatabaseReceiptViewport {DatabaseOutputReceiptView(record:record,acknowledge:{})})
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:420,height:180),styleMask:[.titled,.resizable],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false;window.contentView=host
        defer {window.contentView=nil;window.close()}
        host.frame=NSRect(x:0,y:0,width:420,height:180);host.layoutSubtreeIfNeeded()
        try await Task.sleep(for:.milliseconds(80));host.layoutSubtreeIfNeeded()
        func scroll(_ root:NSView)->NSScrollView? {if let value=root as? NSScrollView{return value};return root.subviews.compactMap(scroll).first}
        let view=try XCTUnwrap(scroll(host)),document=try XCTUnwrap(view.documentView)
        XCTAssertLessThanOrEqual(view.contentView.bounds.height,181)
        XCTAssertGreaterThan(document.bounds.height,view.contentView.bounds.height)
        document.scroll(NSPoint(x:0,y:max(0,document.bounds.height-view.contentView.bounds.height)))
        XCTAssertGreaterThan(view.contentView.bounds.origin.y,0,"Acknowledgment/footer must remain reachable at a compact dock height")
        window.setContentSize(NSSize(width:420,height:110));host.layoutSubtreeIfNeeded()
        try await Task.sleep(for:.milliseconds(80));host.layoutSubtreeIfNeeded()
        XCTAssertLessThanOrEqual(view.contentView.bounds.height,111)
        XCTAssertGreaterThan(document.bounds.height,view.contentView.bounds.height)
    }

    func testPrivateProfilesRequireWriteOptInAndExactEndpointAndFreezeLogin()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try TableOutputPracticeExample.document()
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowNodeAddress(domain:.data,localID:"table-output")
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        let store=host.outputDestinations.databaseOutputs,login=DatabaseOutputMemoryLogin();store.credentials=login
        var db=DatabaseOutputSettings.defaults;db["backend"] = .string("postgresql");db["host"] = .string("127.0.0.1");db["database"] = .string("disposable_test");db["schema"] = .string("public");db["table"] = .string("orders")
        db["mapping"] = .array([.object(["source":.string("order_id"),"target":.string("id"),"type":.string("TEXT"),"nullable":false])])
        try store.saveProfile(name:"Synthetic test login",config:db,username:"test",password:"synthetic-first",allowWrites:false,address:address)
        XCTAssertEqual(try store.secret(for:address,config:db,write:false).password,"synthetic-first")
        XCTAssertThrowsError(try store.secret(for:address,config:db,write:true))
        let id=try XCTUnwrap(store.profile(for:address)?.id)
        try store.saveProfile(name:"Synthetic test login",config:db,username:"test",password:"synthetic-first",allowWrites:true,address:address,updating:id)
        var node=DataFlowNode(id:"table-output",operatorID:"kwip.data.outputDelimited",config:["destination":.string("database"),"database":.object(db)])
        let captured=try store.captureAuthority(node)
        try store.saveProfile(name:"Synthetic test login",config:db,username:"test",password:"synthetic-second",allowWrites:true,address:address,updating:id)
        XCTAssertEqual(captured.credential.password,"synthetic-first")
        XCTAssertEqual(try store.secret(for:address,config:db,write:true).password,"synthetic-second")
        db["host"] = .string("127.0.0.2");node.config["database"] = .object(db)
        XCTAssertThrowsError(try store.captureAuthority(node))
        let bytes=try Data(contentsOf:store.libraryRoot.appendingPathComponent("database-connections/profiles.json"))
        XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("synthetic-first"));XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("synthetic-second"))
        host.owner.setExecutionBusy(true)
        XCTAssertThrowsError(try store.bind(id,address:address));host.owner.setExecutionBusy(false)
    }

}

private final class DatabaseOutputMemoryLogin:DatabaseSourceCredentialStorage {
    var items:[String:DatabaseSourceCredentials]=[:]
    func read(projectID:String,nodeID:String,connectionID:String)throws->DatabaseSourceCredentials? {items[projectID+nodeID+connectionID]}
    func write(_ value:DatabaseSourceCredentials,projectID:String,nodeID:String,connectionID:String)throws {items[projectID+nodeID+connectionID]=value}
    func clear(projectID:String,nodeID:String,connectionID:String)throws {items[projectID+nodeID+connectionID]=nil}
}
