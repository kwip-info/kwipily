import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class APITableTests:XCTestCase {
    var input:[String:JSONValue] {APITableSettings.defaults.merging(["url":.string("https://api.example.com/orders"),"recordsPath":.string("/items"),"mapping":.array([.object(["source":.string("/id"),"target":.string("id"),"type":.string("int64")])])]){_,v in v}}
    var output:[String:JSONValue] {input.merging(["direction":.string("output"),"method":.string("POST"),"mapping":.array([.object(["source":.string("id"),"target":.string("/id"),"type":.string("int64")])])]){_,v in v}}
    func testStrictSettingsAndSourceTransportPreserveLegacyDefaults()throws {
        XCTAssertNoThrow(try APITableSettings.normalized(input))
        let source:[String:JSONValue]=["format":.string("api"),"options":.object(input)]
        XCTAssertEqual(try TableInputConfiguration.validate(source),source)
        XCTAssertTrue(try TableInputConfiguration.transportJSON(source).contains("api.example.com"))
        XCTAssertEqual(try TableInputConfiguration.format(of:[:]),.csv)
        for (key,value) in [("password",JSONValue.string("never")),("maximumPages",.bool(true)),("headers",.object(["Authorization":.string("never")])),("url",.string("http://example.com")),("method",.string("POST"))] {var c=input;c[key]=value;XCTAssertThrowsError(try APITableSettings.normalized(c))}
        var local=input;local["url"] = .string("http://127.0.0.1:8765/orders");local["allowLoopback"] = true
        XCTAssertNoThrow(try APITableSettings.normalized(local))
    }
    func testAPIOutputDoesNotAcquireFileGrantAndPreservesFileDraft()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let definition=try XCTUnwrap(registry.operatorDefinition(id:"kwip.data.outputDelimited"))
        var c=definition.defaultConfig;c["filename"] = .string("kept.csv");c["destination"] = .string("api");c["api"] = .object(output)
        XCTAssertEqual(try TableOutputConfiguration.validate(c,definition:definition,schema:[]),c)
        XCTAssertFalse(DatabaseOutputSettings.requiresLocalFolder(c));XCTAssertTrue(APITableSettings.isExternal(c))
        c["destination"] = .string("file");XCTAssertTrue(DatabaseOutputSettings.requiresLocalFolder(c));XCTAssertEqual(c["filename"],.string("kept.csv"))
    }
    func testAPIUnknownReceiptSurvivesReopenAndBlocksWholeNewOperation()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,ledger=DatabaseOutputLedger(root:root,projectID:project)
        func record()->DatabaseOutputLedger.Record {.init(id:UUID().uuidString,nodeID:"api",preparationID:UUID().uuidString,targetIdentity:String(repeating:"a",count:64),configuration:output,sourceSHA256:String(repeating:"b",count:64),createdAt:Date(),receipt:["status":.string("intent")])}
        let first=record();try ledger.begin(first)
        try ledger.finish(id:first.id,receipt:["status":.string("unknown"),"kind":.string("apiWrite"),"operationID":.string(first.id),"acceptedRows":2])
        let reopened=DatabaseOutputLedger(root:root,projectID:project)
        XCTAssertThrowsError(try reopened.begin(record()));XCTAssertEqual(try reopened.load().first?.receipt["acceptedRows"],2)
        try reopened.acknowledge(first.id);XCTAssertNoThrow(try reopened.begin(record()))
    }
    func testAPIProfilePinsOriginAuthModeAndExplicitWriteGrant()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false),document=try TableOutputPracticeExample.document()
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowNodeAddress(domain:.data,localID:"table-output")
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        let store=host.outputDestinations.databaseOutputs;store.credentials=APIMemoryCredentials()
        var c=output;c["auth"] = .string("bearer")
        try store.saveProfile(name:"Fixture",config:c,username:"",password:"synthetic-api-secret",allowWrites:false,address:address)
        XCTAssertEqual(try store.secret(for:address,config:c,write:false).password,"synthetic-api-secret")
        XCTAssertThrowsError(try store.secret(for:address,config:c,write:true))
        let id=try XCTUnwrap(store.profile(for:address)?.id)
        try store.saveProfile(name:"Fixture",config:c,username:"",password:"synthetic-api-secret",allowWrites:true,address:address,updating:id)
        XCTAssertNoThrow(try store.secret(for:address,config:c,write:true))
        c["url"] = .string("https://other.example.com/orders");XCTAssertThrowsError(try store.secret(for:address,config:c,write:true))
        c=output;c["auth"] = .string("basic");XCTAssertThrowsError(try store.secret(for:address,config:c,write:true))
        let data=try Data(contentsOf:store.libraryRoot.appendingPathComponent("database-connections/profiles.json"));XCTAssertFalse(String(decoding:data,as:UTF8.self).contains("synthetic-api-secret"))
    }
    func testResponseGridRetainsEveryBoundedRequestWithoutChangingSourcePreviewLimit()throws {
        let columns:[[String:Any]]=[["name":"request","dtype":"int64","nullable":false],["name":"outcome","dtype":"string","nullable":false]]
        let rows:[[Any]]=(1...12).map {[$0,"accepted"]}
        let value:[String:Any]=["columns":columns,"rows":rows,"rowCount":12]
        XCTAssertEqual(CapturedDataGrid(preview:value).rows.count,8)
        let responses=CapturedDataGrid(preview:value,rowLimit:256)
        XCTAssertEqual(responses.rows.count,12);XCTAssertEqual(responses.rows.last?.cells.first?.text,"12")
    }

    func testNativeAPISettingsMatchTheActualPythonAdapter()throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the development interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var sending=output;sending["constants"] = .array([.object(["target":.string("/source"),"json":.string("\"canvas\"")])])
        let configurations=try [input,sending].map {try APITableSettings.normalized($0)}
        let p=Process(),stdin=Pipe(),stdout=Pipe(),stderr=Pipe();p.executableURL=URL(fileURLWithPath:python);p.currentDirectoryURL=repository
        p.arguments=["-c","import json,sys; from runner.api_tables import options; print(json.dumps([options(c) for c in json.load(sys.stdin)],sort_keys=True))"]
        p.standardInput=stdin;p.standardOutput=stdout;p.standardError=stderr;try p.run();try stdin.fileHandleForWriting.write(contentsOf:WorkflowArchiveCodec.encode(configurations));try stdin.fileHandleForWriting.close();p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus,0,String(decoding:stderr.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self))
        XCTAssertEqual(try JSONDecoder().decode([[String:JSONValue]].self,from:stdout.fileHandleForReading.readDataToEndOfFile()),configurations)
    }

    func testManagedAPIDeliveryRetainsCompleteInputAndHistoryDoesNotResend()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the development interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var document=try TableOutputPracticeExample.document()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address.localID=="table-output"})
        var c=output;c["responsePath"] = .string("/result");c["mapping"] = .array([.object(["source":.string("order_id"),"target":.string("/id"),"type":.string("string")])])
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.outputDelimited",config:["destination":.string("api"),"api":.object(c),"filename":.string("old.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false])
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowNodeAddress(domain:.data,localID:"table-output")
        XCTAssertTrue(host.bindings.setPath(try TableOutputPracticeExample.resourceURL().path,for:.init(domain:.data,localID:"source"),lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        let store=host.outputDestinations.databaseOutputs;store.credentials=APIMemoryCredentials()
        try store.saveProfile(name:"Fixture",config:c,username:"",password:"",allowWrites:true,address:address)
        try host.owner.setRunHistoryPolicy(.none)
        var writes=0
        store.apiExecute={action,settings,secret,id,handle,hash,memory in
            let bytes=try XCTUnwrap(handle).readToEnd() ?? Data();XCTAssertEqual(WorkflowArchiveCodec.hash(bytes),hash)
            let table=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            guard case .array(let rows)=table["rows"] else {throw WorkflowArchiveError("No complete input")}
            XCTAssertEqual(rows.count,12);writes+=1
            return ["schemaVersion":1,"kind":.string("apiWrite"),"operationID":.string(id),"status":.string("committed"),"sourceSHA256":.string(hash),"inputRows":12,"acceptedRows":12,"requests":.array([.object(["request":1,"responseJSON":.string("optional-body-must-not-be-retained")])])]
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
        XCTAssertEqual(records.count,1);XCTAssertEqual(records.first?.receipt["kind"],.string("apiWrite"));XCTAssertEqual(records.first?.receipt["responseCapture"],.string("disabledByRunPolicy"));XCTAssertFalse(String(decoding:try WorkflowArchiveCodec.encode(records.first?.receipt),as:UTF8.self).contains("optional-body-must-not-be-retained"));XCTAssertEqual(writes,1)
    }

}
private final class APIMemoryCredentials:DatabaseSourceCredentialStorage {
    var values:[String:DatabaseSourceCredentials]=[:]
    func read(projectID:String,nodeID:String,connectionID:String)throws->DatabaseSourceCredentials? {values[projectID+nodeID+connectionID]}
    func write(_ value:DatabaseSourceCredentials,projectID:String,nodeID:String,connectionID:String)throws {values[projectID+nodeID+connectionID]=value}
    func clear(projectID:String,nodeID:String,connectionID:String)throws {values[projectID+nodeID+connectionID]=nil}
}
