import XCTest
import KwipMLCore
@testable import KwipMLApp

extension MCPWorkflowGatewayTests {
    func testHistoryGatewayRequiresExplicitGrantAndFiltersCapturedIdentities()async throws {
        let f=try fixture(),project=f.owner.projectID,operation=UUID().uuidString
        let root=f.windows.restorationWorkspace.workspaceDirectory
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let document=WorkflowDocument(projectID:project,name:"PRIVATE ROOT",nodes:[.init(address:.init(domain:.data,localID:"old-source"),name:"PRIVATE LABEL",payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:]))])
        let receipt:JSONValue = .object(["schemaVersion":1,"projectID":.string(project),"operationID":.string(operation),"completedAt":.number(Date().timeIntervalSinceReferenceDate),"document":try JSONDecoder().decode(JSONValue.self,from:WorkflowCodec.encode(document)),"requestSHA256":.string(String(repeating:"b",count:64)),"manifestSHA256":.string(String(repeating:"a",count:64))])
        try JSONEncoder().encode(receipt).write(to:directory.appendingPathComponent("preparation-history.json"))
        let request:JSONValue = .object(["operation":.string("list"),"family":.string("preparation")])
        do {_=try await call(f,.resultsRead,["history":request]);XCTFail("History grant missing")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
        var settings=f.controller.settings;settings.projects[project]?.readHistory=true;try f.controller.save(settings)
        let page=try await call(f,.resultsRead,["history":request])
        let snapshot=try XCTUnwrap(page["snapshotID"])
        guard case .array(let records)=page["records"] else {return XCTFail("Missing records")}
        XCTAssertEqual(records.count,1)
        let response=String(decoding:try JSONEncoder().encode(page),as:UTF8.self)
        XCTAssertFalse(response.contains("PRIVATE"));XCTAssertFalse(response.contains(root.path));XCTAssertFalse(response.contains("old-source"))
        XCTAssertTrue(f.owner.document.nodes.isEmpty,"History must not restore or select its captured graph")
        let tools=try await f.gateway.dispatch(clientID:f.client,action:"tools_list",arguments:[:])
        guard case .array(let toolRows)=tools["tools"],case .object(let tool)?=toolRows.first(where: {if case .object(let row)=$0{return row["name"] == .string("results_read")};return false}),case .object(let schema)=tool["inputSchema"] else {return XCTFail("Missing discovery schema")}
        XCTAssertNotNil(schema["oneOf"])
        XCTAssertTrue(tool["description"]?.stringValue?.contains("readHistory") == true)

        let inspect:JSONValue = .object(["operation":.string("inspect"),"family":.string("preparation"),"snapshotID":snapshot,"recordID":.string(operation)])
        settings.projects[project]?.readCapturedValues=false;try f.controller.save(settings)
        do {_=try await call(f,.resultsRead,["history":inspect,"includeValues":true]);XCTFail("Values grant missing")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
        settings.projects[project]?.readCapturedValues=true;try f.controller.save(settings)
        do {_=try await call(f,.resultsRead,["history":inspect,"includeValues":true]);XCTFail("Receipt alone verified")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"history_evidence_unavailable")}
        settings.clients[0].disabledComponents.insert("kwip.data.sourceCSV");try f.controller.save(settings)
        let hidden=try await call(f,.resultsRead,["history":request])
        XCTAssertEqual(hidden["records"],.array([]),"Captured component grant applies even though current graph is empty")
        settings.projects[project]?.readHistory=false;try f.controller.save(settings)
        do {_=try await call(f,.resultsRead,["history":inspect,"includeValues":true]);XCTFail("Revoked History grant reused")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
    }
    func testHistoryReadHandlesExpireAndRefuseForeignClientsAndFamilies()async throws {
        let f=try fixture(),project=f.owner.projectID,access=MCPHistoryReadAccess(),root=f.windows.restorationWorkspace.workspaceDirectory
        var now=Date();access.now={now}
        let args:[String:JSONValue]=["operation":.string("list"),"family":.string("backend")]
        let allow:(Set<String>,Bool)throws->Void={_,_ in}
        for field in ["snapshotID","cursorID","recordID"] {
            var malformed=args;malformed[field]=1
            do {_=try await access.read(client:"one",project:project,root:root,args:malformed,includeValues:false,authorize:allow);XCTFail("Malformed handle accepted")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"history_invalid_arguments")}
        }
        let first=try await access.read(client:"one",project:project,root:root,args:args,includeValues:false,authorize:allow)
        var continued=args;continued["snapshotID"]=first["snapshotID"]
        for (client,p) in [("two",project),("one",UUID().uuidString)] {
            do {_=try await access.read(client:client,project:p,root:root,args:continued,includeValues:false,authorize:allow);XCTFail("Foreign handle accepted")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"history_snapshot_unavailable")}
        }
        var unsupported=args;unsupported["family"] = .string("training")
        do {_=try await access.read(client:"one",project:project,root:root,args:unsupported,includeValues:false,authorize:allow);XCTFail("Unsupported family accepted")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"history_family_unsupported")}
        now=now.addingTimeInterval(301)
        do {_=try await access.read(client:"one",project:project,root:root,args:continued,includeValues:false,authorize:allow);XCTFail("Expired handle accepted")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"history_snapshot_unavailable")}
        let fresh=try await access.read(client:"one",project:project,root:root,args:args,includeValues:false,authorize:allow)
        continued["snapshotID"]=fresh["snapshotID"];access.invalidate()
        do {_=try await access.read(client:"one",project:project,root:root,args:continued,includeValues:false,authorize:allow);XCTFail("Invalidated handle accepted")} catch {}
    }
}
