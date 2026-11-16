import XCTest
import KwipMLCore
@testable import KwipMLApp

extension MCPWorkflowGatewayTests {
    func testAssetStatusSeparateGrantAndNoProjectOpening()async throws {
        let f=try fixture()
        let before=f.windows.openControllers.count
        let result=try await call(f,.assetStatusRead)
        XCTAssertEqual(result["sourceStatus"],.string("noRetainedBindings"))
        XCTAssertEqual(f.windows.openControllers.count,before)
        var settings=f.controller.settings
        settings.projects[f.owner.projectID]?.actions.remove(.assetStatusRead)
        try f.controller.save(settings)
        do {_=try await call(f,.assetStatusRead);XCTFail("Project grant bypass")}catch let e as MCPSettings.Refusal {XCTAssertEqual(e.code,"not_permitted")}
        settings.projects[f.owner.projectID]?.actions.insert(.assetStatusRead)
        settings.clients[0].actions.remove(.assetStatusRead);try f.controller.save(settings)
        do {_=try await call(f,.assetStatusRead);XCTFail("Client grant bypass")}catch let e as MCPSettings.Refusal {XCTAssertEqual(e.code,"not_permitted")}
    }
    func testAssetStatusFiltersBlockedModelComponentBeforeReturningMetadata()async throws {
        let f=try fixture(),root=f.windows.restorationWorkspace.workspaceDirectory
        let reference=ModelAssetReference(projectID:f.owner.projectID,directoryID:UUID().uuidString,metadata:["identity":.string(String(repeating:"a",count:64)),"files":.array([.object(["path":.string("config.json"),"bytes":1,"sha256":.string(String(repeating:"b",count:64))])])],sourceConfiguration:LoadModelConfiguration(source:.localFolder).values)
        let encoded=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(reference))
        let envelope:JSONValue = .object(["schemaVersion":1,"projectID":.string(f.owner.projectID),"revision":1,"entries":.array([.object(["address":.object(["domain":.string("control"),"localID":.string("model")]),"reference":encoded])])])
        try DiagnosticFileIO.writeAtomic(try JSONEncoder().encode(envelope),root:root,parts:[ModelAssetReference.storageNamespace,f.owner.projectID,"bindings.json"])
        let first=try await call(f,.assetStatusRead)
        guard case .array(let records)=first["records"] else{return XCTFail("Missing records")}
        XCTAssertEqual(records.count,1)
        var settings=f.controller.settings;settings.clients[0].disabledComponents.insert("workflow.frozenModel");try f.controller.save(settings)
        let filtered=try await call(f,.assetStatusRead)
        XCTAssertEqual(filtered["records"],.array([]))
    }

}
