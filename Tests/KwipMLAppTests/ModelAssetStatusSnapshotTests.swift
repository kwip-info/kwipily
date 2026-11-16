import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelAssetStatusSnapshotTests:XCTestCase {
    private func root()throws->URL {
        let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:url)};return url
    }
    private func write(root:URL,project:String,referenceProject:String?=nil,identity:String=String(repeating:"a",count:64))throws->Data {
        let config=LoadModelConfiguration(source:.localFolder).values
        let ref=ModelAssetReference(projectID:referenceProject ?? project,directoryID:UUID().uuidString,metadata:["identity":.string(identity),"files":.array([.object(["path":.string("config.json"),"bytes":1,"sha256":.string(String(repeating:"b",count:64))])]),"task":.string("typedDecisions"),"architecture":.string("PRIVATE /secret"),"credential":.string("PRIVATE TOKEN"),"capabilities":.object(["PRIVATE":true])],sourceConfiguration:config)
        let reference=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(ref))
        let data=try JSONEncoder().encode(JSONValue.object(["schemaVersion":1,"projectID":.string(project),"revision":1,"entries":.array([.object(["address":.object(["domain":.string("control"),"localID":.string("model")]),"reference":reference])])]))
        try DiagnosticFileIO.writeAtomic(data,root:root,parts:[ModelAssetReference.storageNamespace,project,"bindings.json"]);return data
    }
    func testNoModelBytesRequiredAndPrivateMetadataNeverProjected()async throws {
        let root=try root(),project=UUID().uuidString
        let original=try write(root:root,project:project)
        let result=try await ModelAssetStatusSnapshot.load(root:root,projectID:project)
        XCTAssertEqual(result.items.count,1)
        let item=try XCTUnwrap(result.items.first)
        XCTAssertEqual(item.value["status"],.string("retainedPinnedMetadata"))
        XCTAssertEqual(item.value["contentVerification"],.string("notPerformed"))
        XCTAssertEqual(item.value["currentApplicability"],.string("notVerified"))
        XCTAssertEqual(item.components,["workflow.frozenModel"])
        let serialized=String(decoding:try JSONEncoder().encode(item.value),as:UTF8.self)
        XCTAssertFalse(serialized.contains("PRIVATE"));XCTAssertFalse(serialized.contains("config.json"));XCTAssertFalse(serialized.contains("credential"))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("model-assets").path))
        XCTAssertEqual(try DiagnosticFileIO.read(root:root,parts:[ModelAssetReference.storageNamespace,project,"bindings.json"],limit:1_000_000),original)
    }
    func testOtherProjectNotEnumeratedAndForgedProjectRefused()async throws {
        let root=try root(),a=UUID().uuidString,b=UUID().uuidString
        _=try write(root:root,project:b)
        let empty=try await ModelAssetStatusSnapshot.load(root:root,projectID:a)
        XCTAssertTrue(empty.items.isEmpty)
        _=try write(root:root,project:a,referenceProject:b)
        do {_=try await ModelAssetStatusSnapshot.load(root:root,projectID:a);XCTFail("Cross-project reference accepted")}catch {}
    }
    func testInvalidPinsRemainUnknownProtectedNotReady()async throws {
        let root=try root(),project=UUID().uuidString;_=try write(root:root,project:project,identity:"PRIVATE /bad")
        let result=try await ModelAssetStatusSnapshot.load(root:root,projectID:project)
        let value=try XCTUnwrap(result.items.first).value
        XCTAssertEqual(value["status"],.string("unknownProtectedMetadata"));XCTAssertNil(value["identity"])
        XCTAssertEqual(value["protected"],true);XCTAssertEqual(value["removable"],false)
    }
    func testMalformedOversizedAndSymlinkMetadataRefused()async throws {
        let root=try root(),project=UUID().uuidString
        _=try write(root:root,project:project)
        let file=root.appendingPathComponent("\(ModelAssetReference.storageNamespace)/\(project)/bindings.json")
        for data in [Data("{}".utf8),Data(repeating:32,count:1_000_001)] {
            try data.write(to:file)
            do {_=try await ModelAssetStatusSnapshot.load(root:root,projectID:project);XCTFail("Unsafe metadata accepted")}catch {}
        }
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createSymbolicLink(at:file,withDestinationURL:root.appendingPathComponent("absent"))
        do {_=try await ModelAssetStatusSnapshot.load(root:root,projectID:project);XCTFail("Symlink accepted")}catch {}
    }
    func testSeparatePermissionNotInCreatedProject()throws {
        let action=try XCTUnwrap(MCPAction(rawValue:"asset_status_read"))
        XCTAssertTrue(action.isReadOnly);XCTAssertTrue(action.requiresProject)
        XCTAssertFalse(MCPProjectGrant.createdProject.actions.contains(action))
        XCTAssertFalse(MCPSettings.factory.visibleActions(clientID:"old").contains(action))
    }
}
