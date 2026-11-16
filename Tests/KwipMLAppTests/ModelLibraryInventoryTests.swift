import XCTest
import Security
import KwipMLCore
@testable import KwipMLApp

final class ModelLibraryInventoryTests:XCTestCase {
    func testMetadataInventoryIncludesExternalWorkspacesAndNeverReadsWeights()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        for workspace in [root,root.appendingPathComponent("external-projects/"+UUID().uuidString)] {
            let folder=workspace.appendingPathComponent("model-assets/"+UUID().uuidString+"/"+UUID().uuidString)
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            try Data("{\"model_type\":\"gpt2\"}".utf8).write(to:folder.appendingPathComponent("config.json"))
            try Data([1,2,3]).write(to:folder.appendingPathComponent("model.safetensors"))
        }
        let rows=try ModelLibraryInventory.scan(application:root)
        XCTAssertEqual(rows.count,2);XCTAssertTrue(rows.allSatisfy{$0.architecture=="GPT2" && $0.fileCount==2})
        XCTAssertEqual(try ModelLibraryInventory.scan(application:root),rows)
    }
    func testInventoryRejectsSymbolicLinkInsteadOfFollowingAnotherFolder()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("model-assets"),withDestinationURL:root)
        XCTAssertThrowsError(try ModelLibraryInventory.scan(application:root))
    }
    func testRemovalRequiresOneBoundedCredentialNameBeforeKeychainAccess() {
        XCTAssertThrowsError(try ModelHubCredentials.remove(name:""))
        XCTAssertThrowsError(try ModelHubCredentials.remove(name:String(repeating:"x",count:101)))
    }
}

extension ModelLibraryInventoryTests {
    func testCredentialReferencePersistsSeparatelyAndRemovingSelectionDoesNotTouchModel()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let project=UUID().uuidString,address=WorkflowNodeAddress(domain:.control,localID:"load")
        let selections=ModelHubCredentialSelection(root:root,project:project)
        try selections.select("My private account",for:address)
        XCTAssertEqual(try ModelHubCredentialSelection(root:root,project:project).name(for:address),"My private account")
        try selections.select(nil,for:address)
        XCTAssertNil(try selections.name(for:address))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("projects").path))
    }
}

extension ModelLibraryInventoryTests {
    func testDownloadDeadlineCancelsOutstandingReadAndFastReadsReturnExactly()async throws {
        let value=try await ModelHubBudget.read(remainingSeconds:1){Data([7,8])}
        XCTAssertEqual(value,Data([7,8]))
        do {
            _=try await ModelHubBudget.read(remainingSeconds:0.01){try await Task.sleep(for:.seconds(30));return Data([1])}
            XCTFail("A range read must not outlive the captured download deadline")
        } catch {XCTAssertTrue(error.localizedDescription.contains("time allowance"))}
    }
}

extension ModelLibraryInventoryTests {
    func testCredentialEnumerationNeverRequestsDataAndRemovalTargetsOnlyOneAppAccount()throws {
        var keychain=ModelHubKeychain()
        keychain.copy={query,result in
            let values=query as NSDictionary
            XCTAssertNil(values[kSecReturnData])
            XCTAssertEqual(values[kSecReturnAttributes] as? Bool,true)
            XCTAssertNotNil(values[kSecAttrService])
            result?.pointee=[[kSecAttrAccount as String:"review-fixture"]] as NSArray
            return errSecSuccess
        }
        XCTAssertEqual(try ModelHubCredentials.names(using:keychain),["review-fixture"])
        var deleted=false
        keychain.delete={query in
            let values=query as NSDictionary
            XCTAssertEqual(values[kSecAttrAccount] as? String,"review-fixture")
            XCTAssertNotNil(values[kSecAttrService])
            XCTAssertEqual(values.count,4,"No wildcard or cross-service deletion")
            deleted=true;return errSecSuccess
        }
        try ModelHubCredentials.remove(name:"review-fixture",using:keychain)
        XCTAssertTrue(deleted)
    }
    func testCredentialRotationUpdatesOnlyTheExistingNamedSecret()throws {
        var keychain=ModelHubKeychain(),updated=false
        keychain.add={_,_ in errSecDuplicateItem}
        keychain.update={query,changes in
            XCTAssertEqual((query as NSDictionary)[kSecAttrAccount] as? String,"fixture")
            XCTAssertEqual((changes as NSDictionary).count,1)
            XCTAssertNotNil((changes as NSDictionary)[kSecValueData])
            updated=true;return errSecSuccess
        }
        try ModelHubCredentials.save(name:"fixture",token:"hf_noncredential_test_value",using:keychain)
        XCTAssertTrue(updated)
    }
}
