import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPrivateModelDestinationsTests:XCTestCase {
    private let save=WorkflowNodeAddress(domain:.control,localID:"save-model")
    @MainActor private func fixture()throws->(URL,WorkflowAuthoringStore) {
        let root=try canonicalBackendTemporaryDirectory()
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        var document=CurrentExecutionWorkflowTestFixture.make(block:false)
        document=try WorkflowTrainedModelSource.promoteAuthoring(document)
        document.nodes.append(.init(address:save,payload:.saveModel(config:WorkflowSaveModelConfiguration().values)))
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),
            dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        return(root,owner)
    }
    @MainActor func testPrivateParentBookmarkPersistsWithoutPortableSourceChanges()throws {
        let(root,owner)=try fixture(),before=owner.document
        let destinations=try WorkflowPrivateModelDestinations(owner:owner,directory:root)
        let destination=root.appendingPathComponent("chosen.pt")
        XCTAssertTrue(destinations.set(destination:destination,for:save,lease:owner.lease(),expectedRevision:0),destinations.statusMessage)
        XCTAssertEqual(destinations.revision,1);XCTAssertEqual(destinations.path(for:save),destination.path)
        XCTAssertEqual(owner.document,before)
        let portable=try WorkflowArchiveCodec.encode(owner.document)
        XCTAssertFalse(String(decoding:portable,as:UTF8.self).contains(destination.path))
        let restored=try WorkflowPrivateModelDestinations(owner:owner,directory:root)
        XCTAssertEqual(restored.path(for:save),destination.path)
        let access=try restored.acquire(for:save,lease:owner.lease());try access.check()
        XCTAssertEqual(access.destination,destination);XCTAssertEqual(access.revision,1)
        access.close();XCTAssertThrowsError(try access.check())
        XCTAssertTrue(restored.clear(for:save,lease:owner.lease(),expectedRevision:1));XCTAssertNil(restored.path(for:save))
        XCTAssertEqual(owner.document,before)
    }
    @MainActor func testStaleLeaseRevisionBusyAndWrongNodeCannotReplaceGrant()throws {
        let(root,owner)=try fixture(),destinations=try WorkflowPrivateModelDestinations(owner:owner,directory:root)
        let original=root.appendingPathComponent("original.pt"),lease=owner.lease()
        XCTAssertTrue(destinations.set(destination:original,for:save,lease:lease,expectedRevision:0))
        XCTAssertFalse(destinations.set(destination:root.appendingPathComponent("other.pt"),for:save,lease:lease,expectedRevision:0))
        owner.setExecutionBusy(true)
        XCTAssertFalse(destinations.clear(for:save,lease:lease));owner.setExecutionBusy(false)
        XCTAssertFalse(destinations.set(destination:root.appendingPathComponent("other.pt"),for:.init(domain:.control,localID:"train"),lease:lease))
        XCTAssertEqual(destinations.path(for:save),original.path);XCTAssertEqual(destinations.revision,1)
        XCTAssertTrue(owner.rename("Renamed workflow",lease:owner.lease()))
        XCTAssertFalse(destinations.clear(for:save,lease:lease));XCTAssertEqual(destinations.path(for:save),original.path)
    }
    @MainActor func testExistingLeafAndSymlinkParentRefuseWithoutPrivateMutation()throws {
        let(root,owner)=try fixture(),destinations=try WorkflowPrivateModelDestinations(owner:owner,directory:root)
        let existing=root.appendingPathComponent("exists.pt"),alias=root.appendingPathComponent("alias")
        try Data("retain".utf8).write(to:existing)
        try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:root)
        XCTAssertFalse(destinations.set(destination:existing,for:save,lease:owner.lease()))
        XCTAssertFalse(destinations.set(destination:alias.appendingPathComponent("new.pt"),for:save,lease:owner.lease()))
        XCTAssertNil(destinations.path(for:save));XCTAssertEqual(destinations.revision,0)
        XCTAssertEqual(try Data(contentsOf:existing),Data("retain".utf8))
    }
}
