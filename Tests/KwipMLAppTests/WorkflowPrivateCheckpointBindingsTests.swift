import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPrivateCheckpointBindingsTests:XCTestCase {
    private let train=WorkflowNodeAddress(domain:.control,localID:"shared")
    private func fixture()throws->(URL,ProjectLibrary,WorkflowAuthoringStore) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:train,payload:.train(config:[:])),
            .init(address:.init(domain:.data,localID:"shared"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:.init(domain:.control,localID:"setup"),payload:.runSetup(config:[:]))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        return(root,library,owner)
    }
    private func ref(_ owner:WorkflowAuthoringStore,project:String?=nil,directory:String=UUID().uuidString,pin:String=String(repeating:"a",count:64))->WorkflowCheckpointReference {
        .init(projectID:project ?? owner.projectID,directoryID:directory,runID:UUID().uuidString,requestIdentity:String(repeating:"b",count:64),modelManifestSHA256:pin)
    }
    private func file(_ root:URL,_ owner:WorkflowAuthoringStore)->URL {root.appendingPathComponent("workflow-checkpoint/\(owner.projectID)/bindings.json")}
    func testPersistRestartClearAndNoOpDoNotMutatePortableWorkflow()throws {
        let(root,library,owner)=try fixture(),binder=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        let before=try ProjectSourceCodec.encode(library.loadSource(id:owner.projectID)),reference=ref(owner)
        XCTAssertEqual(binder.revision,0);XCTAssertNil(binder.reference(for:train))
        XCTAssertTrue(binder.setReference(reference,for:train,lease:owner.lease(),expectedRevision:0))
        XCTAssertEqual(binder.revision,1);XCTAssertEqual(binder.reference(for:train),reference)
        XCTAssertTrue(binder.setReference(reference,for:train,lease:owner.lease(),expectedRevision:1));XCTAssertEqual(binder.revision,1)
        let restarted=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        XCTAssertEqual(restarted.reference(for:train),reference);XCTAssertEqual(restarted.revision,1)
        XCTAssertEqual(try ProjectSourceCodec.encode(library.loadSource(id:owner.projectID)),before)
        XCTAssertTrue(restarted.setReference(nil,for:train,lease:owner.lease(),expectedRevision:1))
        XCTAssertEqual(restarted.revision,2);XCTAssertNil(restarted.reference(for:train))
    }
    func testInvalidIdentityPinAliasLeaseAndBusyPreserveSavedBytes()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        let reference=ref(owner),lease=owner.lease()
        XCTAssertTrue(binder.setReference(reference,for:train,lease:lease,expectedRevision:0))
        let bytes=try Data(contentsOf:file(root,owner))
        for invalid in [ref(owner,project:UUID().uuidString),ref(owner,directory:"../escape"),ref(owner,pin:"bad"),ref(owner,pin:String(repeating:"A",count:64))] {
            XCTAssertFalse(binder.setReference(invalid,for:train,lease:lease,expectedRevision:1))
        }
        for address in [WorkflowNodeAddress(domain:.data,localID:"shared"),.init(domain:.control,localID:"setup"),.init(domain:.control,localID:"missing")] {
            XCTAssertFalse(binder.setReference(reference,for:address,lease:lease,expectedRevision:1));XCTAssertNil(binder.reference(for:address))
        }
        XCTAssertFalse(binder.setReference(nil,for:train,lease:lease,expectedRevision:0))
        owner.setExecutionBusy(true)
        XCTAssertFalse(binder.setReference(nil,for:train,lease:lease,expectedRevision:1))
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.rename("New name",lease:lease))
        XCTAssertFalse(binder.setReference(nil,for:train,lease:lease,expectedRevision:1))
        XCTAssertEqual(try Data(contentsOf:file(root,owner)),bytes);XCTAssertEqual(binder.revision,1)
        owner.invalidate();XCTAssertNil(binder.reference(for:train))
    }
    func testDeletedTrainBindingRetainedForUndoAndFailedWriteDoesNotPublish()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root),reference=ref(owner)
        XCTAssertTrue(binder.setReference(reference,for:train,lease:owner.lease(),expectedRevision:0))
        XCTAssertNotNil(owner.apply(.delete(.init(nodeAddresses:[train])),lease:owner.lease()))
        XCTAssertNil(binder.reference(for:train));XCTAssertEqual(binder.bindings[train],reference)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(binder.reference(for:train),reference)
        let url=file(root,owner)
        try FileManager.default.removeItem(at:url);try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertFalse(binder.setReference(nil,for:train,lease:owner.lease(),expectedRevision:1))
        XCTAssertEqual(binder.reference(for:train),reference);XCTAssertEqual(binder.revision,1);XCTAssertNotNil(binder.error)
    }
    func testUnknownNestedFieldsOversizeAndSymlinksFailWithoutReplacingSavedEvidence()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        XCTAssertTrue(binder.setReference(ref(owner),for:train,lease:owner.lease(),expectedRevision:0))
        let url=file(root,owner),good=try Data(contentsOf:url)
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:good) as? [String:Any])
        var entries=try XCTUnwrap(object["entries"] as? [[String:Any]])
        var reference=try XCTUnwrap(entries[0]["reference"] as? [String:Any]);reference["path"]="/private/not-allowed"
        entries[0]["reference"]=reference;object["entries"]=entries
        let bad=try JSONSerialization.data(withJSONObject:object)
        try bad.write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateCheckpointBindings(owner:owner,directory:root));XCTAssertEqual(try Data(contentsOf:url),bad)
        try Data(repeating:32,count:1_000_001).write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateCheckpointBindings(owner:owner,directory:root))
        try FileManager.default.removeItem(at:url)
        let external=root.appendingPathComponent("external.json");try good.write(to:external)
        try FileManager.default.createSymbolicLink(at:url,withDestinationURL:external)
        XCTAssertThrowsError(try WorkflowPrivateCheckpointBindings(owner:owner,directory:root))
        XCTAssertFalse(binder.setReference(nil,for:train,lease:owner.lease(),expectedRevision:1))
        XCTAssertEqual(try Data(contentsOf:external),good)
    }
    func testCheckpointPinsAndCacheNamespaceRemainIndependent()throws {
        let(root,_,owner)=try fixture()
        let checkpoint=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        let cache=try WorkflowPrivateCacheBindings(owner:owner,directory:root)
        let parent=ref(owner)
        XCTAssertTrue(checkpoint.setReference(parent,for:train,lease:owner.lease(),expectedRevision:0))
        let bytes=try Data(contentsOf:file(root,owner))
        for (run,request) in [("../run",parent.requestIdentity),(UUID().uuidString,"bad"),(UUID().uuidString,String(repeating:"B",count:64))] {
            let bad=WorkflowCheckpointReference(projectID:owner.projectID,directoryID:parent.directoryID,
                runID:run,requestIdentity:request,modelManifestSHA256:parent.modelManifestSHA256)
            XCTAssertFalse(checkpoint.setReference(bad,for:train,lease:owner.lease(),expectedRevision:1))
        }
        let cached=WorkflowArchiveCacheReference(projectID:owner.projectID,directoryID:UUID().uuidString,
            preparationManifestSHA256:String(repeating:"c",count:64))
        XCTAssertTrue(cache.setReference(cached,for:train,lease:owner.lease(),expectedRevision:0))
        XCTAssertEqual(try Data(contentsOf:file(root,owner)),bytes)
        XCTAssertEqual(try WorkflowPrivateCheckpointBindings(owner:owner,directory:root).reference(for:train),parent)
        XCTAssertEqual(try WorkflowPrivateCacheBindings(owner:owner,directory:root).reference(for:train),cached)
    }
    func testBindingCountAndDuplicateAddressDecodeBounds()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        XCTAssertTrue(binder.setReference(ref(owner),for:train,lease:owner.lease(),expectedRevision:0))
        let url=file(root,owner),good=try Data(contentsOf:url)
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:good) as? [String:Any])
        let original=try XCTUnwrap((object["entries"] as? [[String:Any]])?.first)
        object["entries"]=[original,original]
        try JSONSerialization.data(withJSONObject:object).write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateCheckpointBindings(owner:owner,directory:root))
        object["entries"]=(0..<257).map {index->[String:Any] in
            var entry=original;entry["address"]=["domain":"control","localID":"train-\(index)"];return entry
        }
        try JSONSerialization.data(withJSONObject:object).write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateCheckpointBindings(owner:owner,directory:root))
    }
}
