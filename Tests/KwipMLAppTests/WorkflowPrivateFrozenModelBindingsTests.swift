import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPrivateFrozenModelBindingsTests:XCTestCase {
    private let train=WorkflowNodeAddress(domain:.control,localID:"shared")
    private func fixture()throws->(URL,ProjectLibrary,WorkflowAuthoringStore) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:train,payload:.frozenModel(config:[:])),
            .init(address:.init(domain:.data,localID:"shared"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:.init(domain:.control,localID:"setup"),payload:.runSetup(config:[:]))])
        document.schemaVersion=5
        document.nodes.append(.init(address:.init(domain:.control,localID:"ordinary-train"),payload:.train(config:[:])))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        return(root,library,owner)
    }
    private func ref(_ owner:WorkflowAuthoringStore,project:String?=nil,directory:String=UUID().uuidString,pin:String=String(repeating:"a",count:64))->WorkflowFrozenModelReference {
        .init(projectID:project ?? owner.projectID,directoryID:directory,runID:UUID().uuidString,requestIdentity:String(repeating:"b",count:64),outcomeSHA256:String(repeating:"c",count:64),modelManifestSHA256:pin,preparationManifestSHA256:String(repeating:"d",count:64),checkpointSHA256:String(repeating:"e",count:64))
    }
    private func file(_ root:URL,_ owner:WorkflowAuthoringStore)->URL {root.appendingPathComponent("workflow-frozen-model/\(owner.projectID)/bindings.json")}
    func testPersistRestartClearAndNoOpDoNotMutatePortableWorkflow()throws {
        let(root,library,owner)=try fixture(),binder=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
        let before=try ProjectSourceCodec.encode(library.loadSource(id:owner.projectID)),reference=ref(owner)
        XCTAssertEqual(binder.revision,0);XCTAssertNil(binder.reference(for:train))
        XCTAssertTrue(binder.setReference(reference,for:train,lease:owner.lease(),expectedRevision:0))
        XCTAssertEqual(binder.revision,1);XCTAssertEqual(binder.reference(for:train),reference)
        XCTAssertTrue(binder.setReference(reference,for:train,lease:owner.lease(),expectedRevision:1));XCTAssertEqual(binder.revision,1)
        let restarted=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
        XCTAssertEqual(restarted.reference(for:train),reference);XCTAssertEqual(restarted.revision,1)
        XCTAssertEqual(try ProjectSourceCodec.encode(library.loadSource(id:owner.projectID)),before)
        XCTAssertTrue(restarted.setReference(nil,for:train,lease:owner.lease(),expectedRevision:1))
        XCTAssertEqual(restarted.revision,2);XCTAssertNil(restarted.reference(for:train))
    }
    func testInvalidIdentityPinAliasLeaseAndBusyPreserveSavedBytes()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
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
    func testDeletedFrozenBindingRetainedForUndoAndFailedWriteDoesNotPublish()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root),reference=ref(owner)
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
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
        XCTAssertTrue(binder.setReference(ref(owner),for:train,lease:owner.lease(),expectedRevision:0))
        let url=file(root,owner),good=try Data(contentsOf:url)
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:good) as? [String:Any])
        var entries=try XCTUnwrap(object["entries"] as? [[String:Any]])
        var reference=try XCTUnwrap(entries[0]["reference"] as? [String:Any]);reference["path"]="/private/not-allowed"
        entries[0]["reference"]=reference;object["entries"]=entries
        let bad=try JSONSerialization.data(withJSONObject:object)
        try bad.write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root));XCTAssertEqual(try Data(contentsOf:url),bad)
        try Data(repeating:32,count:1_000_001).write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root))
        try FileManager.default.removeItem(at:url)
        let external=root.appendingPathComponent("external.json");try good.write(to:external)
        try FileManager.default.createSymbolicLink(at:url,withDestinationURL:external)
        XCTAssertThrowsError(try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root))
        XCTAssertFalse(binder.setReference(nil,for:train,lease:owner.lease(),expectedRevision:1))
        XCTAssertEqual(try Data(contentsOf:external),good)
    }
    func testAllFrozenPinsAreRequiredAndTrainBindingsKeepTheirNodePolicy()throws {
        let(root,_,owner)=try fixture(),binder=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
        let reference=ref(owner)
        XCTAssertTrue(binder.setReference(reference,for:train,lease:owner.lease(),expectedRevision:0))
        let url=file(root,owner),good=try Data(contentsOf:url)
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:good) as? [String:Any])
        for key in ["runID","requestIdentity","outcomeSHA256","modelManifestSHA256","preparationManifestSHA256","checkpointSHA256"] {
            var malformed=object
            var entries=try XCTUnwrap(malformed["entries"] as? [[String:Any]])
            var selected=try XCTUnwrap(entries[0]["reference"] as? [String:Any])
            selected[key]="invalid";entries[0]["reference"]=selected;malformed["entries"]=entries
            let bytes=try JSONSerialization.data(withJSONObject:malformed)
            try bytes.write(to:url)
            XCTAssertThrowsError(try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root),key)
            XCTAssertEqual(try Data(contentsOf:url),bytes)
        }
        try good.write(to:url)
        let ordinary=WorkflowNodeAddress(domain:.control,localID:"ordinary-train")
        let checkpoint=try WorkflowPrivateCheckpointBindings(owner:owner,directory:root)
        let parent=WorkflowCheckpointReference(projectID:owner.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),modelManifestSHA256:String(repeating:"b",count:64))
        XCTAssertFalse(checkpoint.setReference(parent,for:train,lease:owner.lease(),expectedRevision:0))
        XCTAssertTrue(checkpoint.setReference(parent,for:ordinary,lease:owner.lease(),expectedRevision:0))
        let cache=try WorkflowPrivateCacheBindings(owner:owner,directory:root)
        let cached=WorkflowArchiveCacheReference(projectID:owner.projectID,directoryID:UUID().uuidString,preparationManifestSHA256:String(repeating:"f",count:64))
        XCTAssertFalse(cache.setReference(cached,for:train,lease:owner.lease(),expectedRevision:0))
        XCTAssertTrue(cache.setReference(cached,for:ordinary,lease:owner.lease(),expectedRevision:0))
        XCTAssertEqual(try Data(contentsOf:url),good)
        XCTAssertEqual(try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root).reference(for:train),reference)
    }
}
