import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPrivateSourceBindingsTests: XCTestCase {
    private let source=WorkflowNodeAddress(domain:.data,localID:"source")
    @MainActor private func setup() throws -> (URL,WorkflowAuthoringStore) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:.init(domain:.model,localID:"source"),payload:.modelPrimitive(type:"input",configurationVersion:2,config:["features":2])),
            .init(address:.init(domain:.data,localID:"cast"),payload:.dataOperator(operatorID:"kwip.data.cast",config:[:]))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        return(root,owner)
    }
    private func file(_ root:URL,_ id:String)->URL {root.appendingPathComponent("workflow-sources").appendingPathComponent(id).appendingPathComponent("bindings.json")}
    @MainActor func testPrivateRoundTripNoOpUnlinkAndPortableIsolation() throws {
        let(root,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let store=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        let before=owner.document,lease=owner.lease()
        XCTAssertEqual(store.revision,0);XCTAssertTrue(store.bindings.isEmpty)
        XCTAssertTrue(store.setPath("/private/observations.csv",for:source,lease:lease,expectedRevision:0))
        XCTAssertEqual(store.revision,1);XCTAssertEqual(store.path(for:source),"/private/observations.csv")
        let bytes=try Data(contentsOf:file(root,owner.projectID))
        XCTAssertTrue(store.setPath("/private/observations.csv",for:source,lease:lease,expectedRevision:1))
        XCTAssertEqual(store.revision,1);XCTAssertEqual(try Data(contentsOf:file(root,owner.projectID)),bytes)
        let reopened=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        XCTAssertEqual(reopened.bindings,store.bindings);XCTAssertEqual(reopened.revision,1)
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease);XCTAssertFalse(owner.canUndo)
        XCTAssertFalse(String(decoding:try WorkflowCodec.encode(owner.document),as:UTF8.self).contains("/private/"))
        XCTAssertTrue(store.setPath(nil,for:source,lease:lease,expectedRevision:1))
        XCTAssertNil(store.path(for:source));XCTAssertEqual(store.revision,2)
    }
    @MainActor func testLeaseBindingRevisionBusyAndNodeContractGuards() throws {
        let(root,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let store=try WorkflowPrivateSourceBindings(owner:owner,directory:root),old=owner.lease()
        XCTAssertNotNil(owner.apply(.rename(address:source,name:"Renamed"),lease:old))
        XCTAssertFalse(store.setPath("/tmp/a.csv",for:source,lease:old,expectedRevision:0))
        for address in [WorkflowNodeAddress(domain:.model,localID:"source"),.init(domain:.data,localID:"cast"),.init(domain:.data,localID:"missing")] {
            XCTAssertFalse(store.setPath("/tmp/a.csv",for:address,lease:owner.lease(),expectedRevision:0))
        }
        for path in ["relative.csv","/tmp/a\u{0}.csv",String(repeating:"/",count:4097)] {
            XCTAssertFalse(store.setPath(path,for:source,lease:owner.lease(),expectedRevision:0))
        }
        owner.setExecutionBusy(true)
        XCTAssertFalse(store.setPath("/tmp/a.csv",for:source,lease:owner.lease(),expectedRevision:0))
        owner.setExecutionBusy(false)
        XCTAssertTrue(store.setPath("/tmp/a.csv",for:source,lease:owner.lease(),expectedRevision:0))
        XCTAssertFalse(store.setPath("/tmp/late.csv",for:source,lease:owner.lease(),expectedRevision:0))
        XCTAssertEqual(store.path(for:source),"/tmp/a.csv")
        owner.invalidate()
        XCTAssertFalse(store.setPath("/tmp/retired.csv",for:source,lease:owner.lease(),expectedRevision:1))
        XCTAssertNil(store.path(for:source))
    }
    @MainActor func testDeletedBindingsStayPrivateAndReactivateOnlyWithExactSourceUndo() throws {
        let(root,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let store=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        XCTAssertTrue(store.setPath("/tmp/a.csv",for:source,lease:owner.lease(),expectedRevision:0))
        XCTAssertNotNil(owner.apply(.delete(.init(nodeAddresses:[source])),lease:owner.lease()))
        XCTAssertNil(store.path(for:source));XCTAssertEqual(store.bindings[source],"/tmp/a.csv")
        let reopened=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        XCTAssertNil(reopened.path(for:source));XCTAssertEqual(reopened.revision,1)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(reopened.path(for:source),"/tmp/a.csv")
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let imported=try library.importSource(from:library.documentURL(id:owner.projectID))
        let newOwner=try WorkflowAuthoringStore(library:library,projectID:imported.id,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        XCTAssertTrue(try WorkflowPrivateSourceBindings(owner:newOwner,directory:root).bindings.isEmpty)
    }
    @MainActor func testFailedAtomicSaveDoesNotPublishOrFollowFileAndDirectorySymlinks() throws {
        let(root,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let store=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        XCTAssertTrue(store.setPath("/tmp/a.csv",for:source,lease:owner.lease(),expectedRevision:0))
        let url=file(root,owner.projectID),bytes=try Data(contentsOf:url),external=root.appendingPathComponent("external.json")
        try bytes.write(to:external);try FileManager.default.removeItem(at:url)
        try FileManager.default.createSymbolicLink(at:url,withDestinationURL:external)
        XCTAssertFalse(store.setPath("/tmp/b.csv",for:source,lease:owner.lease(),expectedRevision:1))
        XCTAssertEqual(store.revision,1);XCTAssertEqual(store.path(for:source),"/tmp/a.csv");XCTAssertNotNil(store.error)
        XCTAssertEqual(try Data(contentsOf:external),bytes)
        XCTAssertThrowsError(try WorkflowPrivateSourceBindings(owner:owner,directory:root))
        try FileManager.default.removeItem(at:root.appendingPathComponent("workflow-sources"))
        let outside=root.appendingPathComponent("outside");try FileManager.default.createDirectory(at:outside,withIntermediateDirectories:true)
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("workflow-sources"),withDestinationURL:outside)
        XCTAssertThrowsError(try WorkflowPrivateSourceBindings(owner:owner,directory:root))
        XCTAssertFalse(store.setPath("/tmp/b.csv",for:source,lease:owner.lease(),expectedRevision:1))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath:outside.path).isEmpty)
    }
    @MainActor func testStrictEnvelopeBoundsProjectIdentityAndRevisionExhaustion() throws {
        let(root,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let url=file(root,owner.projectID)
        try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
        let entry:[String:Any]=["address":["domain":"data","localID":"source"],"path":"/tmp/a.csv"]
        let base:[String:Any]=["schemaVersion":1,"projectID":owner.projectID,"revision":1,"entries":[entry]]
        var variants:[[String:Any]]=[]
        var wrong=base;wrong["projectID"]=UUID().uuidString;variants.append(wrong)
        wrong=base;wrong["schemaVersion"]=2;variants.append(wrong)
        wrong=base;wrong["extra"]=true;variants.append(wrong)
        wrong=base;wrong["entries"]=[entry,entry];variants.append(wrong)
        wrong=base;wrong["entries"]=(0..<129).map{["address":["domain":"data","localID":"inactive-\($0)"],"path":"/tmp/a.csv"]};variants.append(wrong)
        for object in variants {
            try JSONSerialization.data(withJSONObject:object).write(to:url)
            XCTAssertThrowsError(try WorkflowPrivateSourceBindings(owner:owner,directory:root))
        }
        try Data(repeating:32,count:1_000_001).write(to:url)
        XCTAssertThrowsError(try WorkflowPrivateSourceBindings(owner:owner,directory:root))
        let exhausted="{\"schemaVersion\":1,\"projectID\":\"\(owner.projectID)\",\"revision\":18446744073709551615,\"entries\":[]}"
        try Data(exhausted.utf8).write(to:url)
        let store=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        XCTAssertFalse(store.setPath("/tmp/a.csv",for:source,lease:owner.lease(),expectedRevision:UInt64.max))
        XCTAssertEqual(store.revision,UInt64.max);XCTAssertTrue(store.bindings.isEmpty)
    }
    func testSharedAtomicWriterKeepsImmutableCommitExclusive() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let parts=["private","entry.json"],first=Data("first".utf8),second=Data("second".utf8)
        try DiagnosticFileIO.writeImmutable(first,root:root,parts:parts)
        XCTAssertThrowsError(try DiagnosticFileIO.writeImmutable(second,root:root,parts:parts))
        XCTAssertEqual(try DiagnosticFileIO.read(root:root,parts:parts),first)
        try DiagnosticFileIO.writeAtomic(second,root:root,parts:parts)
        XCTAssertEqual(try DiagnosticFileIO.read(root:root,parts:parts),second)
        XCTAssertThrowsError(try DiagnosticFileIO.writeImmutable(first,root:root,parts:parts))
        XCTAssertEqual(try DiagnosticFileIO.read(root:root,parts:parts),second)
    }

}
