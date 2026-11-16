import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCachePolicyEditingTests:XCTestCase {
    private let setup=WorkflowNodeAddress(domain:.control,localID:"setup")
    private let second=WorkflowNodeAddress(domain:.control,localID:"other")
    private func fixture()throws->(ProjectLibrary,WorkflowAuthoringStore) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let original=ModelAuthoringDocument(projectID:UUID().uuidString,source:CompositeExamples.parallelClassifier)
        var document=try WorkflowCodec.migrateModel(.blockModel(original),projectID:original.projectID).document
        document.name="User named source";document.revision=9
        document.nodes.append(.init(address:setup,name:"Training budget",x:123,y:456,payload:.runSetup(config:["epochs":12,"learningRate":.number(0.02)])))
        document.nodes.append(.init(address:second,name:"Other path",payload:.runSetup(config:["seed":8])))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        _=owner.select(.init(nodeAddresses:[setup]),lease:owner.lease())
        return(library,owner)
    }
    func testGenuinePolicyChangeMigratesOnceAndSingleUndoRestoresLegacySource()throws {
        let(library,owner)=try fixture(),before=owner.document
        XCTAssertEqual(before.schemaVersion,1)
        let selection=owner.selection
        XCTAssertTrue(owner.setPreparationCachePolicy(.materialize,for:setup,lease:owner.lease()))
        var expected=try WorkflowCodec.migrateToVersion2(before)
        let index=try XCTUnwrap(expected.nodes.firstIndex {$0.address == setup})
        guard case .runSetup(var config)=expected.nodes[index].payload else {return XCTFail("Run setup fixture")}
        config["preparationCachePolicy"] = .string("materialize")
        expected.nodes[index].payload = .runSetup(config:config);expected.revision=before.revision+1
        XCTAssertEqual(owner.document,expected)
        XCTAssertEqual(owner.selection,selection)
        XCTAssertEqual(try library.loadSource(id:owner.projectID),.workflow(expected))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        var undone=before;undone.revision=expected.revision+1
        XCTAssertEqual(owner.document,undone);XCTAssertFalse(owner.canUndo)
        XCTAssertTrue(owner.redo(lease:owner.lease()))
        expected.revision=undone.revision+1;XCTAssertEqual(owner.document,expected)
    }
    func testLegacyOffNoOpDoesNotMigrateWriteOrCreateHistory()throws {
        let(library,owner)=try fixture()
        let before=owner.document,lease=owner.lease(),url=try library.documentURL(id:owner.projectID)
        let bytes=try Data(contentsOf:url)
        XCTAssertTrue(owner.setPreparationCachePolicy(.off,for:setup,lease:lease))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease)
        XCTAssertFalse(owner.canUndo);XCTAssertEqual(try Data(contentsOf:url),bytes)
    }
    func testContextPolicyCommitRenewsAuthorityButPendingNumericEditsVetoMigration()throws {
        let(_,owner)=try fixture(),context=try WorkflowControlNodeContext(owner:owner,address:setup)
        XCTAssertEqual(context.preparationCachePolicy,.off)
        let original=context.configuration,identity=context.contextIdentity,origin=context.numericOrigin(key:"epochs")
        context.setNumericPending(origin:origin,key:"epochs",pending:true)
        XCTAssertFalse(context.setPreparationCachePolicy(.reuse,expected:original))
        XCTAssertTrue(owner.hasPendingInspectorEdits);XCTAssertEqual(owner.document.schemaVersion,1)
        XCTAssertFalse(owner.setPreparationCachePolicy(.reuse,for:setup,lease:owner.lease()),"Direct owner calls must not bypass unfinished inspector values.")
        context.setNumericPending(origin:origin,key:"epochs",pending:false)
        XCTAssertTrue(context.setPreparationCachePolicy(.reuse,expected:original))
        XCTAssertTrue(context.isCurrent);XCTAssertEqual(context.contextIdentity,identity)
        XCTAssertEqual(context.preparationCachePolicy,.reuse)
        XCTAssertFalse(context.setPreparationCachePolicy(.off,expected:original))
        XCTAssertEqual(context.preparationCachePolicy,.reuse)
    }
    func testBusyStaleWrongDomainAndOpenBlockDraftRejectWithoutAlteringDraft()throws {
        let(library,owner)=try fixture(),lease=owner.lease()
        owner.setExecutionBusy(true)
        XCTAssertFalse(owner.setPreparationCachePolicy(.reuse,for:setup,lease:lease))
        owner.setExecutionBusy(false)
        XCTAssertFalse(owner.setPreparationCachePolicy(.reuse,for:.init(domain:.model,localID:"setup"),lease:lease))
        XCTAssertFalse(owner.setPreparationCachePolicy(.reuse,for:.init(domain:.model,localID:"features"),lease:lease))
        XCTAssertTrue(owner.rename("Changed metadata",lease:lease))
        XCTAssertFalse(owner.setPreparationCachePolicy(.reuse,for:setup,lease:lease))
        XCTAssertTrue(owner.beginDraft(root:.init(domain:.model,localID:"features"),lease:owner.lease()))
        let before=owner.document,url=try library.documentURL(id:owner.projectID),bytes=try Data(contentsOf:url)
        XCTAssertFalse(owner.setPreparationCachePolicy(.materialize,for:setup,lease:owner.lease()))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(try Data(contentsOf:url),bytes)
    }
    func testFailedPersistenceDoesNotPublishMigrationHistoryOrRetireContext()throws {
        let(library,owner)=try fixture(),context=try WorkflowControlNodeContext(owner:owner,address:setup)
        let before=owner.document,lease=owner.lease(),url=try library.documentURL(id:owner.projectID)
        let bytes=try Data(contentsOf:url)
        try FileManager.default.removeItem(at:url);try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertFalse(context.setPreparationCachePolicy(.materialize,expected:context.configuration))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease)
        XCTAssertFalse(owner.canUndo);XCTAssertTrue(context.isCurrent);XCTAssertNotNil(owner.error)
        try FileManager.default.removeItem(at:url);try bytes.write(to:url,options:.atomic)
        XCTAssertTrue(context.setPreparationCachePolicy(.materialize,expected:context.configuration))
        XCTAssertEqual(owner.document.schemaVersion,2);XCTAssertTrue(context.isCurrent)
    }
}
