import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTrainingModeEditingTests:XCTestCase {
    private let setup=WorkflowNodeAddress(domain:.control,localID:"setup")
    private let second=WorkflowNodeAddress(domain:.control,localID:"other")
    private func fixture()throws->(ProjectLibrary,WorkflowAuthoringStore) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let original=ModelAuthoringDocument(projectID:UUID().uuidString,source:CompositeExamples.parallelClassifier)
        var document=try WorkflowCodec.migrateModel(.blockModel(original),projectID:original.projectID).document
        document.name="User named source";document.revision=9
        document.nodes.append(.init(address:setup,name:"Training budget",x:123,y:456,payload:.train(config:[:])))
        document.nodes.append(.init(address:second,name:"Other path",payload:.train(config:[:])))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        _=owner.select(.init(nodeAddresses:[setup]),lease:owner.lease())
        return(library,owner)
    }
    func testGenuineModeChangeMigratesOnceAndSingleUndoRestoresLegacySource()throws {
        let(library,owner)=try fixture(),before=owner.document
        XCTAssertEqual(before.schemaVersion,1)
        let selection=owner.selection
        XCTAssertTrue(owner.setTrainingMode(.exactResume,for:setup,lease:owner.lease()))
        var expected=try WorkflowCodec.migrateToVersion3(before)
        let index=try XCTUnwrap(expected.nodes.firstIndex {$0.address == setup})
        guard case .train(var config)=expected.nodes[index].payload else {return XCTFail("Run setup fixture")}
        config["trainingMode"] = .string("exactResume")
        expected.nodes[index].payload = .train(config:config);expected.revision=before.revision+1
        XCTAssertEqual(owner.document,expected)
        XCTAssertEqual(owner.selection,selection)
        XCTAssertEqual(try library.loadSource(id:owner.projectID),.workflow(expected))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        var undone=before;undone.revision=expected.revision+1
        XCTAssertEqual(owner.document,undone);XCTAssertFalse(owner.canUndo)
        XCTAssertTrue(owner.redo(lease:owner.lease()))
        expected.revision=undone.revision+1;XCTAssertEqual(owner.document,expected)
    }
    func testLegacyFreshNoOpDoesNotMigrateWriteOrCreateHistory()throws {
        let(library,owner)=try fixture()
        let before=owner.document,lease=owner.lease(),url=try library.documentURL(id:owner.projectID)
        let bytes=try Data(contentsOf:url)
        XCTAssertTrue(owner.setTrainingMode(.fresh,for:setup,lease:lease))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease)
        XCTAssertFalse(owner.canUndo);XCTAssertEqual(try Data(contentsOf:url),bytes)
    }
    func testBusyStaleWrongDomainAndOpenBlockDraftRejectWithoutAlteringDraft()throws {
        let(library,owner)=try fixture(),lease=owner.lease()
        owner.setExecutionBusy(true)
        XCTAssertFalse(owner.setTrainingMode(.fineTune,for:setup,lease:lease))
        owner.setExecutionBusy(false)
        XCTAssertFalse(owner.setTrainingMode(.fineTune,for:.init(domain:.model,localID:"setup"),lease:lease))
        XCTAssertFalse(owner.setTrainingMode(.fineTune,for:.init(domain:.model,localID:"features"),lease:lease))
        XCTAssertTrue(owner.rename("Changed metadata",lease:lease))
        XCTAssertFalse(owner.setTrainingMode(.fineTune,for:setup,lease:lease))
        XCTAssertTrue(owner.beginDraft(root:.init(domain:.model,localID:"features"),lease:owner.lease()))
        let before=owner.document,url=try library.documentURL(id:owner.projectID),bytes=try Data(contentsOf:url)
        XCTAssertFalse(owner.setTrainingMode(.exactResume,for:setup,lease:owner.lease()))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(try Data(contentsOf:url),bytes)
    }

    func testFailedSavePreservesSourceLeaseAndHistory()throws {
        let(library,owner)=try fixture()
        let before=owner.document,lease=owner.lease(),url=try library.documentURL(id:owner.projectID)
        let bytes=try Data(contentsOf:url)
        try FileManager.default.removeItem(at:url)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertFalse(owner.setTrainingMode(.fineTune,for:setup,lease:lease))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease);XCTAssertFalse(owner.canUndo)
        try FileManager.default.removeItem(at:url);try bytes.write(to:url)
        XCTAssertTrue(owner.setTrainingMode(.fineTune,for:setup,lease:lease))
        XCTAssertEqual(owner.document.schemaVersion,3)
    }

    func testContextModeCommitRenewsAuthorityAndRejectsStaleOrPendingEdits()throws {
        let(_,owner)=try fixture(),context=try WorkflowControlNodeContext(owner:owner,address:setup)
        XCTAssertEqual(context.trainingMode,.fresh)
        let original=context.configuration,identity=context.contextIdentity,origin=context.numericOrigin(key:"trainingMode")
        context.setNumericPending(origin:origin,key:"trainingMode",pending:true)
        XCTAssertFalse(context.setTrainingMode(.exactResume,expected:original))
        XCTAssertEqual(owner.document.schemaVersion,1)
        context.setNumericPending(origin:origin,key:"trainingMode",pending:false)
        XCTAssertTrue(context.setTrainingMode(.exactResume,expected:original))
        XCTAssertTrue(context.isCurrent);XCTAssertEqual(context.contextIdentity,identity)
        XCTAssertEqual(context.trainingMode,.exactResume)
        XCTAssertFalse(context.setTrainingMode(.fineTune,expected:original))
        XCTAssertEqual(context.trainingMode,.exactResume)
        context.retire()
        XCTAssertNil(context.trainingMode)
        XCTAssertFalse(context.setTrainingMode(.fresh,expected:context.configuration))
    }

}
