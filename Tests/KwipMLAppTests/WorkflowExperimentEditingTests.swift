import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExperimentEditingTests:XCTestCase {
    private let setup=WorkflowNodeAddress(domain:.control,localID:"setup")
    private let second=WorkflowNodeAddress(domain:.control,localID:"other")
    private func fixture()throws->(ProjectLibrary,WorkflowAuthoringStore) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let original=ModelAuthoringDocument(projectID:UUID().uuidString,source:CompositeExamples.parallelClassifier)
        var document=try WorkflowCodec.migrateModel(.blockModel(original),projectID:original.projectID).document
        document.name="User named source";document.revision=9
        document.nodes.append(.init(address:setup,name:"Training budget",x:123,y:456,payload:.runSetup(config:["seed":.integer(42)])))
        document.nodes.append(.init(address:second,name:"Other path",payload:.runSetup(config:["seed":.integer(42)])))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        _=owner.select(.init(nodeAddresses:[setup]),lease:owner.lease())
        return(library,owner)
    }
    private var specification:WorkflowExperimentSpecification {
        .init(axes:[.init(field:"seed",values:[.integer(7),.integer(8)])],
              budget:.init(maximumRuns:2,maximumTotalEpochs:100))
    }
    func testSaveClearUndoRedoPreserveOrdinarySettingsAndOtherNodes()throws {
        let(library,owner)=try fixture(),before=owner.document,selection=owner.selection
        XCTAssertTrue(owner.setExperimentSpecification(specification,for:setup,lease:owner.lease()))
        var expected=try WorkflowCodec.migrateToVersion4(before)
        let index=try XCTUnwrap(expected.nodes.firstIndex {$0.address == setup})
        let encoded=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(specification))
        expected.nodes[index].payload = .runSetup(config:["seed":.integer(42),"experiment":encoded])
        expected.revision += 1
        XCTAssertEqual(owner.document,expected);XCTAssertEqual(owner.selection,selection)
        XCTAssertEqual(try library.loadSource(id:owner.projectID),.workflow(expected))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        var undone=before;undone.revision=expected.revision+1
        XCTAssertEqual(owner.document,undone)
        XCTAssertTrue(owner.redo(lease:owner.lease()))
        expected.revision=undone.revision+1;XCTAssertEqual(owner.document,expected)
        XCTAssertTrue(owner.setExperimentSpecification(nil,for:setup,lease:owner.lease()))
        XCTAssertEqual(owner.document.schemaVersion,4)
        XCTAssertEqual(owner.document.nodes[index].payload,before.nodes[index].payload)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        expected.revision += 2;XCTAssertEqual(owner.document,expected)
    }
    func testNoOpAndInvalidSpecificationDoNotWriteOrRenewLease()throws {
        let(library,owner)=try fixture(),before=owner.document,lease=owner.lease()
        let url=try library.documentURL(id:owner.projectID),bytes=try Data(contentsOf:url)
        XCTAssertTrue(owner.setExperimentSpecification(nil,for:setup,lease:lease))
        let invalid=WorkflowExperimentSpecification(axes:[],budget:.init(maximumRuns:2,maximumTotalEpochs:10))
        XCTAssertFalse(owner.setExperimentSpecification(invalid,for:setup,lease:lease))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease)
        XCTAssertEqual(try Data(contentsOf:url),bytes);XCTAssertFalse(owner.canUndo)
        XCTAssertTrue(owner.setExperimentSpecification(specification,for:setup,lease:lease))
        let saved=owner.document,current=owner.lease()
        XCTAssertTrue(owner.setExperimentSpecification(specification,for:setup,lease:current))
        XCTAssertEqual(owner.document,saved);XCTAssertEqual(owner.lease(),current)
    }
    func testBusyStaleWrongAddressPendingAndDraftReject()throws {
        let(_,owner)=try fixture(),lease=owner.lease()
        owner.setExecutionBusy(true)
        XCTAssertFalse(owner.setExperimentSpecification(specification,for:setup,lease:lease))
        owner.setExecutionBusy(false)
        XCTAssertFalse(owner.setExperimentSpecification(specification,for:.init(domain:.model,localID:"features"),lease:lease))
        let context=UUID()
        XCTAssertTrue(owner.setInspectorEditPending(contextID:context,address:setup,key:"seed",pending:true,lease:lease))
        XCTAssertFalse(owner.setExperimentSpecification(specification,for:setup,lease:lease))
        owner.clearInspectorEdits(contextID:context)
        XCTAssertTrue(owner.rename("Changed",lease:lease))
        XCTAssertFalse(owner.setExperimentSpecification(specification,for:setup,lease:lease))
        XCTAssertTrue(owner.beginDraft(root:.init(domain:.model,localID:"features"),lease:owner.lease()))
        let before=owner.document
        XCTAssertFalse(owner.setExperimentSpecification(specification,for:setup,lease:owner.lease()))
        XCTAssertEqual(owner.document,before)
    }
    func testFailedPersistenceKeepsHistoryAndSourceUntouched()throws {
        let(library,owner)=try fixture(),before=owner.document,lease=owner.lease()
        let url=try library.documentURL(id:owner.projectID),bytes=try Data(contentsOf:url)
        try FileManager.default.removeItem(at:url)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertFalse(owner.setExperimentSpecification(specification,for:setup,lease:lease))
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease);XCTAssertFalse(owner.canUndo)
        try FileManager.default.removeItem(at:url);try bytes.write(to:url)
        XCTAssertTrue(owner.setExperimentSpecification(specification,for:setup,lease:lease))
    }
    func testInspectorAuthorityRenewsAndRejectsOutdatedConfiguration()throws {
        let(_,owner)=try fixture(),context=try WorkflowControlNodeContext(owner:owner,address:setup)
        let original=context.configuration,identity=context.contextIdentity
        XCTAssertTrue(context.setExperimentSpecification(specification,expected:original))
        XCTAssertEqual(context.experimentSpecification,specification)
        XCTAssertTrue(context.isCurrent);XCTAssertEqual(context.contextIdentity,identity)
        XCTAssertFalse(context.setExperimentSpecification(nil,expected:original))
        context.retire()
        XCTAssertNil(context.experimentSpecification)
        XCTAssertFalse(context.setExperimentSpecification(nil,expected:context.configuration))
    }
}
