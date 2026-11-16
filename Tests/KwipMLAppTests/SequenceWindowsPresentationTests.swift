import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SequenceWindowsPresentationTests:XCTestCase {
    func testAlignedPairMaskVocabularySearchAndBadCapture()throws {
        var c=SequenceWindowsConfiguration.defaults;c["contextLength"]=3;c["incomplete"] = .string("pad")
        let columns=SequenceWindowsConfiguration.columns(c).enumerated().map {CapturedDataGrid.Column(id:$0.offset,name:$0.element.name,label:$0.element.name,dtype:$0.element.dtype,nullability:"required")}
        func row(_ mask:[Int64])->CapturedDataGrid.Row {.init(index:0,cells:[.init(text:"0",isAbsent:false),.init(text:"a",isAbsent:false),.init(text:"0",isAbsent:false),.init(text:"source:0",isAbsent:false),.init(text:"3 tokens",isAbsent:false,sequence:[2,3,1]),.init(text:"3 tokens",isAbsent:false,sequence:[3,4,1]),.init(text:"2",isAbsent:false),.init(text:"3 tokens",isAbsent:false,sequence:mask)])}
        let pair=try SequenceWindowSelection(row:row([1,1,0]),columns:columns,configuration:c,vocabulary:["<unk>","<pad>","T","o"," "])
        XCTAssertEqual(pair.inputs,[2,3,1]);XCTAssertEqual(pair.targets,[3,4,1]);XCTAssertEqual(pair.token(4),"\" \"")
        XCTAssertEqual(pair.positions(matching:"\"T\""),[0]);XCTAssertEqual(pair.positions(matching:"4"),[1]);XCTAssertEqual(pair.mask,[1,1,0])
        XCTAssertThrowsError(try SequenceWindowSelection(row:row([1,1,1]),columns:columns,configuration:c,vocabulary:nil))
        XCTAssertThrowsError(try SequenceWindowSelection(row:row([1,1,0]),columns:columns,configuration:c,vocabulary:["<unk>"]))
        XCTAssertEqual(try SequenceWindowSelection(row:row([1,1,0]),columns:columns,configuration:c,vocabulary:nil).token(2),"Unavailable")
    }
    func testChangingPortModePreservesConnectedWorkflowAndDraftAuthority()throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:directory)}
        let document=try SequenceWindowsPracticeExample.document(),library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let address=SequenceWindowsPracticeExample.address
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:SequenceSourceProbe(owner:owner))
        let original=try XCTUnwrap(context.node?.config),before=owner.document
        var changed=original;changed["mode"] = .string("partitions")
        XCTAssertFalse(context.applyTableConfiguration(changed,expected:original))
        XCTAssertEqual(owner.document,before);XCTAssertTrue(context.statusMessage.contains("Disconnect"))
        changed=original;changed["stride"]=4
        XCTAssertTrue(context.applyTableConfiguration(changed,expected:original),context.statusMessage)
        XCTAssertEqual(owner.document.edges,before.edges)
        XCTAssertFalse(context.applyTableConfiguration(original,expected:original))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(context.isCurrent)
    }
    func testKnownSchemaProjectionAndValidation()throws {
        let schema:[DataColumnSchema]=[.init(name:"tokens",dtype:"int64Sequence",nullable:false),.init(name:"document",dtype:"string",nullable:false),.init(name:"fragment",dtype:"int64",nullable:false)]
        var c=SequenceWindowsConfiguration.defaults;c["scope"] = .string("document");c["documentColumn"] = .string("document");c["orderColumn"] = .string("fragment")
        XCTAssertEqual(try DataTableConfiguration.validate(c,operatorID:SequenceWindowsConfiguration.operatorID,schema:schema),c)
        c["orderColumn"] = .string("absent");XCTAssertThrowsError(try DataTableConfiguration.validate(c,operatorID:SequenceWindowsConfiguration.operatorID,schema:schema))
        XCTAssertEqual(ComponentLessonDestination.entryID(type:SequenceWindowsConfiguration.operatorID),"sequence-windows")
    }
}

@MainActor private final class SequenceSourceProbe:WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage="Recorded source"
    var changes:AnyPublisher<Void,Never> {Empty(completeImmediately:false).eraseToAnyPublisher()}
    init(owner:WorkflowAuthoringStore){projectID=owner.projectID;sessionID=owner.sessionID}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] {[.init(name:"document",dtype:"string"),.init(name:"fragment",dtype:"int64"),.init(name:"text",dtype:"string"),.init(name:"tokens",dtype:"int64Sequence")]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {nil}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
}
