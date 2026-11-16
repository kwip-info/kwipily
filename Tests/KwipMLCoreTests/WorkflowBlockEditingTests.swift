import XCTest
@testable import KwipMLCore

final class WorkflowBlockEditingTests: XCTestCase {
    private let rootAddress=WorkflowNodeAddress(domain:.model,localID:"features")
    private func fixture(nested:Bool=false) throws -> WorkflowDocument {
        var source=CompositeExamples.parallelClassifier
        if nested {
            let pin=source.root.instances[0].definition
            let wrapper=CompositeDefinition(reference:.init("wrapper"),title:"Wrapper",body:.init(instances:[.init(id:"nested",definition:pin)]),inputs:[.init("in",targets:[.init("nested","in")])],outputs:[.init("out",source:.init("nested","out"))])
            source.definitions.append(wrapper);source.root.instances[0].definition=wrapper.reference
        }
        let model=ModelAuthoringDocument(projectID:UUID().uuidString,revision:4,source:source)
        var workflow=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        workflow.nodes += [
            .init(address:.init(domain:.data,localID:"features"),name:"Data same ID",x:901,y:902,payload:.dataOperator(operatorID:"future.operator",config:["untouched":.array([1,true])])),
            .init(address:.init(domain:.control,localID:"setup"),name:"Run",payload:.runSetup(config:["seed":.integer(9007199254740993)])),
            .init(address:.init(domain:.control,localID:"definition"),payload:.modelDefinition)
        ]
        workflow.edges.append(.init(id:"virtual",from:.init(address:.init(domain:.model,localID:"output"),port:"definition"),to:.init(address:.init(domain:.control,localID:"definition"),port:"source"),kind:.modelDefinition))
        // Mixed versions are editable source, not a reason for this operation to migrate primitives.
        if let i=workflow.nodes.firstIndex(where:{if case .modelPrimitive = $0.payload {return true};return false}),case .modelPrimitive(let type,_,let config)=workflow.nodes[i].payload {
            workflow.nodes[i].payload = .modelPrimitive(type:type,configurationVersion:1,config:config)
        }
        return workflow
    }
    func testDraftLifecyclePreservesMixedSourceAndRejectsQualifiedOrStaleContext() throws {
        let original=try fixture()
        var begun=try WorkflowBlockEditing.beginDraft(original,root:rootAddress)
        XCTAssertEqual(begun.nodes,original.nodes);XCTAssertEqual(begun.edges,original.edges)
        XCTAssertEqual(begun.definitions,original.definitions);XCTAssertEqual(begun.revision,4)
        XCTAssertEqual(try WorkflowBlockEditing.beginDraft(begun,root:rootAddress),begun)
        var proposal=begun.drafts[0].proposed;proposal.body.edges=[]
        begun=try WorkflowBlockEditing.updateDraft(begun,proposed:proposal,expectedBaseRevision:4,root:rootAddress)
        XCTAssertEqual(begun.nodes,original.nodes);XCTAssertEqual(begun.definitions,original.definitions)
        XCTAssertEqual(begun.drafts[0].proposed,proposal)
        XCTAssertThrowsError(try WorkflowBlockEditing.applyDraft(begun,root:rootAddress,expectedBaseRevision:4))
        XCTAssertThrowsError(try WorkflowBlockEditing.discardDraft(begun,root:rootAddress,expectedBaseRevision:3))
        XCTAssertThrowsError(try WorkflowBlockEditing.beginDraft(original,root:.init(domain:.data,localID:"features")))
        XCTAssertThrowsError(try WorkflowBlockEditing.beginDraft(original,root:.init(domain:.model,localID:"output")))
        XCTAssertEqual(try WorkflowBlockEditing.discardDraft(begun,root:rootAddress,expectedBaseRevision:4),original)
    }
    func testNestedApplyChangesOnlySelectedRootPinDefinitionsDraftsAndOneRevision() throws {
        var original=try fixture(nested:true)
        let selected=try XCTUnwrap(original.nodes.first{$0.address==rootAddress})
        guard case .modelBlock(let pin,let overrides)=selected.payload else {return XCTFail("Expected Block")}
        original.nodes.append(.init(address:.init(domain:.model,localID:"other"),payload:.modelBlock(reference:pin,overrides:overrides)))
        var begun=try WorkflowBlockEditing.beginDraft(original,root:rootAddress,nestedPath:["nested"])
        var proposal=begun.drafts[0].proposed;proposal.title="Changed nested definition"
        begun=try WorkflowBlockEditing.updateDraft(begun,proposed:proposal,expectedBaseRevision:4,root:rootAddress,nestedPath:["nested"])
        var serial=0
        let applied=try WorkflowBlockEditing.applyDraft(begun,root:rootAddress,nestedPath:["nested"],expectedBaseRevision:4,idProvider:{serial+=1;return "private-\(serial)"})
        var expected=original
        expected.revision=5
        let index=try XCTUnwrap(expected.nodes.firstIndex{$0.address==rootAddress})
        expected.nodes[index].payload = .modelBlock(reference:.init("private-2"),overrides:overrides)
        expected.definitions=applied.definitions
        XCTAssertEqual(applied,expected)
        XCTAssertEqual(Array(applied.definitions.prefix(original.definitions.count)),original.definitions)
        XCTAssertEqual(applied.definitions.count,original.definitions.count+2)
        XCTAssertEqual(applied.definitions.last?.body.instances.first?.definition.id,"private-1")
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(original)),original,"An external undo snapshot retains every original source field")
        XCTAssertEqual(original.revision,4)
    }
    func testApplyCannotStrandIncidentWrongKindOrCrossDomainWire() throws {
        var original=try fixture()
        original.edges.removeAll{$0.from.address==rootAddress}
        original.edges.append(.init(id:"unresolved-wire",from:.init(address:rootAddress,port:"out"),to:.init(address:.init(domain:.control,localID:"future"),port:"input"),kind:.table))
        var begun=try WorkflowBlockEditing.beginDraft(original,root:rootAddress)
        var proposal=begun.drafts[0].proposed;proposal.outputs[0].id="new-output"
        begun=try WorkflowBlockEditing.updateDraft(begun,proposed:proposal,expectedBaseRevision:4,root:rootAddress)
        XCTAssertThrowsError(try WorkflowBlockEditing.applyDraft(begun,root:rootAddress,expectedBaseRevision:4,idProvider:{"copy"}))
        XCTAssertEqual(begun.nodes,original.nodes)
        // A wire on a data node with the same local ID is not a Block boundary obligation.
        original.edges.removeLast()
        original.edges.append(.init(id:"data-wire",from:.init(address:.init(domain:.data,localID:"features"),port:"unrelated"),to:.init(address:.init(domain:.control,localID:"future"),port:"input"),kind:.table))
        var independent=try WorkflowBlockEditing.beginDraft(original,root:rootAddress)
        independent=try WorkflowBlockEditing.updateDraft(independent,proposed:proposal,expectedBaseRevision:4,root:rootAddress)
        XCTAssertNoThrow(try WorkflowBlockEditing.applyDraft(independent,root:rootAddress,expectedBaseRevision:4,idProvider:{"copy"}))
    }
}
