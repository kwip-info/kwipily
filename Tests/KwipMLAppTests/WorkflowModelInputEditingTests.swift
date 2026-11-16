import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowModelInputEditingTests:XCTestCase {
    private func fixture()throws->(ProjectLibrary,WorkflowAuthoringStore,WorkflowCanvasAdapter,WorkflowControlNodeContext) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let document=try WorkflowExamples.preparedParallelClassifier()
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:data)
        let model=try XCTUnwrap(document.nodes.first {if case .modelDefinition=$0.payload{return true};return false})
        XCTAssertTrue(owner.select(.init(nodeAddresses:[model.address]),lease:owner.lease()))
        return (library,owner,try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:data),try WorkflowControlNodeContext(owner:owner,address:model.address))
    }
    func testModelConfigurationEditsOriginalBoundaryPersistsAndUndoRedoRetainsIdentity()throws {
        let(library,owner,canvas,context)=try fixture()
        let boundary=try XCTUnwrap(context.modelInputBoundary),original=owner.document
        XCTAssertTrue(context.isModelDefinition);XCTAssertEqual(context.modelInputConfiguration,["features":2])
        XCTAssertEqual(canvas.creationEntries.filter {WorkflowModelInput.componentIDs.contains($0.iconType)}.map(\.component.title),["Model input"])
        XCTAssertTrue(canvas.creationEntries.contains {$0.iconType=="bag_input"})
        XCTAssertNotNil(canvas.projection.displayIDByAddress[boundary.address])
        XCTAssertTrue(context.setModelInput(configuration:["features":3]))
        XCTAssertTrue(context.isCurrent)
        XCTAssertEqual(context.modelInputBoundary?.address,boundary.address)
        XCTAssertEqual(context.modelInputConfiguration,["features":3])
        XCTAssertEqual(owner.document.edges,original.edges)
        let restored=try WorkflowAuthoringStore(library:library,projectID:owner.projectID,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:context.authority.address,in:restored.document)?.payload,context.modelInputBoundary?.payload)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,original.nodes)
        XCTAssertFalse(context.isCurrent);XCTAssertFalse(context.setModelInput(configuration:["features":4]))
        XCTAssertTrue(owner.redo(lease:owner.lease()))
        XCTAssertEqual(try WorkflowModelInput.boundary(for:context.authority.address,in:owner.document)?.address,boundary.address)
    }
    func testEntryTargetsAndIntegerShapeUseCanonicalEditsAndRefuseOccupiedPorts()throws {
        let(_,owner,_,context)=try fixture()
        let targets=context.modelInputDestinations
        XCTAssertFalse(targets.isEmpty)
        XCTAssertTrue(context.modelInputTargets.contains {$0.available && targets.contains($0.endpoint)})
        XCTAssertTrue(context.modelInputTargets.allSatisfy(\.available))
        let occupied=try XCTUnwrap(owner.document.edges.first {$0.from.address.domain == .model && $0.to.address.domain == .model && $0.from.address != context.modelInputBoundary?.address}?.to)
        let unchanged=owner.document
        XCTAssertFalse(context.setModelInput(destinations:targets+[occupied]));XCTAssertEqual(owner.document,unchanged)
        let before=owner.document
        XCTAssertFalse(context.setModelInput(configuration:["features":0]));XCTAssertEqual(owner.document,before)
        XCTAssertTrue(context.setModelInput(destinations:[]));XCTAssertEqual(context.modelInputDestinations,[])
        XCTAssertNotNil(context.modelInputBoundary)
        XCTAssertTrue(context.setModelInput(destinations:targets))
        XCTAssertTrue(context.setModelInput(type:"index_input"))
        XCTAssertEqual(context.modelInputType,"index_input")
        XCTAssertEqual(try IndexInputConfiguration(values:context.modelInputConfiguration).tensor.dtype,.int64)
        // No automatic float conversion is inserted to make an incompatible model run.
        XCTAssertThrowsError(try WorkflowCompiler.modelSnapshot(owner.document,definition:context.authority.address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry))
        XCTAssertTrue(context.setModelInput(type:"input",configuration:["features":2]))
        XCTAssertNoThrow(try WorkflowCompiler.modelSnapshot(owner.document,definition:context.authority.address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry))
    }
    func testNamedTokenBoundaryIsSelectedWhenOpeningItsModelDefinition()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),document=try TextGenerationPracticeExample.document(device:"cpu")
        _=try library.saveSource(.workflow(document))
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let definition=try XCTUnwrap(document.nodes.first(where:WorkflowModelBoundary.isDefinition))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[definition.address]),lease:owner.lease()))
        let context=try WorkflowControlNodeContext(owner:owner,address:definition.address)
        XCTAssertEqual(context.selectedModelInputName,"tokens")
        XCTAssertEqual(context.modelInputType,"index_input")
        XCTAssertEqual(context.modelInputConfiguration,["shape":.array([3])])
        XCTAssertNotNil(context.modelInputBoundary)
    }
    func testSelectionDepartureRetiresModelInputCallbacks()throws {
        let(_,owner,_,context)=try fixture()
        let before=owner.document
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setModelInput(configuration:["features":5]));XCTAssertEqual(owner.document,before)
    }
}

extension WorkflowModelInputEditingTests {
    func testNamedInputOwnershipPersistsDuplicatesRemovesAndUndoes()throws {
        let(library,owner,_,context)=try fixture()
        XCTAssertTrue(try owner.migrateRetiredToolsForAuthoring())
        let definition=try XCTUnwrap(owner.document.nodes.first(where:WorkflowModelBoundary.isDefinition))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[definition.address]),lease:owner.lease()))
        let current=try WorkflowControlNodeContext(owner:owner,address:definition.address)
        let primary=try XCTUnwrap(current.modelInputBoundary)
        XCTAssertTrue(current.authority.apply(.setModelInput(definition:current.authority.address,type:"input",configuration:["features":3],destinations:[],inputName:"metadata")))
        XCTAssertEqual(owner.document.schemaVersion,19)
        let names=try XCTUnwrap(owner.document.namedModelInputs?[current.authority.address.localID])
        XCTAssertEqual(names["input"],primary.address.localID)
        XCTAssertEqual(names.count,2)
        let reloaded=try WorkflowAuthoringStore(library:library,projectID:owner.projectID,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        XCTAssertEqual(reloaded.document.namedModelInputs,owner.document.namedModelInputs)
        let before=owner.document
        XCTAssertFalse(current.authority.apply(.setModelInput(definition:current.authority.address,type:"input",configuration:["features":3],destinations:[],inputName:"not valid")))
        XCTAssertEqual(owner.document,before)
        XCTAssertTrue(current.authority.apply(.removeModelInput(definition:current.authority.address,inputName:"metadata")))
        XCTAssertEqual(owner.document.namedModelInputs?[current.authority.address.localID]?.count,1)
        XCTAssertFalse(owner.document.nodes.contains {$0.address.localID==names["metadata"]})
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,before.nodes)
        XCTAssertEqual(owner.document.edges,before.edges)
        XCTAssertEqual(owner.document.namedModelInputs,before.namedModelInputs)
        let refreshed=try WorkflowControlNodeContext(owner:owner,address:current.authority.address)
        XCTAssertTrue(refreshed.authority.apply(.duplicate(addresses:[current.authority.address],offset:.init(x:20,y:20))))
        let copy=try XCTUnwrap(owner.selection.nodeAddresses.first {address in owner.document.nodes.first(where:{$0.address==address}).map(WorkflowModelBoundary.isDefinition)==true})
        let copied=try XCTUnwrap(owner.document.namedModelInputs?[copy.localID])
        XCTAssertEqual(Set(copied.keys),Set(names.keys))
        XCTAssertNotEqual(copied["metadata"],names["metadata"])
        XCTAssertNotNil(owner.apply(.delete(.init(nodeAddresses:[copy])),lease:owner.lease()))
        XCTAssertNil(owner.document.namedModelInputs?[copy.localID])
        XCTAssertNoThrow(try WorkflowCodec.validate(owner.document))
    }
}
