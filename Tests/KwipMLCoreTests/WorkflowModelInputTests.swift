import XCTest
@testable import KwipMLCore

final class WorkflowModelInputTests:XCTestCase {
    let model=WorkflowNodeAddress(domain:.control,localID:"model")
    let input=WorkflowNodeAddress(domain:.model,localID:"input")
    let layer=WorkflowNodeAddress(domain:.model,localID:"linear")
    private func fixture() -> WorkflowDocument {
        WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:input,name:"Saved boundary",x:17,y:23,payload:.modelPrimitive(type:"input",configurationVersion:2,config:["features":2])),
            .init(address:layer,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true])),
            .init(address:.init(domain:.model,localID:"output"),payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:])),
            .init(address:model,payload:.modelDefinition)],edges:[
                .init(id:"entry",from:.init(address:input,port:"out"),to:.init(address:layer,port:"in"),kind:.tensor),
                .init(id:"prediction",from:.init(address:layer,port:"out"),to:.init(address:.init(domain:.model,localID:"output"),port:"in"),kind:.tensor),
                .init(id:"definition",from:.init(address:.init(domain:.model,localID:"output"),port:"definition"),to:.init(address:model,port:"source"),kind:.modelDefinition)])
    }
    private func registry() throws -> DataFlowOperatorRegistry {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try .load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
    }
    func testLegacyInputIsVisibleWithoutChangingSavedOrCompiledGraph() throws {
        let source=fixture(),data=try registry()
        let encoded=try WorkflowCodec.encode(source)
        let before=try WorkflowCompiler.modelSnapshot(source,definition:model,modelRegistry:.builtIn(),dataRegistry:data)
        let canvas=try WorkflowCanvasProjector.project(source,modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertNotNil(canvas.displayIDByAddress[input],"The real tensor boundary must mark the model starting point.")
        XCTAssertNotNil(canvas.displayIDByEdgeID["entry"],"Its actual tensor connection must remain visible.")
        XCTAssertEqual(try WorkflowCodec.encode(source),encoded)
        XCTAssertEqual(try WorkflowCompiler.modelSnapshot(source,definition:model,modelRegistry:.builtIn(),dataRegistry:data).graph,before.graph)
    }
    private func apply(_ command:WorkflowCommand,to source:WorkflowDocument)throws->WorkflowDocument {
        try WorkflowEditing.apply(command,to:source,selection:.init(nodeAddresses:[model]),modelRegistry:.builtIn(),dataRegistry:registry()).document
    }
    func testConfigurationOwnershipRoundTripsAndPreservesExactModelIdentity()throws {
        let source=fixture(),data=try registry()
        let before=try WorkflowCompiler.modelSnapshot(source,definition:model,modelRegistry:.builtIn(),dataRegistry:data)
        let next=try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[.init(address:layer,port:"in")]),to:source)
        XCTAssertEqual(next.schemaVersion,14);XCTAssertEqual(next.modelInputs,["model":"input"])
        XCTAssertEqual(next.nodes,source.nodes);XCTAssertEqual(next.edges,source.edges)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(next)),next)
        XCTAssertEqual(try WorkflowCompiler.modelSnapshot(next,definition:model,modelRegistry:.builtIn(),dataRegistry:data).graph,before.graph)
        let empty=try apply(.setModelInput(definition:model,type:"input",configuration:["features":5],destinations:[]),to:next)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:model,in:empty)?.address,input)
        XCTAssertEqual(WorkflowModelInput.destinations(of:try XCTUnwrap(WorkflowModelInput.boundary(for:model,in:empty)),in:empty),[])
        XCTAssertThrowsError(try WorkflowCompiler.modelSnapshot(empty,definition:model,modelRegistry:.builtIn(),dataRegistry:data))
    }
    func testNewModelCreatesInputWithoutPlacingAToolAndRejectsInvalidTargets()throws {
        var source=fixture();source.nodes.removeAll {$0.address==input};source.edges.removeAll {$0.from.address==input}
        let next=try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[.init(address:layer,port:"in")]),to:source)
        let boundary=try XCTUnwrap(WorkflowModelInput.boundary(for:model,in:next))
        XCTAssertNotEqual(boundary.address,input)
        XCTAssertNoThrow(try WorkflowCompiler.modelSnapshot(next,definition:model,modelRegistry:.builtIn(),dataRegistry:registry()))
        XCTAssertThrowsError(try apply(.setModelInput(definition:model,type:"input",configuration:["features":0],destinations:[]),to:next))
        XCTAssertThrowsError(try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[.init(address:model,port:"source")]),to:next))
        XCTAssertThrowsError(try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[.init(address:layer,port:"missing")]),to:next))
        XCTAssertEqual(source.modelInputs,nil)
    }
    func testIntegerInputRetainsInt64AndBagInputRetainsNamedCanvasPorts()throws {
        var integer=fixture();integer.nodes[0].payload = .modelPrimitive(type:"index_input",configurationVersion:2,config:["shape":.array([2])])
        let boundary=try XCTUnwrap(WorkflowModelInput.boundary(for:model,in:integer))
        XCTAssertTrue(WorkflowModelInput.isBoundary(boundary))
        XCTAssertNotNil(try WorkflowCanvasProjector.project(integer,modelRegistry:.builtIn(),dataRegistry:registry()).displayIDByAddress[input])
        XCTAssertEqual(try IndexInputConfiguration(values:["shape":.array([2])]).tensor.dtype,.int64)
        let changed=try apply(.setModelInput(definition:model,type:"index_input",configuration:["shape":.array([3])],destinations:[.init(address:layer,port:"in")]),to:integer)
        XCTAssertEqual(changed.nodes[0].payload,.modelPrimitive(type:"index_input",configurationVersion:2,config:["shape":.array([3])]))
        var bag=fixture();bag.nodes[0].payload = .modelPrimitive(type:"bag_input",configurationVersion:2,config:BagSourceConfiguration.defaults)
        XCTAssertNotNil(try WorkflowCanvasProjector.project(bag,modelRegistry:.builtIn(),dataRegistry:registry()).displayIDByAddress[input])
        XCTAssertThrowsError(try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[]),to:bag))
    }
    func testVisibleEntryRequiresExplicitDisconnectBeforeReplacement()throws {
        var source=fixture()
        let upstream=WorkflowNodeAddress(domain:.model,localID:"upstream")
        source.nodes.append(.init(address:upstream,payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:])))
        XCTAssertThrowsError(try apply(.connect(from:.init(address:upstream,port:"out"),to:.init(address:layer,port:"in")),to:source))
        let disconnected=try apply(.delete(.init(edgeIDs:["entry"])),to:source)
        let next=try apply(.connect(from:.init(address:upstream,port:"out"),to:.init(address:layer,port:"in")),to:disconnected)
        XCTAssertEqual(next.edges.first {$0.to == .init(address:layer,port:"in")}?.from.address,upstream)
        XCTAssertTrue(next.nodes.contains {$0.address==input},"Disconnecting does not silently delete the boundary")
    }
    func testHiddenProjectionPresentationReuseDoesNotIndexCanonicalNodes()throws {
        let source=fixture(),data=try registry()
        let before=try WorkflowCanvasProjector.project(source,modelRegistry:.builtIn(),dataRegistry:data)
        var moved=source;moved.nodes[1].x=99;moved.nodes[1].name="Moved";moved.revision+=1
        let cached=try XCTUnwrap(before.updatingPresentation(moved))
        let fresh=try WorkflowCanvasProjector.project(moved,modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(cached.graph,fresh.graph)
    }
    func testOwnershipCannotPointAtMissingWrongDomainOrBagAndOldSchemaRefusesIt()throws {
        var source=fixture();source.modelInputs=["model":"input"]
        XCTAssertThrowsError(try WorkflowCodec.encode(source))
        source.schemaVersion=14;XCTAssertNoThrow(try WorkflowCodec.encode(source))
        source.modelInputs=["model":"linear"];XCTAssertThrowsError(try WorkflowCodec.encode(source))
        source.modelInputs=["missing":"input"];XCTAssertThrowsError(try WorkflowCodec.encode(source))
    }

    func testIntegerExamplePreservesCompiledGraphAcrossOwnershipEdit()throws {
        let source=try WorkflowExamples.preparedEmbeddingClassifier(),data=try registry()
        let definition=WorkflowNodeAddress(domain:.control,localID:"model")
        let boundary=try XCTUnwrap(WorkflowModelInput.boundary(for:definition,in:source))
        guard case .modelPrimitive("index_input",_,let config)=boundary.payload else {return XCTFail("Expected exact integer boundary")}
        let before=try WorkflowCompiler.modelSnapshot(source,definition:definition,modelRegistry:.builtIn(),dataRegistry:data)
        let next=try apply(.setModelInput(definition:definition,type:"index_input",configuration:config,destinations:WorkflowModelInput.destinations(of:boundary,in:source)),to:source)
        XCTAssertEqual(next.nodes,source.nodes);XCTAssertEqual(next.edges,source.edges)
        XCTAssertEqual(try WorkflowCompiler.modelSnapshot(next,definition:definition,modelRegistry:.builtIn(),dataRegistry:data).graph,before.graph)
    }
    func testSharedFanoutRetainsAllDestinationsAndWholeModelCopyOwnsItsBoundary()throws {
        var source=fixture()
        let second=WorkflowNodeAddress(domain:.control,localID:"second")
        source.nodes.append(.init(address:second,payload:.modelDefinition))
        source.edges.append(.init(id:"shared-definition",from:.init(address:.init(domain:.model,localID:"output"),port:"definition"),to:.init(address:second,port:"source"),kind:.modelDefinition))
        let configured=try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[.init(address:layer,port:"in")]),to:source)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:second,in:configured)?.address,input)
        var count=0
        let selected=Set(configured.nodes.filter {$0.address != input && $0.address != second}.map(\.address))
        let copied=try WorkflowEditing.apply(.duplicate(addresses:selected,offset:.init(x:20,y:30)),to:configured,modelRegistry:.builtIn(),dataRegistry:registry(),makeID:{count+=1;return "copy-\(count)"})
        let newDefinition=try XCTUnwrap(copied.copies.nodes[model]),newBoundary=try XCTUnwrap(copied.copies.nodes[input])
        XCTAssertEqual(copied.document.modelInputs?[newDefinition.localID],newBoundary.localID)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:newDefinition,in:copied.document)?.address,newBoundary)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:second,in:copied.document)?.address,input)
        XCTAssertNoThrow(try WorkflowCompiler.modelSnapshot(copied.document,definition:newDefinition,modelRegistry:.builtIn(),dataRegistry:registry()))
        let deleted=try apply(.delete(.init(nodeAddresses:[newDefinition])),to:copied.document)
        XCTAssertNil(deleted.modelInputs?[newDefinition.localID]);XCTAssertEqual(deleted.modelInputs?[model.localID],input.localID)
    }
    func testAmbiguousLegacyInputsStayVisibleAndCannotBeSilentlyReassigned()throws {
        var source=fixture()
        let another=WorkflowNodeAddress(domain:.model,localID:"other-input")
        source.nodes.append(.init(address:another,payload:.modelPrimitive(type:"index_input",configurationVersion:2,config:["shape":.array([2])])))
        source.edges.append(.init(id:"ambiguous",from:.init(address:another,port:"out"),to:.init(address:layer,port:"unavailable"),kind:.tensor))
        XCTAssertThrowsError(try WorkflowModelInput.boundary(for:model,in:source))
        let projected=try WorkflowCanvasProjector.project(source,modelRegistry:.builtIn(),dataRegistry:registry())
        XCTAssertNotNil(projected.displayIDByAddress[input]);XCTAssertNotNil(projected.displayIDByAddress[another])
        XCTAssertThrowsError(try apply(.setModelInput(definition:model,type:"input",configuration:["features":2],destinations:[]),to:source))
    }

    func testCopyOneOfTwoModelsSharingAnInputGetsAnIndependentConnectedBoundary()throws {
        var source=fixture()
        let otherLayer=WorkflowNodeAddress(domain:.model,localID:"other-layer"),otherOutput=WorkflowNodeAddress(domain:.model,localID:"other-output"),otherModel=WorkflowNodeAddress(domain:.control,localID:"other-model")
        source.nodes += [.init(address:otherLayer,payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:])),.init(address:otherOutput,payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:])),.init(address:otherModel,payload:.modelDefinition)]
        source.edges += [.init(id:"other-entry",from:.init(address:input,port:"out"),to:.init(address:otherLayer,port:"in"),kind:.tensor),.init(id:"other-prediction",from:.init(address:otherLayer,port:"out"),to:.init(address:otherOutput,port:"in"),kind:.tensor),.init(id:"other-definition",from:.init(address:otherOutput,port:"definition"),to:.init(address:otherModel,port:"source"),kind:.modelDefinition)]
        var count=0
        let copied=try WorkflowEditing.apply(.duplicate(addresses:[layer,.init(domain:.model,localID:"output"),model],offset:.init(x:0,y:30)),to:source,modelRegistry:.builtIn(),dataRegistry:registry(),makeID:{count+=1;return "branch-copy-\(count)"})
        let definition=try XCTUnwrap(copied.copies.nodes[model]),boundary=try XCTUnwrap(copied.copies.nodes[input])
        XCTAssertNotEqual(boundary,input)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:definition,in:copied.document)?.address,boundary)
        XCTAssertNoThrow(try WorkflowCompiler.modelSnapshot(copied.document,definition:definition,modelRegistry:.builtIn(),dataRegistry:registry()))
        XCTAssertEqual(copied.document.edges.first {$0.id=="other-entry"},source.edges.first {$0.id=="other-entry"})
    }

    func testEditedModelStillPlansExperimentVariantsWithOwnedInput()throws {
        let source=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),data=try registry()
        let definition=WorkflowNodeAddress(domain:.control,localID:"model")
        let boundary=try XCTUnwrap(WorkflowModelInput.boundary(for:definition,in:source))
        let configured=try apply(.setModelInput(definition:definition,type:"input",configuration:["features":2],destinations:WorkflowModelInput.destinations(of:boundary,in:source)),to:source)
        let plan=try WorkflowExperimentPlan.build(configured,train:WorkflowExamples.trainAddress,axes:[.init(field:"epochs",values:[1,2])],budget:.init(maximumRuns:2,maximumTotalEpochs:3),modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(plan.variants.count,2)
        XCTAssertTrue(plan.variants.allSatisfy {$0.document.modelInputs==configured.modelInputs})
        XCTAssertTrue(plan.variants.allSatisfy {$0.execution.model.graph.nodes.contains {$0.id==boundary.address.localID}})
    }

    func testLegacyReusableModelKeepsExactCompositionAndSemanticIdentityWhenInputMoves()throws {
        let source=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),data=try registry()
        let definition=WorkflowNodeAddress(domain:.control,localID:"model")
        let boundary=try XCTUnwrap(WorkflowModelInput.boundary(for:definition,in:source))
        let before=try WorkflowCompiler.modelSnapshot(source,definition:definition,modelRegistry:.builtIn(),dataRegistry:data)
        let configured=try apply(.setModelInput(definition:definition,type:"input",configuration:["features":2],destinations:WorkflowModelInput.destinations(of:boundary,in:source)),to:source)
        let after=try WorkflowCompiler.modelSnapshot(configured,definition:definition,modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(after.graph,before.graph)
        XCTAssertEqual(after.semanticSHA256,before.semanticSHA256)
        XCTAssertEqual(after.provenance,before.provenance)
    }

    func testUnrelatedNativeContainerCannotPromoteRetainedLegacyModel()throws {
        let source=try WorkflowExamples.preparedParallelClassifier(),data=try registry()
        let definition=WorkflowNodeAddress(domain:.control,localID:"model")
        let boundary=try XCTUnwrap(WorkflowModelInput.boundary(for:definition,in:source))
        var configured=try apply(.setModelInput(definition:definition,type:"input",configuration:["features":2],destinations:WorkflowModelInput.destinations(of:boundary,in:source)),to:source)
        var unrelated=configured.definitions[0]
        unrelated.reference = .init("unrelated-native",version:unrelated.reference.version)
        unrelated.composition = .init(kind:.graph,children:unrelated.body.nodes.map {.init(localID:$0.id)})
        configured.definitions.append(unrelated)
        let snapshot=try WorkflowCompiler.modelSnapshot(configured,definition:definition,modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(snapshot.provenance?.nativeExpansionContractVersion,1)
        XCTAssertEqual(snapshot.provenance?.sourceCompositeDocument.schemaVersion,2)
    }

}
