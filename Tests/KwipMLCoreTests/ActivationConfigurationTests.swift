import XCTest
@testable import KwipMLCore

final class ActivationConfigurationTests:XCTestCase {
    private var dataRegistry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    func testCatalogUsesExactlyExistingActivationTypesAndReviewedDefaults()throws {
        XCTAssertEqual(Set(ActivationConfiguration.Function.allCases.map(\.rawValue)),Set(["relu","leaky_relu","prelu","elu","gelu","silu","sigmoid","tanh","softmax","log_softmax"]))
        XCTAssertEqual(ActivationConfiguration.defaultFunction,.relu)
        let registry=try ComponentRegistry.builtIn()
        for function in ActivationConfiguration.Function.allCases {
            let defaults=try ActivationConfiguration.defaults(for:function,registry:registry)
            let config=try ActivationConfiguration(componentID:function.rawValue,configurationVersion:2,values:defaults,registry:registry)
            XCTAssertEqual(config.function,function)
            XCTAssertEqual(config.values,defaults)
        }
        XCTAssertEqual(try ActivationConfiguration.defaults(for:.prelu,registry:registry),["num_parameters":1,"init":.number(0.25)])
        XCTAssertEqual(try ActivationConfiguration.defaults(for:.softmax,registry:registry),["dim":-1])
    }
    func testAtomicSwitchPreservesWiresIdentityPresentationAndSameFunctionSource()throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let address=WorkflowNodeAddress(domain:.model,localID:"classifier")
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==address})
        document.nodes[index].payload = .modelPrimitive(type:"relu",configurationVersion:1,config:[:])
        document.nodes[index].name="My activation";document.nodes[index].x=135;document.nodes[index].y=246
        let original=try WorkflowCodec.encode(document),node=document.nodes[index]
        let same=try WorkflowEditing.apply(.setActivation(address:address,function:.relu),to:document,modelRegistry:.builtIn(),dataRegistry:dataRegistry).document
        XCTAssertEqual(same.nodes[index],node)
        let changed=try WorkflowEditing.apply(.setActivation(address:address,function:.leakyReLU,configuration:["negative_slope":.number(0.2)]),to:document,modelRegistry:.builtIn(),dataRegistry:dataRegistry).document
        XCTAssertEqual(changed.edges,document.edges)
        XCTAssertEqual(changed.nodes[index].address,address);XCTAssertEqual(changed.nodes[index].name,node.name)
        XCTAssertEqual(changed.nodes[index].x,node.x);XCTAssertEqual(changed.nodes[index].y,node.y)
        XCTAssertEqual(changed.nodes[index].payload,.modelPrimitive(type:"leaky_relu",configurationVersion:2,config:["negative_slope":.number(0.2)]))
        XCTAssertEqual(changed.schemaVersion,document.schemaVersion)
        XCTAssertEqual(try WorkflowCodec.encode(document),original)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(changed)),changed)
    }
    func testMalformedSettingsAndNonActivationTargetsRefuseWithoutChangingSource()throws {
        let registry=try ComponentRegistry.builtIn()
        for (type,values) in [("softmax",["dim":JSONValue.integer(0)]),("prelu",["num_parameters":.integer(0),"init":.number(0.25)]),("relu",["inplace":.bool(true)]),("gelu",["approximate_tanh":.string("yes")])] {
            XCTAssertThrowsError(try ActivationConfiguration(componentID:type,configurationVersion:2,values:values,registry:registry))
        }
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),bytes=try WorkflowCodec.encode(document)
        XCTAssertThrowsError(try WorkflowEditing.apply(.setActivation(address:.init(domain:.model,localID:"classifier"),function:.relu),to:document,modelRegistry:registry,dataRegistry:dataRegistry))
        XCTAssertEqual(try WorkflowCodec.encode(document),bytes)
    }
    func testPReLUParameterOwnershipSurvivesSameFunctionAndClearsOnlyOnExplicitSwap()throws {
        let registry=try ComponentRegistry.builtIn()
        let settings:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        let original=WorkflowNode(address:.init(domain:.model,localID:"activation"),name:"Learned slope",x:17,y:29,payload:.modelPrimitive(type:"prelu",configurationVersion:2,config:["num_parameters":1,"init":.number(0.25)]),parameterSettings:settings)
        let same=try ActivationConfiguration.switching(original,to:.prelu,configuration:["num_parameters":2,"init":.number(0.1)],registry:registry)
        XCTAssertEqual(same.parameterSettings,settings)
        let swapped=try ActivationConfiguration.switching(original,to:.relu,registry:registry)
        XCTAssertNil(swapped.parameterSettings)
        XCTAssertEqual(swapped.address,original.address);XCTAssertEqual(swapped.name,original.name)
        XCTAssertEqual(original.parameterSettings,settings)
        XCTAssertThrowsError(try ActivationConfiguration.switching(original,to:.prelu,configuration:["num_parameters":0,"init":.number(0.25)],registry:registry))
    }
    func testClassicCommandPromotesLegacyGraphWithoutChangingLinearBiasSemantics()throws {
        var legacy=ExampleFlows.parallelClassifier;legacy.schemaVersion=1
        for index in legacy.nodes.indices where legacy.nodes[index].type=="linear" {legacy.nodes[index].config.removeValue(forKey:"bias")}
        let state=GraphEditorState(document:legacy,selection:.init(nodeIDs:["activation-a"]))
        let result=try GraphCommandEngine.apply(.setActivation(nodeID:"activation-a",function:.gelu,configuration:["approximate_tanh":true]),to:state,registry:.builtIn())
        XCTAssertEqual(result.before,state)
        XCTAssertEqual(result.after.document.schemaVersion,2)
        XCTAssertEqual(result.after.document.edges,legacy.edges)
        XCTAssertEqual(result.after.selection,state.selection)
        XCTAssertTrue(result.after.document.nodes.filter {$0.type=="linear"}.allSatisfy {$0.config["bias"] == .bool(true)})
        let activation=try XCTUnwrap(result.after.document.nodes.first {$0.id=="activation-a"})
        XCTAssertEqual(activation.type,"gelu");XCTAssertEqual(activation.config,["approximate_tanh":true])
        XCTAssertFalse(GraphValidator.validate(result.after.document,registry:try .builtIn()).issues.contains {$0.nodeID=="activation-a"})
        XCTAssertTrue(legacy.nodes.filter {$0.type=="linear"}.allSatisfy {$0.config["bias"] == nil})
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setActivation(nodeID:"branch-a",function:.relu),to:state,registry:.builtIn()))
    }
    func testInlineActivationSwapUsesCopyOnWriteAndPreservesContainerBindings()throws {
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),bytes=try WorkflowCodec.encode(document)
        let root=WorkflowNodeAddress(domain:.model,localID:"features")
        let swapped=try WorkflowInlineContainer.apply(.setActivation(childID:"activation-a",function:.prelu),to:document,root:root,registry:.builtIn(),idProvider:{"activation-private"})
        let child=try WorkflowInlineContainer.node(in:swapped,root:root,childID:"activation-a",registry:.builtIn())
        guard case .modelPrimitive(let type,_,let values)=child.payload else {return XCTFail("Missing activation")}
        XCTAssertEqual(type,"prelu");XCTAssertEqual(values,["num_parameters":1,"init":.number(0.25)])
        XCTAssertEqual(Array(swapped.definitions.prefix(document.definitions.count)),document.definitions)
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:swapped,root:root).parameters,document.definitions[0].parameters)
        XCTAssertEqual(swapped.edges,document.edges)
        XCTAssertEqual(try WorkflowCodec.encode(document),bytes)
    }
    func testExplicitValidReplacementRepairsMalformedActivationButNoopCannotBlessIt()throws {
        let registry=try ComponentRegistry.builtIn()
        let malformed=WorkflowNode(address:.init(domain:.model,localID:"activation"),payload:.modelPrimitive(type:"softmax",configurationVersion:2,config:["dim":0]))
        XCTAssertThrowsError(try ActivationConfiguration.switching(malformed,to:.softmax,registry:registry))
        let repaired=try ActivationConfiguration.switching(malformed,to:.softmax,configuration:["dim":-1],registry:registry)
        XCTAssertEqual(repaired.payload,.modelPrimitive(type:"softmax",configurationVersion:2,config:["dim":-1]))
        XCTAssertEqual(try ActivationConfiguration.switching(malformed,to:.relu,registry:registry).payload,.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))
        let state=GraphEditorState(document:.init(nodes:[GraphNode(id:"activation",type:"softmax",config:["dim":0])]))
        let classic=try GraphCommandEngine.apply(.setActivation(nodeID:"activation",function:.softmax,configuration:["dim":-1]),to:state,registry:registry)
        XCTAssertEqual(classic.after.document.nodes[0].config,["dim":-1])
        XCTAssertEqual(malformed.payload,.modelPrimitive(type:"softmax",configurationVersion:2,config:["dim":0]))
        XCTAssertThrowsError(try ActivationConfiguration.switching(malformed,to:.prelu,configuration:["num_parameters":0,"init":.number(0.25)],registry:registry))
    }
    func testModelScopeSwitchRetainsNewSettingsAndMigratedLegacySiblingConfiguration()throws {
        var scope=CompositeExamples.parallelClassifier.definitions[0].body
        scope.configurationVersion=1
        for index in scope.nodes.indices where scope.nodes[index].type=="linear" {scope.nodes[index].config.removeValue(forKey:"bias")}
        let original=scope
        let edited=try ModelScopeEditing.apply(.setActivation(nodeID:"activation-a",function:.gelu,configuration:["approximate_tanh":true]),to:scope,definitions:CompositeExamples.parallelClassifier.definitions,registry:.builtIn())
        XCTAssertEqual(edited.scope.configurationVersion,2)
        let changed=try XCTUnwrap(edited.scope.nodes.first {$0.id=="activation-a"})
        XCTAssertEqual(changed.type,"gelu");XCTAssertEqual(changed.config,["approximate_tanh":true])
        XCTAssertEqual(edited.scope.edges,original.edges)
        XCTAssertTrue(edited.scope.nodes.filter {$0.type=="linear"}.allSatisfy {$0.config["bias"] == .bool(true)})
        XCTAssertEqual(scope,original)
        XCTAssertEqual(edited.command.before.document.schemaVersion,1)
        XCTAssertEqual(edited.command.after.document.schemaVersion,2)
    }
}
