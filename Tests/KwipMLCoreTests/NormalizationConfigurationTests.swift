import XCTest
@testable import KwipMLCore

final class NormalizationConfigurationTests:XCTestCase {
    private var dataRegistry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    func testSupportedVariantsAndReviewedDefaults()throws {
        XCTAssertEqual(Set(NormalizationConfiguration.Variant.allCases.map(\.rawValue)),Set(["layer_norm","batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm","rms_norm"]))
        XCTAssertEqual(NormalizationConfiguration.defaultVariant,.layerNorm)
        XCTAssertEqual(NormalizationConfiguration.Variant.batchNorm2d.method,.batch)
        XCTAssertEqual(NormalizationConfiguration.Variant.instanceNorm1d.dimensions,1)
        XCTAssertNil(NormalizationConfiguration.Variant.layerNorm.dimensions)
        XCTAssertNil(NormalizationConfiguration.Variant.groupNorm.dimensions)
        let registry=try ComponentRegistry.builtIn()
        for variant in NormalizationConfiguration.Variant.allCases {
            let defaults=try NormalizationConfiguration.defaults(for:variant,registry:registry)
            XCTAssertEqual(try NormalizationConfiguration(componentID:variant.rawValue,configurationVersion:2,values:defaults,registry:registry).values,defaults)
        }
    }
    func testCanonicalWorkflowRetainsNamedPoliciesAndRefusesRemovingExplicitParameters()throws {
        let registry=try ComponentRegistry.builtIn(),address=WorkflowNodeAddress(domain:.model,localID:"normalization")
        let config=try NormalizationConfiguration.defaults(for:.layerNorm,registry:registry)
        let settings:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")]),"bias":.object(["initialization":.object(["algorithm":.string("zeros"),"settings":.object([:])])])]
        let node=WorkflowNode(address:address,name:"My normalization",x:13,y:27,payload:.modelPrimitive(type:"layer_norm",configurationVersion:2,config:config),parameterSettings:settings)
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[node,.init(address:.init(domain:.model,localID:"consumer"),payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))],edges:[.init(from:.init(address:address,port:"out"),to:.init(address:.init(domain:.model,localID:"consumer"),port:"in"),kind:.tensor)]);document.schemaVersion=15
        let bytes=try WorkflowCodec.encode(document)
        let changed=try WorkflowEditing.apply(.setNormalization(address:address,variant:.groupNorm),to:document,modelRegistry:registry,dataRegistry:dataRegistry).document
        XCTAssertEqual(changed.nodes[0].parameterSettings,settings)
        XCTAssertEqual(changed.nodes[0].address,node.address);XCTAssertEqual(changed.nodes[0].name,node.name)
        XCTAssertEqual(changed.nodes[0].x,node.x);XCTAssertEqual(changed.edges,document.edges)
        var noScale=config;noScale["elementwise_affine"]=false
        XCTAssertThrowsError(try WorkflowEditing.apply(.setNormalization(address:address,variant:.layerNorm,configuration:noScale),to:document,modelRegistry:registry,dataRegistry:dataRegistry))
        var noBias=config;noBias["bias"]=false
        XCTAssertThrowsError(try NormalizationConfiguration.switching(node,to:.layerNorm,configuration:noBias,registry:registry))
        XCTAssertThrowsError(try NormalizationConfiguration.switching(node,to:.instanceNorm1d,registry:registry),"Instance defaults must not silently discard affine policies")
        XCTAssertEqual(try WorkflowCodec.encode(document),bytes)
    }
    func testStatisticsTypeChangesDoNotRewriteCapturedSourceOrInventBufferFields()throws {
        let registry=try ComponentRegistry.builtIn(),config=try NormalizationConfiguration.defaults(for:.batchNorm2d,registry:registry)
        let node=GraphNode(id:"normalization",type:"batch_norm2d",name:"Statistics",x:19,y:31,config:config)
        let original=GraphEditorState(document:.init(nodes:[node]))
        var instance=try NormalizationConfiguration.defaults(for:.instanceNorm2d,registry:registry);instance["track_running_stats"]=false
        let changed=try GraphCommandEngine.apply(.setNormalization(nodeID:node.id,variant:.instanceNorm2d,configuration:instance),to:original,registry:registry)
        XCTAssertEqual(changed.before,original)
        XCTAssertEqual(changed.after.document.nodes[0].type,"instance_norm2d")
        XCTAssertEqual(changed.after.document.nodes[0].config["track_running_stats"],false)
        XCTAssertEqual(original.document.nodes[0].config["track_running_stats"],true)
        var sourceState=config;sourceState["running_mean"] = .array([0,0])
        XCTAssertThrowsError(try NormalizationConfiguration(componentID:"batch_norm2d",configurationVersion:2,values:sourceState,registry:registry))
    }
    func testScopedStatisticsOptionsPersistAndInvalidGeometryCannotBeBypassed()throws {
        let registry=try ComponentRegistry.builtIn(),config=try NormalizationConfiguration.defaults(for:.batchNorm1d,registry:registry)
        let scope=CompositeGraph(nodes:[.init(id:"normalization",type:"batch_norm1d",config:config)])
        var next=config;next["eps"] = .number(1e-4);next["averaging"] = .string("cumulative");next["track_running_stats"]=true
        let changed=try ModelScopeEditing.apply(.setNormalization(nodeID:"normalization",variant:.batchNorm1d,configuration:next),to:scope,definitions:[],registry:registry)
        XCTAssertEqual(changed.scope.nodes[0].config["eps"],.number(1e-4))
        XCTAssertEqual(changed.scope.nodes[0].config["averaging"],.string("cumulative"))
        XCTAssertEqual(changed.scope.nodes[0].config["track_running_stats"],true)
        XCTAssertEqual(scope.nodes[0].config,config)
        var grouped=try NormalizationConfiguration.defaults(for:.groupNorm,registry:registry);grouped["num_channels"]=3;grouped["num_groups"]=2
        XCTAssertThrowsError(try NormalizationConfiguration(componentID:"group_norm",configurationVersion:2,values:grouped,registry:registry))
        var bad=config;bad["eps"]=0
        XCTAssertThrowsError(try NormalizationConfiguration(componentID:"batch_norm1d",configurationVersion:2,values:bad,registry:registry))
        bad=config;bad["momentum"] = .number(-0.1)
        XCTAssertThrowsError(try NormalizationConfiguration(componentID:"batch_norm1d",configurationVersion:2,values:bad,registry:registry))
    }
    func testInlineRawBindingAndLayerPolicySurviveSameVariantWhileSwapRefuses()throws {
        let registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        let config=try NormalizationConfiguration.defaults(for:.layerNorm,registry:registry)
        let field=try XCTUnwrap(registry.component(type:"layer_norm")?.fields.first {$0.key=="eps"})
        let policy:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        let definition=CompositeDefinition(reference:.init("normalization-fixture"),title:"Normalization graph",body:.init(nodes:[.init(id:"normalization",type:"layer_norm",config:config,parameterSettings:policy)]),inputs:[.init("in",targets:[.init("normalization","in")])],outputs:[.init("out",source:.init("normalization","out"))],parameters:[.init(field,targets:[.init("normalization","eps")])],composition:.init(kind:.graph,children:[.init(localID:"normalization")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:["eps":.number(1e-4)]))],definitions:[definition]);document.schemaVersion=15
        let proxy=try WorkflowInlineContainer.node(in:document,root:root,childID:"normalization",registry:registry)
        guard case .modelPrimitive(_,_,var effective)=proxy.payload else {return XCTFail("Missing normalization")}
        XCTAssertEqual(effective["eps"],.number(1e-4));effective["bias"]=false
        let changed=try WorkflowInlineContainer.apply(.setNormalization(childID:"normalization",variant:.layerNorm,configuration:effective),to:document,root:root,registry:registry,idProvider:{"normalization-private"})
        let stored=try WorkflowInlineContainer.definition(in:changed,root:root).body.nodes[0]
        XCTAssertEqual(stored.config["eps"],config["eps"])
        XCTAssertEqual(stored.parameterSettings,policy)
        XCTAssertEqual(stored.config["bias"],false)
        XCTAssertEqual(changed.definitions[0],definition)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setNormalization(childID:"normalization",variant:.groupNorm),to:document,root:root,registry:registry,idProvider:{"bound-refusal"}))
    }
    func testSameVariantNoopPreservesRawSourceAndExplicitRepairValidatesTarget()throws {
        let registry=try ComponentRegistry.builtIn()
        let node=WorkflowNode(address:.init(domain:.model,localID:"normalization"),payload:.modelPrimitive(type:"batch_norm1d",configurationVersion:2,config:[:]))
        XCTAssertEqual(try NormalizationConfiguration.switching(node,to:.batchNorm1d,registry:registry),node)
        var malformed=try NormalizationConfiguration.defaults(for:.batchNorm1d,registry:registry);malformed["eps"]=0
        var bad=node;bad.payload = .modelPrimitive(type:"batch_norm1d",configurationVersion:2,config:malformed)
        XCTAssertThrowsError(try NormalizationConfiguration.switching(bad,to:.batchNorm1d,registry:registry))
        let repaired=try NormalizationConfiguration.switching(bad,to:.batchNorm1d,configuration:NormalizationConfiguration.defaults(for:.batchNorm1d,registry:registry),registry:registry)
        XCTAssertEqual(try NormalizationConfiguration.configuration(of:repaired,registry:registry).variant,.batchNorm1d)
        XCTAssertThrowsError(try NormalizationConfiguration.switching(node,to:.batchNorm1d,connectedOutputPorts:["indices"],registry:registry))
    }
}

extension NormalizationConfigurationTests {
    func testRMSContractParameterInventoryShapeAndPersistence()throws {
        let registry=try ComponentRegistry.builtIn(),values:[String:JSONValue]=["normalized_shape":.array([3]),"eps":.number(1e-5),"elementwise_affine":true]
        let c=try RMSNormConfiguration(values:values)
        XCTAssertEqual(c.parameterCount,3)
        XCTAssertEqual(try c.outputLayout(for:.init(sample:[2,3])).sample.dimensions,[2,3])
        XCTAssertThrowsError(try c.outputLayout(for:.init(sample:[3,2])))
        XCTAssertEqual(try LayerParameterInventory.choices(type:"rms_norm",config:values,version:2,registry:registry).map(\.id),["weight"])
        for bad in [["bias":JSONValue.bool(false)],["track_running_stats":.bool(false)],["eps":.number(0)],["elementwise_affine":.integer(1)],["normalized_shape":.array([0])]] {
            XCTAssertThrowsError(try RMSNormConfiguration(values:values.merging(bad){_,new in new}))
        }
        var graph=ExampleFlows.layerNormClassifier();let i=try XCTUnwrap(graph.nodes.firstIndex(where:{$0.type=="layer_norm"}))
        let old=try JSONEncoder().encode(graph)
        graph.nodes[i]=try NormalizationConfiguration.switching(graph.nodes[i],configurationVersion:2,to:.rmsNorm,configuration:values,registry:registry)
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.issues.isEmpty,result.issues.map(\.message).joined(separator:"; "))
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(graph)),graph)
        XCTAssertNotEqual(try JSONEncoder().encode(graph),old)
        let entry=try XCTUnwrap(BundledNormalizationComponents.entries["rms_norm"])
        let descriptor=try JSONEncoder().encode(entry.runtimeContract)
        XCTAssertNoThrow(try RMSNormRuntimeCapability.decode(descriptor))
        guard case .object(var changed)=entry.runtimeContract else {return XCTFail("Missing contract")};changed["subtractsMean"]=true
        XCTAssertThrowsError(try RMSNormRuntimeCapability.decode(JSONEncoder().encode(JSONValue.object(changed))))
    }
    func testRMSRefusesDiscardingBiasOrScalePolicies()throws {
        let registry=try ComponentRegistry.builtIn(),address=WorkflowNodeAddress(domain:.model,localID:"normalization")
        let values=try NormalizationConfiguration.defaults(for:.layerNorm,registry:registry)
        let node=WorkflowNode(address:address,payload:.modelPrimitive(type:"layer_norm",configurationVersion:2,config:values),parameterSettings:["bias":.object(["trainability":.string("frozen")])])
        XCTAssertThrowsError(try NormalizationConfiguration.switching(node,to:.rmsNorm,registry:registry))
        var rms=node;rms.parameterSettings=["weight":.object(["trainability":.string("frozen")])]
        rms=try NormalizationConfiguration.switching(rms,to:.rmsNorm,registry:registry)
        var noScale=try RMSNormConfiguration(values:try NormalizationConfiguration.defaults(for:.rmsNorm,registry:registry)).values;noScale["elementwise_affine"]=false
        XCTAssertThrowsError(try NormalizationConfiguration.switching(rms,to:.rmsNorm,configuration:noScale,registry:registry))
    }
}
