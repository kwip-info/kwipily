import XCTest
@testable import KwipMLCore

final class ConvolutionConfigurationTests:XCTestCase {
    private var dataRegistry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    func testSevenVariantsUseReviewedExistingContractsAndDefault2D()throws {
        XCTAssertEqual(Set(ConvolutionConfiguration.Variant.allCases.map(\.rawValue)),Set(["conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d","depthwise_conv2d"]))
        XCTAssertEqual(ConvolutionConfiguration.defaultVariant,.conv2d)
        XCTAssertEqual(ConvolutionConfiguration.Variant.convTranspose3d.operation,.transposed)
        XCTAssertEqual(ConvolutionConfiguration.Variant.depthwiseConv2d.dimensions,2)
        let registry=try ComponentRegistry.builtIn()
        for variant in ConvolutionConfiguration.Variant.allCases {
            let defaults=try ConvolutionConfiguration.defaults(for:variant,registry:registry)
            XCTAssertEqual(try ConvolutionConfiguration(componentID:variant.rawValue,configurationVersion:2,values:defaults,registry:registry).values,defaults)
        }
    }
    func testAtomicWorkflowSwitchRetainsIdentityWiresAndExplicitParameterPolicies()throws {
        let registry=try ComponentRegistry.builtIn(),address=WorkflowNodeAddress(domain:.model,localID:"convolution")
        let config=try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry)
        let policies:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")]),"bias":.object(["initialization":.object(["algorithm":.string("zeros"),"settings":.object([:])])])]
        let node=WorkflowNode(address:address,name:"My kernel",x:31,y:47,payload:.modelPrimitive(type:"conv2d",configurationVersion:2,config:config),parameterSettings:policies)
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[node,.init(address:.init(domain:.model,localID:"next"),payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))],edges:[.init(from:.init(address:address,port:"out"),to:.init(address:.init(domain:.model,localID:"next"),port:"in"),kind:.tensor)]);document.schemaVersion=15
        let bytes=try WorkflowCodec.encode(document)
        let changed=try WorkflowEditing.apply(.setConvolution(address:address,variant:.convTranspose2d),to:document,modelRegistry:registry,dataRegistry:dataRegistry).document
        XCTAssertEqual(changed.nodes[0].address,node.address);XCTAssertEqual(changed.nodes[0].name,node.name)
        XCTAssertEqual(changed.nodes[0].x,node.x);XCTAssertEqual(changed.nodes[0].y,node.y)
        XCTAssertEqual(changed.nodes[0].parameterSettings,policies)
        XCTAssertEqual(changed.edges,document.edges)
        XCTAssertEqual(try WorkflowCodec.encode(document),bytes)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(changed)),changed)
    }
    func testRemovingBiasCannotSilentlyDiscardItsAuthoredPolicy()throws {
        let registry=try ComponentRegistry.builtIn()
        let config=try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry)
        let node=WorkflowNode(address:.init(domain:.model,localID:"conv"),payload:.modelPrimitive(type:"conv2d",configurationVersion:2,config:config),parameterSettings:["bias":.object(["trainability":.string("frozen")])])
        var noBias=config;noBias["bias"]=false
        XCTAssertThrowsError(try ConvolutionConfiguration.switching(node,to:.conv2d,configuration:noBias,registry:registry))
        var transposed=try ConvolutionConfiguration.defaults(for:.convTranspose2d,registry:registry);transposed["bias"]=false
        XCTAssertThrowsError(try ConvolutionConfiguration.switching(node,to:.convTranspose2d,configuration:transposed,registry:registry))
        XCTAssertEqual(try ConvolutionConfiguration.switching(node,to:.conv2d,registry:registry),node)
        var cleared=node;cleared.parameterSettings=nil
        XCTAssertNil(try ConvolutionConfiguration.switching(cleared,to:.conv2d,configuration:noBias,registry:registry).parameterSettings)
    }
    func testScopedSwitchPersistsExactTargetGeometryAndRefusesUnsupportedPorts()throws {
        let registry=try ComponentRegistry.builtIn(),config=try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry)
        let scope=CompositeGraph(nodes:[GraphNode(id:"conv",type:"conv2d",config:config)])
        var transposed=try ConvolutionConfiguration.defaults(for:.convTranspose2d,registry:registry)
        transposed["stride"] = .array([2,2]);transposed["output_padding"] = .array([1,1])
        let result=try ModelScopeEditing.apply(.setConvolution(nodeID:"conv",variant:.convTranspose2d,configuration:transposed),to:scope,definitions:[],registry:registry)
        XCTAssertEqual(result.scope.nodes[0].type,"conv_transpose2d")
        XCTAssertEqual(result.scope.nodes[0].config["stride"],.array([2,2]))
        XCTAssertEqual(result.scope.nodes[0].config["output_padding"],.array([1,1]))
        XCTAssertEqual(result.command.before.document.nodes[0].type,"conv2d")
        XCTAssertEqual(scope.nodes[0].type,"conv2d")
        let graph=GraphDocument(nodes:scope.nodes+[.init(id:"consumer",type:"relu")],edges:[.init(source:"conv",sourcePort:"indices",target:"consumer",targetPort:"in")])
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setConvolution(nodeID:"conv",variant:.conv3d),to:.init(document:graph),registry:registry),"Even an unfinished saved port must not be silently discarded")
    }
    func testInlineSameVariantKeepsRawBindingAndParameterPolicyWhileVariantSwapRefuses()throws {
        let registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        var config=try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry);config["bias"]=false
        let field=try XCTUnwrap(registry.component(type:"conv2d")?.fields.first {$0.key=="bias"})
        let policies:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        let definition=CompositeDefinition(reference:.init("conv-fixture"),title:"Convolution graph",body:.init(nodes:[.init(id:"conv",type:"conv2d",config:config,parameterSettings:policies)]),inputs:[.init("in",targets:[.init("conv","in")])],outputs:[.init("out",source:.init("conv","out"))],parameters:[.init(field,targets:[.init("conv","bias")])],composition:.init(kind:.graph,children:[.init(localID:"conv")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:["bias":true]))],definitions:[definition]);document.schemaVersion=15
        let proxy=try WorkflowInlineContainer.node(in:document,root:root,childID:"conv",registry:registry)
        guard case .modelPrimitive(_,_,var effective)=proxy.payload else {return XCTFail("Missing convolution")}
        XCTAssertEqual(effective["bias"],true);effective["stride"] = .array([2,2])
        let changed=try WorkflowInlineContainer.apply(.setConvolution(childID:"conv",variant:.conv2d,configuration:effective),to:document,root:root,registry:registry,idProvider:{"same-conv"})
        let stored=try WorkflowInlineContainer.definition(in:changed,root:root).body.nodes[0]
        XCTAssertEqual(stored.config["bias"],false)
        XCTAssertEqual(stored.config["stride"],.array([2,2]))
        XCTAssertEqual(stored.parameterSettings,policies)
        XCTAssertEqual(changed.definitions[0],definition)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setConvolution(childID:"conv",variant:.convTranspose2d),to:document,root:root,registry:registry,idProvider:{"bound-swap"}))
    }
    func testExactValidationAndExplicitRepairDoNotBlessMalformedSources()throws {
        let registry=try ComponentRegistry.builtIn()
        var transposed=try ConvolutionConfiguration.defaults(for:.convTranspose2d,registry:registry)
        transposed["stride"] = .array([1,1]);transposed["dilation"] = .array([1,1]);transposed["output_padding"] = .array([1,1])
        XCTAssertThrowsError(try ConvolutionConfiguration(componentID:"conv_transpose2d",configurationVersion:2,values:transposed,registry:registry))
        var depthwise=try ConvolutionConfiguration.defaults(for:.depthwiseConv2d,registry:registry);depthwise["groups"]=1
        XCTAssertThrowsError(try ConvolutionConfiguration(componentID:"depthwise_conv2d",configurationVersion:2,values:depthwise,registry:registry))
        var invalid=try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry);invalid["in_channels"]=0
        let node=WorkflowNode(address:.init(domain:.model,localID:"conv"),payload:.modelPrimitive(type:"conv2d",configurationVersion:2,config:invalid))
        XCTAssertThrowsError(try ConvolutionConfiguration.switching(node,to:.conv2d,registry:registry))
        let repaired=try ConvolutionConfiguration.switching(node,to:.conv2d,configuration:ConvolutionConfiguration.defaults(for:.conv2d,registry:registry),registry:registry)
        XCTAssertEqual(try ConvolutionConfiguration.configuration(of:repaired,registry:registry).variant,.conv2d)
        XCTAssertThrowsError(try ConvolutionConfiguration.switching(.init(address:.init(domain:.model,localID:"other"),payload:.modelPrimitive(type:"linear",configurationVersion:2,config:[:])),to:.conv2d,registry:registry))
    }
}
