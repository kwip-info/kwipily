import XCTest
@testable import KwipMLCore

final class PoolingConfigurationTests:XCTestCase {
    private var dataRegistry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    func testCatalogMatchesEightSupportedExistingContracts()throws {
        XCTAssertEqual(Set(PoolingConfiguration.Variant.allCases.map(\.rawValue)),Set(["max_pool1d","max_pool2d","max_pool3d","avg_pool1d","avg_pool2d","adaptive_avg_pool1d","adaptive_avg_pool2d","adaptive_max_pool2d"]))
        XCTAssertEqual(PoolingConfiguration.defaultVariant,.maxPool2d)
        XCTAssertEqual(PoolingConfiguration.Variant.maxPool3d.dimensions,3)
        XCTAssertEqual(PoolingConfiguration.Variant.adaptiveMaxPool2d.method,.max)
        XCTAssertEqual(PoolingConfiguration.Variant.adaptiveAvgPool1d.sizing,.adaptive)
        let registry=try ComponentRegistry.builtIn()
        for variant in PoolingConfiguration.Variant.allCases {
            let defaults=try PoolingConfiguration.defaults(for:variant,registry:registry)
            XCTAssertEqual(try PoolingConfiguration(componentID:variant.rawValue,configurationVersion:2,values:defaults,registry:registry).values,defaults)
        }
    }
    func testCanonicalClassicCommandProtectsActualIndicesEdgesAndPermitsCompatibleMax2D()throws {
        let registry=try ComponentRegistry.builtIn()
        var config=try PoolingConfiguration.defaults(for:.maxPool2d,registry:registry);config["return_indices"]=true
        let node=GraphNode(id:"pool",type:"max_pool2d",name:"My pooling",x:17,y:29,config:config)
        let graph=GraphDocument(nodes:[node,.init(id:"consumer",type:"relu")],edges:[.init(id:"indices-edge",source:"pool",sourcePort:"indices",target:"consumer",targetPort:"in")])
        let state=GraphEditorState(document:graph)
        for variant:PoolingConfiguration.Variant in [.avgPool2d,.maxPool1d,.maxPool3d] {
            XCTAssertThrowsError(try GraphCommandEngine.apply(.setPooling(nodeID:"pool",variant:variant),to:state,registry:registry))
        }
        var disabled=config;disabled["return_indices"]=false
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setPooling(nodeID:"pool",variant:.maxPool2d,configuration:disabled),to:state,registry:registry))
        var adaptive=try PoolingConfiguration.defaults(for:.adaptiveMaxPool2d,registry:registry);adaptive["return_indices"]=true
        let compatible=try GraphCommandEngine.apply(.setPooling(nodeID:"pool",variant:.adaptiveMaxPool2d,configuration:adaptive),to:state,registry:registry)
        XCTAssertEqual(compatible.after.document.edges,graph.edges)
        XCTAssertEqual(compatible.after.document.nodes[0].id,node.id)
        XCTAssertEqual(compatible.after.document.nodes[0].name,node.name)
        XCTAssertEqual(compatible.after.document.nodes[0].x,node.x)
        XCTAssertEqual(compatible.after.document.nodes[0].type,"adaptive_max_pool2d")
        XCTAssertEqual(state.document,graph)
    }
    func testWorkflowCommandCannotRemoveConnectedIndicesEvenFromUnfinishedGraph()throws {
        let registry=try ComponentRegistry.builtIn(),pool=WorkflowNodeAddress(domain:.model,localID:"pool"),consumer=WorkflowNodeAddress(domain:.model,localID:"consumer")
        var config=try PoolingConfiguration.defaults(for:.maxPool2d,registry:registry);config["return_indices"]=true
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:pool,payload:.modelPrimitive(type:"max_pool2d",configurationVersion:2,config:config)),.init(address:consumer,payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))],edges:[.init(from:.init(address:pool,port:"indices"),to:.init(address:consumer,port:"in"),kind:.tensor)])
        let bytes=try WorkflowCodec.encode(document)
        XCTAssertThrowsError(try WorkflowEditing.apply(.setPooling(address:pool,variant:.avgPool2d),to:document,modelRegistry:registry,dataRegistry:dataRegistry))
        var disabled=config;disabled["return_indices"]=false
        XCTAssertThrowsError(try WorkflowEditing.apply(.setPooling(address:pool,variant:.maxPool2d,configuration:disabled),to:document,modelRegistry:registry,dataRegistry:dataRegistry))
        XCTAssertEqual(try WorkflowCodec.encode(document),bytes)
    }
    private func container(indices:Bool=false,bound:Bool=false)throws->WorkflowDocument {
        let registry=try ComponentRegistry.builtIn()
        var config=try PoolingConfiguration.defaults(for:.maxPool2d,registry:registry);config["return_indices"] = .bool(indices)
        let parameter=try XCTUnwrap(registry.component(type:"max_pool2d")?.fields.first {$0.key=="ceil_mode"})
        let definition=CompositeDefinition(reference:.init("pool-fixture"),title:"Pooling graph",body:.init(nodes:[.init(id:"pool",type:"max_pool2d",config:config)]),inputs:[.init("in",targets:[.init("pool","in")])],outputs:[.init("out",source:.init("pool","out"))]+(indices ? [.init("indices",source:.init("pool","indices"))]:[]),parameters:bound ? [.init(parameter,targets:[.init("pool","ceil_mode")])]:[],composition:.init(kind:.graph,children:[.init(localID:"pool")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:.init(domain:.model,localID:"container"),payload:.modelBlock(reference:definition.reference,overrides:bound ? ["ceil_mode":true]:[:]))],definitions:[definition]);document.schemaVersion=11
        return document
    }
    func testInlineBoundaryIndicesAreProtectedAndCompatibleMax2DSwapPreservesInterface()throws {
        let document=try container(indices:true),registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        let bytes=try WorkflowCodec.encode(document)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setPooling(childID:"pool",variant:.avgPool2d),to:document,root:root,registry:registry,idProvider:{"refused"}))
        var adaptive=try PoolingConfiguration.defaults(for:.adaptiveMaxPool2d,registry:registry);adaptive["return_indices"]=true
        let changed=try WorkflowInlineContainer.apply(.setPooling(childID:"pool",variant:.adaptiveMaxPool2d,configuration:adaptive),to:document,root:root,registry:registry,idProvider:{"compatible"})
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:changed,root:root).outputs,document.definitions[0].outputs)
        XCTAssertEqual(Array(changed.definitions.prefix(document.definitions.count)),document.definitions)
        XCTAssertEqual(try WorkflowCodec.encode(document),bytes)
    }
    func testInlineSameVariantPreservesRawBoundFieldsAndVariantSwapRefuses()throws {
        let document=try container(bound:true),registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        let proxy=try WorkflowInlineContainer.node(in:document,root:root,childID:"pool",registry:registry)
        guard case .modelPrimitive(_,_,var config)=proxy.payload else {return XCTFail("Missing pool")}
        XCTAssertEqual(config["ceil_mode"],true)
        config["kernel_size"] = .array([3,3])
        let changed=try WorkflowInlineContainer.apply(.setPooling(childID:"pool",variant:.maxPool2d,configuration:config),to:document,root:root,registry:registry,idProvider:{"same-variant"})
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:changed,root:root).body.nodes[0].config["ceil_mode"],false)
        XCTAssertEqual(try PoolingConfiguration.configuration(of:WorkflowInlineContainer.node(in:changed,root:root,childID:"pool",registry:registry),registry:registry).values["ceil_mode"],true)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setPooling(childID:"pool",variant:.avgPool2d),to:document,root:root,registry:registry,idProvider:{"bound-swap"}))
    }
    func testScopedConfigChangesPersistAndStrictGeometryCannotBeBypassed()throws {
        let registry=try ComponentRegistry.builtIn(),document=try container(),scope=document.definitions[0].body
        var average=try PoolingConfiguration.defaults(for:.avgPool2d,registry:registry);average["divisor_override"] = -3
        let changed=try ModelScopeEditing.apply(.setPooling(nodeID:"pool",variant:.avgPool2d,configuration:average),to:scope,definitions:document.definitions,registry:registry)
        XCTAssertEqual(changed.scope.nodes[0].type,"avg_pool2d")
        XCTAssertEqual(changed.scope.nodes[0].config["divisor_override"],-3)
        XCTAssertEqual(changed.scope.configurationVersion,2)
        average["divisor_override"]=0
        XCTAssertThrowsError(try PoolingConfiguration(componentID:"avg_pool2d",configurationVersion:2,values:average,registry:registry))
        var adaptive=try PoolingConfiguration.defaults(for:.adaptiveAvgPool1d,registry:registry);adaptive["output_size"] = .null
        XCTAssertThrowsError(try PoolingConfiguration(componentID:"adaptive_avg_pool1d",configurationVersion:2,values:adaptive,registry:registry))
        var unknown=try PoolingConfiguration.defaults(for:.maxPool2d,registry:registry);unknown["unexpected"]=true
        XCTAssertThrowsError(try PoolingConfiguration(componentID:"max_pool2d",configurationVersion:2,values:unknown,registry:registry))
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let bytes=try encoder.encode(scope)
        XCTAssertThrowsError(try ModelScopeEditing.apply(.setPooling(nodeID:"pool",variant:.avgPool2d,configuration:average),to:scope,definitions:document.definitions,registry:registry))
        XCTAssertEqual(try encoder.encode(scope),bytes)
    }
}
