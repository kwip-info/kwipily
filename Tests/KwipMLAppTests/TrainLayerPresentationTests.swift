import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainLayerPresentationTests:XCTestCase {
    func testAuthoredRootNamesAndFriendlyDuplicateNumberingDoNotExposeInternalIDs()throws {
        let registry=try ComponentRegistry.builtIn()
        let nodes=[GraphNode(id:"private-a",type:"linear",name:"Projection"),GraphNode(id:"private-b",type:"linear",name:"Projection"),GraphNode(id:"private-c",type:"relu",name:"  ")]
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:nodes.map {.init(address:.init(domain:.model,localID:$0.id),name:$0.name,payload:.modelPrimitive(type:$0.type,configurationVersion:2,config:$0.config))})
        let graph=GraphDocument(nodes:nodes),origins=Dictionary(uniqueKeysWithValues:nodes.map {($0.id,CompositeNodeOrigin(instancePath:[],localNodeID:$0.id,definition:nil))})
        let names=TrainLayerPresentation.names(for:graph,origins:origins,document:document,registry:registry)
        XCTAssertEqual(names["private-a"],"Projection (1)")
        XCTAssertEqual(names["private-b"],"Projection (2)")
        XCTAssertEqual(names["private-c"],registry.component(type:"relu")?.title)
        XCTAssertFalse(names.values.contains {$0.contains("private-")})
        XCTAssertEqual(names,TrainLayerPresentation.names(for:graph,origins:origins,document:document,registry:registry))
        XCTAssertEqual(graph.nodes,nodes,"Presentation must not rewrite model identities or configuration")
    }
    func testNestedAuthoredBlockPathsDisambiguateRepeatedInstancesAndLayers()throws {
        let registry=try ComponentRegistry.builtIn(),inner=CompositeDefinitionReference("inner-private"),outer=CompositeDefinitionReference("outer-private")
        let leafA=GraphNode(id:"local-a",type:"linear",name:"Dense"),leafB=GraphNode(id:"local-b",type:"linear",name:"Dense")
        let definitions=[CompositeDefinition(reference:inner,title:"Inner fallback",body:.init(nodes:[leafA,leafB]),inputs:[],outputs:[]),CompositeDefinition(reference:outer,title:"Outer fallback",body:.init(instances:[.init(id:"branch-a",definition:inner,name:"Branch"),.init(id:"branch-b",definition:inner,name:"Branch")]),inputs:[],outputs:[])]
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:.init(domain:.model,localID:"root-a"),name:"Encoder",payload:.modelBlock(reference:outer,overrides:[:])),.init(address:.init(domain:.model,localID:"root-b"),name:"Encoder",payload:.modelBlock(reference:outer,overrides:[:]))],definitions:definitions)
        var nodes:[GraphNode]=[],origins:[String:CompositeNodeOrigin]=[:]
        for root in ["root-a","root-b"] {for branch in ["branch-a","branch-b"] {for local in [leafA,leafB] {
            let id="n-"+root+"-"+branch+"-"+local.id
            nodes.append(.init(id:id,type:"linear",name:local.name))
            origins[id] = .init(instancePath:[root,branch],localNodeID:local.id,definition:inner)
        }}}
        let names=TrainLayerPresentation.names(for:.init(nodes:nodes),origins:origins,document:document,registry:registry)
        XCTAssertEqual(names[nodes[0].id],"Encoder (1) / Branch (1) / Dense (1)")
        XCTAssertEqual(names[nodes[3].id],"Encoder (1) / Branch (2) / Dense (2)")
        XCTAssertEqual(names[nodes[7].id],"Encoder (2) / Branch (2) / Dense (2)")
        XCTAssertEqual(Set(names.values).count,8)
        XCTAssertFalse(names.values.contains {$0.contains("n-") || $0.contains("private") || $0.contains("root-")})
    }
    func testMissingProvenanceFallsBackToComponentTitlesAndNeverHashIDs()throws {
        let registry=try ComponentRegistry.builtIn(),graph=GraphDocument(nodes:[.init(id:String(repeating:"a",count:64),type:"linear"),.init(id:String(repeating:"b",count:64),type:"linear")])
        let names=TrainLayerPresentation.names(for:graph,origins:[:],document:.init(projectID:UUID().uuidString),registry:registry)
        let title=try XCTUnwrap(registry.component(type:"linear")?.title)
        XCTAssertEqual(names[graph.nodes[0].id],title+" (1)");XCTAssertEqual(names[graph.nodes[1].id],title+" (2)")
    }
    func testParameterAndGradientTitlesKeepExactBindingIdentitiesPrivate() {
        let weight=InitializationOwnerChoice(node:"n-private",parameter:"weight",name:"Encoder / Projection",shape:[4,2])
        let bias=InitializationOwnerChoice(node:"n-private",parameter:"bias",name:"Encoder / Projection",shape:[4])
        XCTAssertEqual(weight.title,"Encoder / Projection · Weights · 4 × 2")
        XCTAssertEqual(bias.title,"Encoder / Projection · Biases · 4")
        XCTAssertEqual(weight.node,"n-private");XCTAssertEqual(weight.parameter,"weight");XCTAssertNotEqual(weight.id,bias.id)
        let endpoint=GradientEndpointChoice(node:"n-private",port:"out",name:"Encoder / Projection",dtype:.float32,shape:[4],isModelInput:false)
        XCTAssertEqual(endpoint.title,"Encoder / Projection · out · float32 [4]")
        XCTAssertEqual(endpoint.node,"n-private");XCTAssertEqual(endpoint.port,"out")
        XCTAssertFalse(endpoint.title.contains(endpoint.node))
    }
    func testActualOwnerResolverUsesAuthoredNamesWhilePreservingNativeParameterBindings()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let index=try XCTUnwrap(document.nodes.firstIndex {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false})
        let address=document.nodes[index].address;document.nodes[index].name="Customer prediction"
        let owners=InitializationOwnerResolver.choices(document:document,modelRegistry:registry,dataRegistry:data,selectedTrain:WorkflowExamples.trainAddress)
        let weight=try XCTUnwrap(owners.first {$0.node==address.localID && $0.parameter=="weight"})
        XCTAssertEqual(weight.name,"Customer prediction")
        XCTAssertTrue(weight.title.hasPrefix("Customer prediction · Weights · "))
        XCTAssertEqual(weight.node,address.localID);XCTAssertFalse(weight.shape.isEmpty)
    }
}
