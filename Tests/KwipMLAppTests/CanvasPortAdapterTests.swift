import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasPortAdapterTests:XCTestCase {
    @MainActor private func workspace()->WorkspaceStore {
        WorkspaceStore(rootOverride:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),detectCapabilities:false)
    }
    @MainActor func testModelInputAndOutputDeclarationsHaveDifferentWidthsAndNoMeasuredClaims() {
        let store=workspace()
        store.graph.nodes=[GraphNode(id:"linear",type:"linear",name:"User label",x:0,y:0,config:["in_features":2,"out_features":6,"bias":true])]
        func describe<W:CanvasWorkspace>(_ store:W,_ port:String,_ direction:PortDirection)->CanvasPortDescriptor {store.canvasPortDescriptor(nodeID:"linear",port:port,direction:direction)}
        XCTAssertTrue(describe(store,"in",.input).declared.contains("[…, 2]"))
        XCTAssertTrue(describe(store,"out",.output).declared.contains("[…, 6]"))
        XCTAssertTrue(describe(store,"out",.output).measured.isEmpty)
        XCTAssertEqual(describe(store,"out",.input).kind,.unknown)
    }
    @MainActor func testDataRegistryDistinguishesPartitionStateAndTableWithoutInferringPreviewSchema() {
        let store=DataFlowWorkspaceStore(workspace:workspace());store.loadExample(columns:["x","y"])
        XCTAssertEqual(store.canvasConfiguration(nodeID:"split"),store.project.graph.nodes.first(where:{$0.id == "split"})?.config)
        XCTAssertEqual(store.canvasConfiguration(nodeID:"split")["seed"],42)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"split",port:"training",direction:.output).kind,.table)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"split",port:"training",direction:.output).partition,.training)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"fit",port:"state",direction:.output).kind,.fittedState)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"cast",port:"table",direction:.output).kind,.table)
        store.previews["cast"]=["table":["kind":"table","preview":["rowCount":100]]]
        XCTAssertTrue(store.canvasPortDescriptor(nodeID:"cast",port:"table",direction:.output).measured.isEmpty)
    }
    @MainActor func testCompositeBoundaryUsesReferencedDeclarationAndRejectsMissingOrConflictingTargets() throws {
        let store=CompositeWorkspaceStore(workspace:workspace())
        let definition=try XCTUnwrap(store.project.definitions.first)
        let input=try XCTUnwrap(definition.inputs.first)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"features",port:input.id,direction:.input).kind,.tensor)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"features",port:"absent",direction:.input).kind,.unknown)
        let original=store.state.project.definitions[0]
        let conflicting=GraphNode(id:"conflicting",type:"linear",name:"Different declared input",x:0,y:0,config:["in_features":7,"out_features":6,"bias":true])
        store.state.project.definitions[0].body.nodes.append(conflicting)
        store.state.project.definitions[0].inputs[0].targets.append(CompositePortReference("conflicting","in"))
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"features",port:input.id,direction:.input).kind,.unknown)
        store.state.project.definitions[0]=original
        store.state.project.definitions[0].inputs[0].targets.append(CompositePortReference("missing","in"))
        XCTAssertEqual(store.canvasPortDescriptor(nodeID:"features",port:input.id,direction:.input).kind,.unknown)
    }
}


extension CanvasPortAdapterTests {
    @MainActor func testActiveOuterDefaultsOverrideNestedOverridesLikeExpansion() throws {
        let store=CompositeWorkspaceStore(workspace:workspace())
        let innerRef=CompositeDefinitionReference("inner"),outerRef=CompositeDefinitionReference("outer")
        let inner=CompositeDefinition(reference:innerRef,title:"Inner",body:CompositeGraph(nodes:[GraphNode(id:"layer",type:"linear",config:["in_features":2,"out_features":2,"bias":true])]),inputs:[CompositeInput("in",targets:[CompositePortReference("layer","in")])],outputs:[CompositeOutput("out",source:CompositePortReference("layer","out"))],parameters:[CompositeParameter(ComponentField(key:"width",title:"Width",defaultValue:2),targets:[CompositeParameterTarget("layer","in_features"),CompositeParameterTarget("layer","out_features")])])
        let outer=CompositeDefinition(reference:outerRef,title:"Outer",body:CompositeGraph(instances:[CompositeInstance(id:"nested",definition:innerRef,overrides:["width":3])]),inputs:[CompositeInput("in",targets:[CompositePortReference("nested","in")])],outputs:[CompositeOutput("out",source:CompositePortReference("nested","out"))],parameters:[CompositeParameter(ComponentField(key:"outerWidth",title:"Outer width",defaultValue:5),targets:[CompositeParameterTarget("nested","width")])])
        store.state.project=CompositeProjectDocument(root:CompositeGraph(nodes:[GraphNode(id:"input",type:"input",config:["features":2]),GraphNode(id:"expand",type:"linear",config:["in_features":2,"out_features":5,"bias":true]),GraphNode(id:"head",type:"linear",config:["in_features":5,"out_features":2,"bias":true]),GraphNode(id:"output",type:"output")],instances:[CompositeInstance(id:"outer",definition:outerRef)],edges:[GraphEdge(source:"input",target:"expand"),GraphEdge(source:"expand",target:"outer"),GraphEdge(source:"outer",target:"head"),GraphEdge(source:"head",target:"output")]),definitions:[inner,outer])
        let expansion=try CompositeExpander.expand(store.project,registry:store.primitiveRegistry)
        let expandedID=try XCTUnwrap(expansion.nodeOrigins.first(where:{$0.value.localNodeID == "layer"})?.key)
        XCTAssertEqual(expansion.graph.nodes.first(where:{$0.id == expandedID})?.config["in_features"],5)
        XCTAssertEqual(expansion.graph.nodes.first(where:{$0.id == expandedID})?.config["out_features"],5)
        let root=store.canvasPortDescriptor(nodeID:"outer",port:"in",direction:.input)
        XCTAssertTrue(root.declared.contains("[…, 5]"))
        store.state.activeReference=outerRef
        let opened=store.canvasPortDescriptor(nodeID:"nested",port:"in",direction:.input)
        XCTAssertEqual(opened,root)
        XCTAssertEqual(store.canvasConfiguration(nodeID:"nested")["width"],5)
        XCTAssertTrue(store.canvasPortDescriptor(nodeID:"nested",port:"out",direction:.output).declared.contains("[…, 5]"))
        XCTAssertEqual(store.scope.instances[0].overrides["width"],3,"Read-only projection must not rewrite stored overrides")
        store.state.draftBase=outerRef;store.state.draft=outer
        store.state.draft?.parameters[0].field.defaultValue=7
        XCTAssertTrue(store.canvasPortDescriptor(nodeID:"nested",port:"in",direction:.input).declared.contains("[…, 7]"))
        XCTAssertEqual(store.canvasConfiguration(nodeID:"nested")["width"],7)
        XCTAssertEqual(store.project.definitions[1].parameters[0].field.defaultValue,5)
    }
}
