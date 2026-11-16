import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CompositeIntegerProductAppTests: XCTestCase {
    private func source() throws -> CompositeProjectDocument {
        let original = ExampleFlows.depthwiseConv2dClassifier()
        var depthwise = try XCTUnwrap(original.nodes.first { $0.id == "convolution" })
        depthwise.id = "depthwise"
        var config = Dictionary(uniqueKeysWithValues: Conv2dComponent.definition.fields.map { ($0.key, $0.defaultValue) })
        config["in_channels"] = 4; config["out_channels"] = 4
        config["kernel_size"] = .array([1,1]); config["groups"] = 1
        config["padding_policy"] = .string("valid"); config["padding"] = .array([0,0])
        let pointwise = GraphNode(id: "pointwise", type: "conv2d", config: config)
        var definition = CompositeDefinition(reference: .init("test.separable"), title: "Separable",
            body: .init(nodes: [depthwise,pointwise], edges: [.init(id: "inner", source: "depthwise", target: "pointwise")]),
            inputs: [.init("in",targets:[.init("depthwise","in")])], outputs: [.init("out",source:.init("pointwise","out"))],
            parameters: [
                .init(.init(key:"channels",title:"Channels",defaultValue:2,min:1,max:512),targets:[.init("depthwise","in_channels")]),
                .init(.init(key:"multiplier",title:"Multiplier",defaultValue:2,min:1,max:512),targets:[.init("depthwise","channel_multiplier")])])
        definition.integerProducts = [.init(lhs:"channels",rhs:"multiplier",targets:[.init("pointwise","in_channels")])]
        return CompositeProjectDocument(name: "Derived channels", root: .init(
            nodes: original.nodes.filter { $0.id != "convolution" },
            instances: [.init(id:"convolution",definition:definition.reference)], edges: original.edges), definitions:[definition])
    }
    @MainActor private func setup() throws -> (URL,WorkflowAuthoringStore,ModelChildEditorSession,ModelCanvasAdapter) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:directory)
        let model=ModelAuthoringDocument(projectID:UUID().uuidString,revision:5,source:try source())
        let workflow=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        _=try library.saveSource(.workflow(workflow))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:workflow.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:.init(domain:.model,localID:"convolution"))
        let session=ModelChildEditorSession(owner:authority)
        XCTAssertTrue(session.open(path:["convolution"]))
        return(directory,owner,session,try XCTUnwrap(session.editor))
    }
    @MainActor func testUniversalProjectionTracksProductAndRejectsDirectWritesWithoutMutatingAuthoredBody() throws {
        let(directory,owner,session,editor)=try setup();defer{try? FileManager.default.removeItem(at:directory)}
        let body=try XCTUnwrap(editor.activeDefinition?.body)
        XCTAssertNotNil(editor.derivedParameter(nodeID:"pointwise",field:"in_channels"))
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:3,context:editor.boundaryRevision))
        XCTAssertEqual(editor.graph.nodes.first{$0.id=="pointwise"}?.config["in_channels"],6)
        XCTAssertEqual(editor.canvasConfiguration(nodeID:"pointwise")["in_channels"],6)
        XCTAssertEqual(editor.graph.nodes.first{$0.id=="depthwise"}?.config["channel_multiplier"],3)
        XCTAssertEqual(editor.activeDefinition?.body,body)
        editor.selectNode("pointwise",extending:false)
        let before=owner.document
        XCTAssertFalse(editor.setConfigValue("in_channels",value:5,nodeID:"pointwise",lease:editor.owner.lease(path:editor.path)))
        XCTAssertEqual(owner.document,before)
        var configured=try source();configured.definitions[0]=try XCTUnwrap(editor.activeDefinition)
        let expanded=try CompositeExpander.expand(configured,registry:owner.modelRegistry)
        let id=try XCTUnwrap(expanded.nodeOrigins.first{$0.value.localNodeID=="pointwise"}?.key)
        XCTAssertEqual(expanded.graph.nodes.first{$0.id==id}?.config["in_channels"],6)
        XCTAssertTrue(session.apply())
        XCTAssertEqual(owner.document.revision,before.revision+1)
    }
    @MainActor func testInvalidProductSurfacesProjectionErrorAndCannotApply() throws {
        let(directory,owner,session,editor)=try setup();defer{try? FileManager.default.removeItem(at:directory)}
        let originalPin=owner.document.nodes.first{$0.address.localID=="convolution"}?.payload
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:512,context:editor.boundaryRevision))
        XCTAssertNotNil(editor.projectionError)
        XCTAssertTrue(editor.graph.nodes.contains { $0.id == "pointwise" })
        XCTAssertFalse(session.apply())
        XCTAssertEqual(owner.document.nodes.first{$0.address.localID=="convolution"}?.payload,originalPin)
        XCTAssertNotNil(session.editor)
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:3,context:editor.boundaryRevision))
        XCTAssertNil(editor.projectionError)
        XCTAssertEqual(editor.graph.nodes.first { $0.id == "pointwise" }?.config["in_channels"],6)
    }
    @MainActor func testLegacyProjectionAndCanvasConfigurationUseSameProductAndGuardTarget() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        workspace.activeProjectID=UUID().uuidString
        let store=CompositeWorkspaceStore(workspace:workspace)
        store.state.project=try source()
        store.openDefinition(instanceID:"convolution");store.beginRevision()
        XCTAssertNotNil(store.derivedParameter(nodeID:"pointwise",field:"in_channels"))
        XCTAssertTrue(store.setBoundaryValue(key:"multiplier",value:3,context:store.boundaryRevision))
        XCTAssertEqual(store.graph.nodes.first{$0.id=="pointwise"}?.config["in_channels"],6)
        XCTAssertEqual(store.canvasConfiguration(nodeID:"pointwise")["in_channels"],6)
        XCTAssertEqual(store.activeDefinition?.body.nodes.first{$0.id=="pointwise"}?.config["in_channels"],4)
        store.selectNode("pointwise",extending:false)
        let before=store.state
        XCTAssertFalse(store.setConfigValue("in_channels",value:5,nodeID:"pointwise"))
        XCTAssertEqual(store.state,before)
    }
    @MainActor func testReleaseDerivedBindingPreservesValueAndRejectsStaleContext() throws {
        let(directory,owner,session,editor)=try setup();defer{withExtendedLifetime(session) {};try? FileManager.default.removeItem(at:directory)}
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:3,context:editor.boundaryRevision))
        let before=owner.document
        XCTAssertFalse(editor.releaseIntegerProduct(at:0,context:UUID()))
        XCTAssertEqual(owner.document,before)
        XCTAssertTrue(editor.releaseIntegerProduct(at:0,context:editor.boundaryRevision))
        XCTAssertNil(editor.derivedParameter(nodeID:"pointwise",field:"in_channels"))
        XCTAssertEqual(editor.activeDefinition?.body.nodes.first{$0.id=="pointwise"}?.config["in_channels"],6)
        editor.selectNode("pointwise",extending:false)
        XCTAssertTrue(editor.setConfigValue("in_channels",value:5,nodeID:"pointwise",lease:editor.owner.lease(path:editor.path)))
    }

    @MainActor func testChildPortsExposeConfiguredTensorContractsThroughCanvasProtocol() throws {
        let(directory,_,session,editor)=try setup()
        defer { withExtendedLifetime(session) {}; try? FileManager.default.removeItem(at:directory) }
        let canvas: any CanvasWorkspace = editor
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:3,context:editor.boundaryRevision))
        let input = canvas.canvasPortDescriptor(nodeID:"pointwise",port:"in",direction:.input)
        XCTAssertEqual(input.kind,.tensor)
        XCTAssertTrue(input.declared.joined().contains("6"))
        XCTAssertTrue(input.measured.isEmpty)
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"depthwise",port:"out",direction:.output).kind,.tensor)
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"pointwise",port:"missing",direction:.input).kind,.unknown)
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"missing",port:"in",direction:.input).kind,.unknown)
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:512,context:editor.boundaryRevision))
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"pointwise",port:"in",direction:.input).kind,.unknown)
        XCTAssertTrue(editor.setBoundaryValue(key:"multiplier",value:2,context:editor.boundaryRevision))
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"pointwise",port:"in",direction:.input).kind,.tensor)
    }

}
