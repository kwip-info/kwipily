import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConvolutionInspectorDraftTests:XCTestCase {
    func testGridRejectsInvalidGeometryAndCommitsAxesTogether()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=ConvolutionInspectorDraft(variant:.conv3d,values:try ConvolutionConfiguration.defaults(for:.conv3d,registry:registry))
        draft.setText("5",key:"kernel_size",axis:2)
        draft.setText("2",key:"padding",axis:2)
        XCTAssertEqual(try draft.configuration(registry:registry)["kernel_size"],.array([3,3,5]))
        XCTAssertEqual(draft.values["kernel_size"],.array([3,3,3]))
        draft.setText("-",key:"stride",axis:1)
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("0",key:"stride",axis:1)
        XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    func testOperationChangePreservesCompatibleGeometryAndDerivesDepthwiseChannels()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=ConvolutionInspectorDraft(variant:.conv2d,values:try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry))
        draft.setText("4",key:"in_channels");draft.setText("8",key:"out_channels")
        draft.setText("5",key:"kernel_size",axis:1)
        try draft.select(.depthwiseConv2d,registry:registry)
        XCTAssertEqual(draft.values["in_channels"],4)
        XCTAssertEqual(draft.values["channel_multiplier"],2)
        XCTAssertEqual(draft.values["kernel_size"],.array([3,5]))
        XCTAssertNil(draft.values["groups"]);XCTAssertNil(draft.values["out_channels"])
        try draft.select(.convTranspose2d,registry:registry)
        XCTAssertEqual(draft.values["out_channels"],8)
        XCTAssertEqual(draft.values["groups"],4)
        XCTAssertNil(draft.values["padding_policy"]);XCTAssertNil(draft.values["padding_mode"])
    }
    func testPaddingPolicyClearsHiddenAmountsAndInvalidDrafts()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=ConvolutionInspectorDraft(variant:.conv2d,values:try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry))
        draft.setText("1",key:"padding",axis:0)
        draft.setText("-",key:"padding",axis:1)
        draft.setPaddingPolicy("same")
        XCTAssertEqual(try draft.configuration(registry:registry)["padding"],.array([0,0]))
        XCTAssertEqual(draft.values["padding_policy"],.string("same"))
        draft.setPaddingPolicy("explicit")
        XCTAssertEqual(draft.values["padding"],.array([0,0]))
    }
    @MainActor func testCanonicalChildOwnerKeepsBoundRawFieldsAndRefusesVariantSwap()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        var config=try ConvolutionConfiguration.defaults(for:.conv2d,registry:registry);config["bias"]=false
        let field=try XCTUnwrap(registry.component(type:"conv2d")?.fields.first {$0.key=="bias"})
        let definition=CompositeDefinition(reference:.init("convolution-owner"),title:"Conv",body:.init(nodes:[.init(id:"conv",type:"conv2d",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("conv","in")])],outputs:[.init("out",source:.init("conv","out"))],parameters:[.init(field,targets:[.init("conv","bias")])])
        let source=CompositeProjectDocument(name:"Conv",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setConvolution(nodeID:"conv",variant:.convTranspose2d),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        var changed=config;changed["bias"]=true;changed["stride"] = .array([2,2])
        XCTAssertNotNil(owner.applyDraftCommand(.setConvolution(nodeID:"conv",variant:.conv2d,configuration:changed),lease:owner.lease(path:["group"])),owner.error ?? "No error")
    }

}
