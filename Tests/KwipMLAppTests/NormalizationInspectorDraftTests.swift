import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NormalizationInspectorDraftTests:XCTestCase {
    func testShapeGridKeepsOrderedDraftAndBounds()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=NormalizationInspectorDraft(variant:.layerNorm,values:try NormalizationConfiguration.defaults(for:.layerNorm,registry:registry))
        draft.setText("4",key:"normalized_shape",axis:0)
        try draft.addDimension()
        draft.setText("5",key:"normalized_shape",axis:1)
        XCTAssertEqual(try draft.configuration(registry:registry)["normalized_shape"],.array([4,5]))
        XCTAssertEqual(draft.values["normalized_shape"],.array([2,2]),"Numbers stay unpublished until Apply")
        try draft.removeDimension(at:0)
        XCTAssertEqual(try draft.configuration(registry:registry)["normalized_shape"],.array([5]))
        XCTAssertThrowsError(try draft.removeDimension(at:0))
        for _ in 1..<7 {try draft.addDimension()}
        XCTAssertThrowsError(try draft.addDimension())
    }
    func testScientificNumbersAndInvalidGroupDivisibility()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=NormalizationInspectorDraft(variant:.groupNorm,values:try NormalizationConfiguration.defaults(for:.groupNorm,registry:registry))
        draft.setText("1e-4",key:"eps")
        XCTAssertEqual(try draft.configuration(registry:registry)["eps"],.number(0.0001))
        draft.setText("nan",key:"eps");XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("1e-5",key:"eps");draft.setText("3",key:"num_groups")
        XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    func testMethodPreservesChannelsAndAffineChoicesButUsesHonestStatisticsDefaults()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=NormalizationInspectorDraft(variant:.batchNorm2d,values:try NormalizationConfiguration.defaults(for:.batchNorm2d,registry:registry))
        draft.setText("16",key:"num_features")
        draft.values["affine"]=true
        try draft.select(.instanceNorm2d,registry:registry)
        XCTAssertEqual(draft.values["num_features"],16)
        XCTAssertEqual(draft.values["affine"],true)
        XCTAssertEqual(draft.values["track_running_stats"],false)
        XCTAssertNil(draft.values["averaging"])
        try draft.select(.groupNorm,registry:registry)
        XCTAssertEqual(draft.values["num_channels"],16)
        XCTAssertNil(draft.values["track_running_stats"])
    }
    @MainActor func testCanonicalChildOwnerRetainsRawExposedEpsilonAndRefusesMethodSwap()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        var config=try NormalizationConfiguration.defaults(for:.layerNorm,registry:registry);config["eps"] = .number(0.01)
        let field=try XCTUnwrap(registry.component(type:"layer_norm")?.fields.first {$0.key=="eps"})
        let definition=CompositeDefinition(reference:.init("normalization-owner"),title:"Norm",body:.init(nodes:[.init(id:"norm",type:"layer_norm",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("norm","in")])],outputs:[.init("out",source:.init("norm","out"))],parameters:[.init(field,targets:[.init("norm","eps")])])
        let source=CompositeProjectDocument(name:"Norm",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setNormalization(nodeID:"norm",variant:.batchNorm1d),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        var changed=config;changed["eps"]=field.defaultValue;changed["normalized_shape"] = .array([4])
        XCTAssertNotNil(owner.applyDraftCommand(.setNormalization(nodeID:"norm",variant:.layerNorm,configuration:changed),lease:owner.lease(path:["group"])),owner.error ?? "No error")
        guard case .blockModel(let model)=owner.document else {return XCTFail("Expected block model")}
        XCTAssertEqual(model.drafts.first?.proposed.body.nodes.first?.config["eps"],.number(0.01))
    }

}

extension NormalizationInspectorDraftTests {
    @MainActor func testRMSMethodShapeAndScaleDraftsPreserveLayerDefaults()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=NormalizationInspectorDraft(variant:.layerNorm,values:try NormalizationConfiguration.defaults(for:.layerNorm,registry:registry))
        draft.setText("3",key:"normalized_shape",axis:0);draft.setText("1e-6",key:"eps")
        try draft.select(.rmsNorm,registry:registry)
        XCTAssertEqual(draft.values["normalized_shape"],.array([3]));XCTAssertEqual(draft.values["eps"],.number(1e-6));XCTAssertNil(draft.values["bias"])
        try draft.addDimension();draft.setText("4",key:"normalized_shape",axis:1)
        XCTAssertEqual(try draft.configuration(registry:registry)["normalized_shape"],.array([3,4]))
        try draft.removeDimension(at:1);draft.values["elementwise_affine"]=false
        try draft.select(.layerNorm,registry:registry)
        XCTAssertEqual(draft.values["normalized_shape"],.array([3]));XCTAssertEqual(draft.values["elementwise_affine"],false);XCTAssertEqual(draft.values["bias"],true)
        try draft.select(.rmsNorm,registry:registry);draft.setText("0",key:"eps")
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"rms_norm"))
    }
}
