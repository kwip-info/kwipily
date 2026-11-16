import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LinearInspectorDraftTests:XCTestCase {
    func testModeDraftRetainsSharedValuesAndUnpublishedExplicitWidth()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=LinearInspectorDraft(variant:.explicit,values:try LinearConfiguration.defaults(for:.explicit,registry:registry))
        draft.setText("7",key:"in_features");draft.setText("12",key:"out_features");draft.values["bias"]=false
        XCTAssertEqual(draft.values["in_features"],2)
        try draft.select(.inferred,registry:registry)
        XCTAssertNil(draft.values["in_features"]);XCTAssertEqual(draft.values["out_features"],12);XCTAssertEqual(draft.values["bias"],false)
        try draft.select(.explicit,registry:registry)
        XCTAssertEqual(draft.values["in_features"],7)
        draft.setText("0",key:"out_features");XCTAssertThrowsError(try draft.configuration(registry:registry))
        XCTAssertThrowsError(try draft.select(.inferred,registry:registry));XCTAssertEqual(draft.variant,.explicit)
        for invalid in ["513","2.5","nan",""] {draft.setText(invalid,key:"out_features");XCTAssertThrowsError(try draft.configuration(registry:registry))}
    }
    func testExistingLazySourceUsesItsOriginalValuesAndHonestExplicitDefault()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=LinearInspectorDraft(variant:.inferred,values:try LinearConfiguration.defaults(for:.inferred,registry:registry))
        XCTAssertEqual(draft.values["out_features"],2)
        try draft.select(.explicit,registry:registry)
        XCTAssertEqual(draft.values["in_features"],2);XCTAssertEqual(draft.values["out_features"],2)
    }
    @MainActor func testCanonicalChildOwnerRetainsRawExposedWidthAndRefusesSizingSwap()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        var config=try LinearConfiguration.defaults(for:.explicit,registry:registry);config["out_features"] = .integer(6)
        let field=try XCTUnwrap(registry.component(type:"linear")?.fields.first {$0.key=="out_features"})
        let definition=CompositeDefinition(reference:.init("linear-owner"),title:"Linear",body:.init(nodes:[.init(id:"norm",type:"linear",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("norm","in")])],outputs:[.init("out",source:.init("norm","out"))],parameters:[.init(field,targets:[.init("norm","out_features")])])
        let source=CompositeProjectDocument(name:"Linear",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setLinear(nodeID:"norm",variant:.inferred),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        var changed=config;changed["out_features"]=field.defaultValue;changed["in_features"] = 4
        XCTAssertNotNil(owner.applyDraftCommand(.setLinear(nodeID:"norm",variant:.explicit,configuration:changed),lease:owner.lease(path:["group"])),owner.error ?? "No error")
        guard case .blockModel(let model)=owner.document else {return XCTFail("Expected block model")}
        XCTAssertEqual(model.drafts.first?.proposed.body.nodes.first?.config["out_features"],.integer(6))
    }

}
