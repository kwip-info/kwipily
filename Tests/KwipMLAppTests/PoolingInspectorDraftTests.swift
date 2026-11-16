import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PoolingInspectorDraftTests:XCTestCase {
    func testAxisGridCommitsTogetherAndRejectsIncompleteOrInvalidGeometry()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=PoolingInspectorDraft(variant:.maxPool3d,values:try PoolingConfiguration.defaults(for:.maxPool3d,registry:registry))
        draft.setText("3",key:"kernel_size",axis:2)
        draft.setText("1",key:"padding",axis:2)
        XCTAssertEqual(try draft.configuration(registry:registry)["kernel_size"],.array([2,2,3]))
        XCTAssertEqual(try draft.configuration(registry:registry)["padding"],.array([0,0,1]))
        XCTAssertEqual(draft.values["kernel_size"],.array([2,2,2]),"Draft editing cannot publish source early")
        draft.setText("-",key:"dilation",axis:1)
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("0",key:"dilation",axis:1)
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("2",key:"dilation",axis:1)
        XCTAssertEqual(try draft.configuration(registry:registry)["dilation"],.array([1,2,1]))
    }
    func testSignedDivisorAndAdaptiveKeepAxisRetainExactContracts()throws {
        let registry=try ComponentRegistry.builtIn()
        var average=PoolingInspectorDraft(variant:.avgPool2d,values:try PoolingConfiguration.defaults(for:.avgPool2d,registry:registry))
        average.setText("-7",key:"divisor_override")
        XCTAssertEqual(try average.configuration(registry:registry)["divisor_override"],-7)
        average.setText("0",key:"divisor_override")
        XCTAssertThrowsError(try average.configuration(registry:registry))
        var adaptive=PoolingInspectorDraft(variant:.adaptiveAvgPool2d,values:try PoolingConfiguration.defaults(for:.adaptiveAvgPool2d,registry:registry))
        adaptive.values["output_height"] = .null
        adaptive.setText("7",key:"output_width")
        XCTAssertEqual(try adaptive.configuration(registry:registry),["output_height":.null,"output_width":7])
    }
    func testMethodChangePreservesCompatibleGeometryAndMaxSizingPreservesEnabledIndices()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=PoolingInspectorDraft(variant:.maxPool2d,values:try PoolingConfiguration.defaults(for:.maxPool2d,registry:registry))
        draft.setText("3",key:"kernel_size",axis:1)
        draft.values["stride"] = .array([1,2])
        draft.values["return_indices"] = true
        try draft.select(.avgPool2d,registry:registry)
        XCTAssertEqual(try draft.configuration(registry:registry)["kernel_size"],.array([2,3]))
        XCTAssertEqual(draft.values["stride"],.array([1,2]))
        XCTAssertNil(draft.values["return_indices"]);XCTAssertNil(draft.values["dilation"])
        try draft.select(.maxPool2d,registry:registry)
        draft.values["return_indices"] = true
        try draft.select(.adaptiveMaxPool2d,registry:registry)
        XCTAssertEqual(draft.values["return_indices"],true)
        XCTAssertNil(draft.values["kernel_size"])
    }
    @MainActor func testCanonicalChildOwnerProtectsExposedIndicesWithoutInternalEdges()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        var config=try PoolingConfiguration.defaults(for:.maxPool2d,registry:.builtIn());config["return_indices"]=true
        let definition=CompositeDefinition(reference:.init("pooling-owner"),title:"Pool",body:.init(nodes:[.init(id:"pool",type:"max_pool2d",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("pool","in")])],outputs:[.init("out",source:.init("pool","out")),.init("indices",source:.init("pool","indices"))])
        let source=CompositeProjectDocument(name:"Pool",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setPooling(nodeID:"pool",variant:.avgPool2d),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        var disabled=config;disabled["return_indices"]=false
        XCTAssertNil(owner.applyDraftCommand(.setPooling(nodeID:"pool",variant:.maxPool2d,configuration:disabled),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        var adaptive=try PoolingConfiguration.defaults(for:.adaptiveMaxPool2d,registry:.builtIn());adaptive["return_indices"]=true
        XCTAssertNotNil(owner.applyDraftCommand(.setPooling(nodeID:"pool",variant:.adaptiveMaxPool2d,configuration:adaptive),lease:owner.lease(path:["group"])),owner.error ?? "No error")
    }
}
