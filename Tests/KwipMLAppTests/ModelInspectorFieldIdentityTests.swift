import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelInspectorFieldIdentityTests: XCTestCase {
    @MainActor func testSuccessfulCommitKeepsFieldIdentityWhileOldMutationLeaseIsRejected() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),record=try library.save(ExampleFlows.parallelClassifier)
        let owner=try ModelAuthoringStore(library:library,projectID:record.id),adapter=try ModelCanvasAdapter(owner:owner)
        adapter.selectNode("branch-a",extending:false)
        let node=try XCTUnwrap(adapter.graph.nodes.first{$0.id=="branch-a"})
        let field=try XCTUnwrap(adapter.registry.component(type:node.type)?.fields.first{$0.key=="out_features"})
        let lease=owner.lease()
        let before=ModelInspectorFieldIdentity(contextKey:adapter.contextKey,node:node,field:field,lease:lease)
        XCTAssertTrue(adapter.setConfigValue(field.key,value:.integer(12),nodeID:node.id,lease:lease))
        let changed=try XCTUnwrap(adapter.graph.nodes.first{$0.id==node.id})
        let after=ModelInspectorFieldIdentity(contextKey:adapter.contextKey,node:changed,field:field,lease:owner.lease())
        XCTAssertEqual(before.viewID,after.viewID,"A committed value must not destroy the focused field's SwiftUI identity")
        XCTAssertNotEqual(before.origin,after.origin,"Callback authorization must still track the actual owner revision")
        XCTAssertFalse(adapter.setConfigValue(field.key,value:.integer(14),nodeID:node.id,lease:lease))
        XCTAssertTrue(adapter.setConfigValue(field.key,value:.integer(14),nodeID:node.id,lease:owner.lease()))
        adapter.renameSelected("Renamed",lease:owner.lease())
        let renamed=try XCTUnwrap(adapter.graph.nodes.first{$0.id==node.id})
        XCTAssertEqual(before.viewID,ModelInspectorFieldIdentity(contextKey:adapter.contextKey,node:renamed,field:field,lease:owner.lease()).viewID)
    }
    func testListCountBoundsChangeViewIdentity() {
        let field=ComponentField(key:"shape",title:"Shape",kind:.integerList,defaultValue:.array([-1]),min:-1,max:100,minItems:1,maxItems:7)
        let lease=ModelEditorLease(projectID:"p",sessionID:UUID(),editRevision:1,instancePath:[])
        let node=GraphNode(type:"reshape")
        let before=ModelInspectorFieldIdentity(contextKey:"scope",node:node,field:field,lease:lease)
        var changed=field;changed.maxItems=3
        XCTAssertNotEqual(before.viewID,ModelInspectorFieldIdentity(contextKey:"scope",node:node,field:changed,lease:lease).viewID)
    }
    func testDifferentFieldContractsAndContextsDoNotShareViewIdentity() throws {
        let registry=try ComponentRegistry.builtIn(),node=ExampleFlows.parallelClassifier.nodes[1]
        let field=try XCTUnwrap(registry.component(type:node.type)?.fields.first{$0.key=="out_features"})
        let lease=ModelEditorLease(projectID:"project",sessionID:UUID(),editRevision:1,instancePath:[])
        let original=ModelInspectorFieldIdentity(contextKey:"scope",node:node,field:field,lease:lease)
        var changedNode=node;changedNode.type="other-type"
        XCTAssertNotEqual(original.viewID,ModelInspectorFieldIdentity(contextKey:"scope",node:changedNode,field:field,lease:lease).viewID)
        changedNode=node;changedNode.id="another-node"
        XCTAssertNotEqual(original.viewID,ModelInspectorFieldIdentity(contextKey:"scope",node:changedNode,field:field,lease:lease).viewID)
        var changedField=field;changedField.max=128
        XCTAssertNotEqual(original.viewID,ModelInspectorFieldIdentity(contextKey:"scope",node:node,field:changedField,lease:lease).viewID)
        changedField=field;changedField.defaultValue = .integer(12)
        XCTAssertEqual(original.viewID,ModelInspectorFieldIdentity(contextKey:"scope",node:node,field:changedField,lease:lease).viewID)
        XCTAssertNotEqual(original.viewID,ModelInspectorFieldIdentity(contextKey:"another-scope",node:node,field:field,lease:lease).viewID)
    }
}
