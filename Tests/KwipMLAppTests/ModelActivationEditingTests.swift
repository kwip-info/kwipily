import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelActivationEditingTests:XCTestCase {
    @MainActor private func model()throws->(ModelAuthoringStore,ModelCanvasAdapter) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),record=try library.save(ExampleFlows.parallelClassifier)
        let owner=try ModelAuthoringStore(library:library,projectID:record.id)
        return(owner,try ModelCanvasAdapter(owner:owner))
    }
    @MainActor func testCanonicalOwnerSwitchPreservesIdentityWiresAndUndo()throws {
        let(owner,canvas)=try model()
        let node=try XCTUnwrap(canvas.graph.nodes.first {$0.type=="relu"});canvas.selectNode(node.id,extending:false)
        let before=owner.document,edges=canvas.graph.edges,lease=owner.lease()
        XCTAssertTrue(canvas.setActivation(.prelu,nodeID:node.id,lease:lease,expectedType:node.type,expectedSourceConfig:node.config))
        let changed=try XCTUnwrap(canvas.scope?.nodes.first {$0.id==node.id})
        XCTAssertEqual(changed.type,"prelu");XCTAssertEqual(changed.name,node.name);XCTAssertEqual(changed.x,node.x);XCTAssertEqual(changed.y,node.y)
        XCTAssertEqual(canvas.graph.edges,edges);XCTAssertEqual(canvas.selectedNodeID,node.id)
        XCTAssertFalse(canvas.setActivation(.sigmoid,nodeID:node.id,lease:lease,expectedType:node.type,expectedSourceConfig:node.config),"Old UI lease cannot switch a newer primitive")
        canvas.undo();XCTAssertEqual(owner.document,before)
        canvas.redo();XCTAssertEqual(canvas.scope?.nodes.first {$0.id==node.id}?.type,"prelu")
    }
    @MainActor func testBusyInvalidAndStaleSourceCallbacksNeverMutateOwner()throws {
        let(owner,canvas)=try model(),node=try XCTUnwrap(canvas.graph.nodes.first {$0.type=="relu"})
        canvas.selectNode(node.id,extending:false);let before=owner.document
        XCTAssertFalse(canvas.setActivation(.gelu,nodeID:node.id,lease:owner.lease(),expectedType:"sigmoid",expectedSourceConfig:node.config))
        XCTAssertFalse(canvas.setActivation(.gelu,nodeID:node.id,lease:owner.lease(),expectedType:node.type,expectedSourceConfig:["unknown":.integer(1)]))
        owner.setExecutionBusy(true)
        XCTAssertFalse(canvas.setActivation(.gelu,nodeID:node.id,lease:owner.lease(),expectedType:node.type,expectedSourceConfig:node.config))
        owner.setExecutionBusy(false);canvas.invalidate()
        XCTAssertFalse(canvas.setActivation(.gelu,nodeID:node.id,lease:owner.lease(),expectedType:node.type,expectedSourceConfig:node.config))
        XCTAssertEqual(owner.document,before)
    }
    @MainActor func testClassicSwitchHonorsProjectSelectionRestrictionAndUndo()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:root,storageOverride:storage,detectCapabilities:false)
        store.legacyGraph=try GraphMigration.toCurrent(ExampleFlows.parallelClassifier,registry:store.registry)
        let node=try XCTUnwrap(store.graph.nodes.first {$0.type=="relu"});store.selectNode(node.id)
        let before=store.graph
        XCTAssertFalse(store.setActivation(.sigmoid,nodeID:node.id,expectedType:node.type,expectedSourceConfig:node.config,
            expectedProjectID:store.activeProjectID,allowedFunctions:[.relu]))
        XCTAssertTrue(store.setActivation(.sigmoid,nodeID:node.id,expectedType:node.type,expectedSourceConfig:node.config,
            expectedProjectID:store.activeProjectID,allowedFunctions:[.relu,.sigmoid]))
        XCTAssertEqual(store.graph.nodes.first {$0.id==node.id}?.type,"sigmoid");XCTAssertEqual(store.graph.edges,before.edges)
        store.undo();XCTAssertEqual(store.graph,before)
        XCTAssertFalse(store.setActivation(.tanh,nodeID:node.id,expectedType:node.type,expectedSourceConfig:node.config,expectedProjectID:UUID().uuidString))
    }
    @MainActor func testChildDraftSwitchStaysPrivateUntilApplied()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        let root=try ModelCanvasAdapter(owner:owner),before=root.graph
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        let child=try ModelCanvasAdapter(owner:owner,path:["features"])
        child.selectNode("activation-a",extending:false)
        let node=try XCTUnwrap(child.graph.nodes.first {$0.id=="activation-a"})
        XCTAssertTrue(child.setActivation(.gelu,configuration:["approximate_tanh":.bool(true)],nodeID:node.id,
            lease:owner.lease(path:["features"]),expectedType:node.type,expectedSourceConfig:node.config))
        XCTAssertEqual(child.scope?.nodes.first {$0.id==node.id}?.type,"gelu")
        XCTAssertEqual(child.scope?.nodes.first {$0.id==node.id}?.config["approximate_tanh"],.bool(true))
        XCTAssertEqual(root.graph,before,"Editing an owned draft cannot mutate the applied parent source")
        XCTAssertTrue(owner.applyDraft(lease:owner.lease(path:["features"])))
        XCTAssertTrue(child.isBusy)
        let snapshot=try owner.executionSnapshot()
        let expandedID=try XCTUnwrap(snapshot.origins.first {$0.value.instancePath==["features"] && $0.value.localNodeID=="activation-a"}?.key)
        XCTAssertEqual(snapshot.graph.nodes.first {$0.id==expandedID}?.type,"gelu")
    }
    @MainActor func testOwnerProtectsExposedActivationBindingsAndExactRawValues()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),id=UUID().uuidString
        var source=CompositeExamples.parallelClassifier
        let index=try XCTUnwrap(source.definitions[0].body.nodes.firstIndex {$0.id=="activation-a"})
        source.definitions[0].body.nodes[index].type="prelu"
        source.definitions[0].body.nodes[index].config=[:]
        source.definitions[0].parameters.append(.init(.init(key:"slope",title:"Slope",kind:.number,defaultValue:.number(0.3),min:0,max:1),
            targets:[.init("activation-a","init")]))
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        let child=try ModelCanvasAdapter(owner:owner,path:["features"])
        child.selectNode("activation-a",extending:false)
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setActivation(nodeID:"activation-a",function:.relu),lease:owner.lease(path:["features"])))
        XCTAssertEqual(owner.document,before)
        XCTAssertNil(owner.applyDraftCommand(.setActivation(nodeID:"activation-a",function:.prelu,configuration:["num_parameters":.integer(1),"init":.number(0.5)]),lease:owner.lease(path:["features"])))
        XCTAssertEqual(owner.document,before)
        XCTAssertNotNil(owner.applyDraftCommand(.setActivation(nodeID:"activation-a",function:.prelu,configuration:["num_parameters":.integer(1),"init":.number(0.3)]),lease:owner.lease(path:["features"])),owner.error ?? "No owner error")
        XCTAssertNil(child.scope?.nodes.first {$0.id=="activation-a"}?.config["init"],"Resolved boundary values must not replace the exact authored raw field")
    }

    @MainActor func testAdapterUsesEffectiveBoundPReLUValuesAndPreservesRawOmission()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory),id=UUID().uuidString
        var source=CompositeExamples.parallelClassifier
        let index=try XCTUnwrap(source.definitions[0].body.nodes.firstIndex {$0.id=="activation-a"})
        source.definitions[0].body.nodes[index].type="prelu"
        source.definitions[0].body.nodes[index].config=[:]
        source.definitions[0].parameters.append(.init(.init(key:"slope",title:"Slope",kind:.number,defaultValue:.number(0.3),min:0,max:1),
            targets:[.init("activation-a","init")]))
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        let child=try ModelCanvasAdapter(owner:owner,path:["features"])
        child.selectNode("activation-a",extending:false)
        XCTAssertNil(child.scope?.nodes.first {$0.id=="activation-a"}?.config["init"],"Scope exposes exact authored raw values")
        XCTAssertNil(child.projectionError)
        let displayed=try XCTUnwrap(child.graph.nodes.first {$0.id=="activation-a"})
        XCTAssertEqual(displayed.config["init"],.number(0.3),"Canvas projection resolves the exposed boundary value")
        let effective=try ActivationConfiguration(componentID:displayed.type,configurationVersion:child.graph.schemaVersion,
            values:displayed.config,registry:child.primitiveRegistry)
        XCTAssertFalse(child.setActivation(.relu,nodeID:displayed.id,lease:owner.lease(path:["features"]),
            expectedType:displayed.type,expectedSourceConfig:displayed.config))
        var unauthorized=effective.values;unauthorized["init"] = .number(0.5)
        XCTAssertFalse(child.setActivation(.prelu,configuration:unauthorized,nodeID:displayed.id,lease:owner.lease(path:["features"]),
            expectedType:displayed.type,expectedSourceConfig:displayed.config))
        XCTAssertTrue(child.setActivation(.prelu,configuration:effective.values,nodeID:displayed.id,lease:owner.lease(path:["features"]),
            expectedType:displayed.type,expectedSourceConfig:displayed.config),child.statusMessage)
        XCTAssertNil(child.scope?.nodes.first {$0.id=="activation-a"}?.config["init"])
        XCTAssertEqual(child.graph.nodes.first {$0.id=="activation-a"}?.config["init"],.number(0.3))
    }

}
