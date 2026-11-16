import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCanvasAdapterTests: XCTestCase {
    @MainActor func testEstimatorSummariesDescribeTheAuthoredOperation() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        for (operation,expected) in [("fit","Fit a model on prepared data"),("predict","Predict with a frozen model"),("evaluate","Evaluate with a frozen model")] {
            let address=WorkflowNodeAddress(domain:.control,localID:"summary-"+operation)
            var configuration=PrebuiltModelConfiguration.defaults("logistic")
            configuration["operation"] = .string(operation)
            XCTAssertNotNil(owner.applyBatch([.insert(.init(address:address,payload:.estimator(config:configuration)))],lease:owner.lease()))
            let id=try XCTUnwrap(adapter.projection.displayIDByAddress[address])
            let node=try XCTUnwrap(adapter.graph.nodes.first {$0.id==id})
            XCTAssertEqual(adapter.canvasSummary(node),expected)
        }
    }
    @MainActor func testBusyExecutionAllowsReadOnlySelectionButNeverAuthoring() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let source=try XCTUnwrap(adapter.projection.displayIDByAddress[data])
        let modelID=try XCTUnwrap(adapter.projection.displayIDByAddress[model])
        let document=owner.document,lease=owner.lease()
        owner.setExecutionBusy(true)
        adapter.selectNode(source,extending:false)
        XCTAssertEqual(owner.selection.nodeAddresses,[data])
        adapter.selectNode(modelID,extending:true)
        XCTAssertEqual(owner.selection.nodeAddresses,[data,model])
        let edge=try XCTUnwrap(adapter.projection.displayIDByEdgeID["unknown-wire"])
        adapter.selectEdge(edge)
        XCTAssertEqual(owner.selection.edgeIDs,["unknown-wire"])
        adapter.clearSelection()
        XCTAssertTrue(owner.selection.nodeAddresses.isEmpty && owner.selection.edgeIDs.isEmpty)
        XCTAssertTrue(owner.isCurrent(lease),"Inspection must not invalidate the captured run lease")
        adapter.moveNodes(ids:[source],dx:12,dy:4);adapter.deleteSelection()
        XCTAssertFalse(adapter.apply(.rename(address:data,name:"forbidden"),lease:lease))
        XCTAssertEqual(owner.document,document)
        adapter.invalidate();adapter.selectNode(source,extending:false)
        XCTAssertTrue(owner.selection.nodeAddresses.isEmpty,"Retired adapters remain unavailable")
    }
    private let data=WorkflowNodeAddress(domain:.data,localID:"same")
    private let model=WorkflowNodeAddress(domain:.model,localID:"same")
    @MainActor private func setup() throws -> (URL,WorkflowAuthoringStore,WorkflowCanvasAdapter) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:root)
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:data,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:model,payload:.modelPrimitive(type:"input",configurationVersion:1,config:["features":2])),
            .init(address:.init(domain:.model,localID:"linear"),payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true])),
            .init(address:.init(domain:.data,localID:"unknown"),payload:.dataOperator(operatorID:"future.unknown",config:[:]))
        ],edges:[.init(id:"unknown-wire",from:.init(address:.init(domain:.data,localID:"unknown"),port:"saved-out"),to:.init(address:data,port:"unavailable"),kind:.tensor)])
        _=try library.saveSource(.workflow(doc))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),operators=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:models,dataRegistry:operators)
        return(root,owner,try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:operators))
    }
    @MainActor func testProjectionPreservesQualifiedIdentityUnknownPortsAndCachesSelectionOnlyChanges() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let dataID=try XCTUnwrap(adapter.projection.displayIDByAddress[data]),modelID=try XCTUnwrap(adapter.projection.displayIDByAddress[model])
        XCTAssertNotEqual(dataID,modelID)
        XCTAssertEqual(adapter.canvasIconType(nodeID:dataID),"kwip.data.sourceCSV")
        XCTAssertEqual(adapter.canvasIconType(nodeID:modelID),"input")
        let epoch=adapter.projectionRevision
        adapter.selectNode(dataID,extending:false);adapter.selectNode(modelID,extending:true)
        XCTAssertEqual(owner.selection.nodeAddresses,[data,model])
        XCTAssertEqual(adapter.selectedNodeIDs,[dataID,modelID]);XCTAssertEqual(adapter.projectionRevision,epoch)
        let unknown=try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain:.data,localID:"unknown")])
        let descriptor=adapter.canvasPortDescriptor(nodeID:unknown,port:"saved-out",direction:.output)
        XCTAssertEqual(descriptor.kind,.unknown);XCTAssertTrue(descriptor.measured.isEmpty)
        XCTAssertTrue(adapter.validation.widths.isEmpty)
        XCTAssertFalse(adapter.validation.issues.isEmpty)
        adapter.moveNodes(ids:[dataID],dx:17,dy:9)
        XCTAssertNotEqual(adapter.projectionRevision,epoch)
        XCTAssertEqual(owner.document.nodes.first{$0.address==data}?.x,17)
        XCTAssertEqual(owner.document.nodes.first{$0.address==model}?.x,0)
    }
    @MainActor func testTypedConnectionsHistoryAndStaleCallbacksUseOwnerTransactions() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let input=try XCTUnwrap(adapter.projection.displayIDByAddress[model]),csv=try XCTUnwrap(adapter.projection.displayIDByAddress[data])
        let linearID=try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain:.model,localID:"linear")])
        let linear=try XCTUnwrap(adapter.graph.nodes.first{$0.id==linearID})
        XCTAssertFalse(adapter.canConnect(source:csv,sourcePort:"table",target:linear,port:"in"))
        XCTAssertTrue(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
        let token=owner.lease()
        adapter.connect(source:input,sourcePort:"out",target:linearID,targetPort:"in")
        XCTAssertEqual(owner.document.edges.last?.from.address,model)
        XCTAssertFalse(adapter.apply(.rename(address:data,name:"Late"),lease:token))
        XCTAssertEqual(owner.document.nodes.first{$0.address==data}?.name,"")
        XCTAssertTrue(adapter.canUndo);adapter.undo()
        XCTAssertEqual(owner.document.edges.count,1)
        adapter.redo();XCTAssertEqual(owner.document.edges.count,2)
        owner.setExecutionBusy(true)
        let before=owner.document
        adapter.deleteSelection();adapter.moveNodes(ids:[csv],dx:4,dy:4)
        XCTAssertEqual(owner.document,before)
        adapter.invalidate();owner.setExecutionBusy(false)
        XCTAssertFalse(adapter.apply(.rename(address:data,name:"Retired"),lease:owner.lease()))
    }
    @MainActor func testConnectionHoverReusesValidationAndInvalidatesOnEditsUndoAndBusy() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let input=try XCTUnwrap(adapter.projection.displayIDByAddress[model])
        let csv=try XCTUnwrap(adapter.projection.displayIDByAddress[data])
        let linearID=try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain:.model,localID:"linear")])
        let linear=try XCTUnwrap(adapter.graph.nodes.first{$0.id==linearID})
        let before=owner.document
        for _ in 0..<200 {
            XCTAssertTrue(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
            XCTAssertFalse(adapter.canConnect(source:csv,sourcePort:"table",target:linear,port:"in"))
        }
        XCTAssertEqual(adapter.connectionValidationCount,2)
        XCTAssertEqual(owner.document,before)
        adapter.selectNode(input,extending:false)
        XCTAssertTrue(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
        XCTAssertEqual(adapter.connectionValidationCount,2)
        owner.setExecutionBusy(true)
        XCTAssertFalse(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
        owner.setExecutionBusy(false)
        adapter.connect(source:input,sourcePort:"out",target:linearID,targetPort:"in")
        XCTAssertFalse(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
        XCTAssertEqual(adapter.connectionValidationCount,3)
        adapter.undo()
        XCTAssertTrue(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
        XCTAssertEqual(adapter.connectionValidationCount,4)
        adapter.invalidate()
        XCTAssertFalse(adapter.canConnect(source:input,sourcePort:"out",target:linear,port:"in"))
    }
    @MainActor func testCatalogUsesExplicitPayloadsAndNeverDisplayTypeOrWireInsertion() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let entry=try XCTUnwrap(adapter.creationEntries.first {if case .modelPrimitive("linear",2,_)=$0.payload{return true};return false})
        XCTAssertNotNil(adapter.canvasCatalog.component(type:entry.id))
        XCTAssertTrue(adapter.canvasInsertionTypes(edgeID:"anything").isEmpty)
        let oldCount=owner.document.nodes.count
        adapter.addNode(type:adapter.graph.nodes[0].type,x:50,y:60)
        XCTAssertEqual(owner.document.nodes.count,oldCount)
        adapter.addNode(type:entry.id,x:50,y:60)
        XCTAssertEqual(owner.document.nodes.count,oldCount+1)
        XCTAssertEqual(owner.document.nodes.last?.payload,entry.payload)
        XCTAssertEqual(owner.document.nodes.last?.address.domain,.model)
        XCTAssertEqual(owner.document.nodes.last?.x,50)
        let count=owner.document.nodes.count
        adapter.insertNode(type:entry.id,edgeID:adapter.graph.edges[0].id,x:0,y:0)
        XCTAssertEqual(owner.document.nodes.count,count)
    }
    @MainActor func testPersistedDraftRemainsReachableOnlyThroughHostAndExactRoot() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let model=try ModelBlockEditing.beginDraft(.init(projectID:UUID().uuidString,source:CompositeExamples.parallelClassifier),path:["features"])
        let workflow=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        _=try library.saveSource(.workflow(workflow))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),operators=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:workflow.projectID,modelRegistry:models,dataRegistry:operators)
        let adapter=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:operators)
        let address=WorkflowNodeAddress(domain:.model,localID:"features")
        let display=try XCTUnwrap(adapter.projection.displayIDByAddress[address])
        XCTAssertTrue(adapter.isBusy);XCTAssertFalse(adapter.canOpenBlock(display))
        let recovery=try XCTUnwrap(adapter.savedDraftRecovery)
        XCTAssertEqual(recovery.address,address)
        XCTAssertFalse(adapter.openSavedDraft(recovery),"No child host is installed yet.")
        var requested:WorkflowNodeAddress?,captured:WorkflowEditorLease?
        adapter.onOpenBlock={requested=$0;captured=$1}
        XCTAssertTrue(adapter.canOpenBlock(display));adapter.openBlock(display)
        XCTAssertEqual(requested,address);XCTAssertEqual(captured,owner.lease())
        XCTAssertEqual(owner.document,workflow)
        XCTAssertTrue(adapter.openSavedDraft(recovery))
        XCTAssertEqual(requested,address);XCTAssertEqual(captured,owner.lease())
        owner.setExecutionBusy(true)
        XCTAssertFalse(adapter.openSavedDraft(recovery),"Recovery must not bypass execution ownership.")
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.discardDraft(lease:owner.blockLease(root:address)))
        XCTAssertNil(adapter.savedDraftRecovery)
        requested=nil
        XCTAssertFalse(adapter.openSavedDraft(recovery),"A captured banner cannot reopen a discarded draft.")
        XCTAssertNil(requested)
        owner.setExecutionBusy(true);XCTAssertFalse(adapter.canOpenBlock(display))
        owner.setExecutionBusy(false);adapter.invalidate();XCTAssertFalse(adapter.canOpenBlock(display))
    }

    @MainActor func testEveryCreationEntryDeclaresTheSamePortsAsItsPlacedProjection() throws {
        let(root,_,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        for entry in adapter.creationEntries {
            let id="placed-" + UUID().uuidString
            let address=WorkflowNodeAddress(domain:entry.payload.domain,localID:id)
            var payload = entry.payload
            if case .initializationRecipe(let type, let version, _) = payload {
                // Placement resolves this contextual template from the connected model.
                // This static port test supplies a valid rule rather than storing empty rules.
                let configuration = try InitializationConfiguration(values: [
                    "seed": 42,
                    "rules": .array([.object([
                        "owner": .object(["node": .string("linear"), "parameter": .string("weight")]),
                        "algorithm": .string("normal"),
                        "settings": .object(["mean": 0, "std": 0.02])
                    ])])
                ])
                payload = .initializationRecipe(type: type, configurationVersion: version, config: configuration.values)
            }
            var source=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,payload:payload)])
            // Exercise every current palette payload at the current authoring contract.
            source.schemaVersion=23
            source.definitions = entry.definition.map { [$0] } ?? []
            let projected=try WorkflowCanvasProjector.project(source,modelRegistry:adapter.modelRegistry,dataRegistry:adapter.dataRegistry,inlineTrainControls:true)
            let node=try XCTUnwrap(projected.graph.nodes.first)
            let component=try XCTUnwrap(projected.registry.component(type:node.type))
            XCTAssertEqual(entry.component.inputs,component.inputs,"Creation inputs differ for \(entry.iconType)")
            XCTAssertEqual(entry.component.outputs,component.outputs,"Creation outputs differ for \(entry.iconType)")
            if case .modelPrimitive("output",_,_)=entry.payload {
                XCTAssertEqual(entry.component.outputs,["definition"],"The authoring dependency port must be visible during palette placement")
            }
        }
    }

    @MainActor func testInlineTrainingRemovesLegacyRecipePlacement() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let before=owner.document
        XCTAssertFalse(adapter.creationEntries.contains { ["dataset_source","data_loader"].contains($0.iconType) })
        let train=try XCTUnwrap(adapter.creationEntries.first { $0.iconType == "workflow.train" })
        XCTAssertFalse(train.component.inputs.contains("dataset"))
        XCTAssertFalse(train.component.inputs.contains("loader"))
        XCTAssertEqual(owner.document,before,"Catalog discovery must not mutate saved source.")
    }

    @MainActor func testAdapterRejectsRegistriesDifferentFromItsAuthoritativeOwner() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let original=owner.document
        var changedModel=adapter.modelRegistry
        changedModel.components[0].title="Different model registry"
        XCTAssertThrowsError(try WorkflowCanvasAdapter(owner:owner,modelRegistry:changedModel,dataRegistry:adapter.dataRegistry))
        changedModel=adapter.modelRegistry
        changedModel.components[0].outputs.append("invented-output")
        XCTAssertThrowsError(try WorkflowCanvasAdapter(owner:owner,modelRegistry:changedModel,dataRegistry:adapter.dataRegistry))
        var changedData=adapter.dataRegistry
        changedData.operators[0].title="Different data registry"
        XCTAssertThrowsError(try WorkflowCanvasAdapter(owner:owner,modelRegistry:adapter.modelRegistry,dataRegistry:changedData))
        changedData=adapter.dataRegistry
        changedData.operators[0].outputs.removeAll()
        XCTAssertThrowsError(try WorkflowCanvasAdapter(owner:owner,modelRegistry:adapter.modelRegistry,dataRegistry:changedData))
        XCTAssertNoThrow(try WorkflowCanvasAdapter(owner:owner,modelRegistry:adapter.modelRegistry,dataRegistry:adapter.dataRegistry))
        XCTAssertEqual(owner.document,original)
        XCTAssertFalse(owner.canUndo)
    }

    @MainActor func testSeparablePaletteInsertsPinnedEditableBlockAtomicallyWithUndo() throws {
        let(root,owner,adapter)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let entry=try XCTUnwrap(adapter.creationEntries.first { $0.component.title == "Separable Conv2d" })
        XCTAssertEqual(entry.component.category,"Layers")
        XCTAssertEqual(entry.component.fields.count,11)
        XCTAssertNotNil(entry.definition)
        let original=owner.document
        adapter.addNode(type:entry.id,x:450,y:220)
        XCTAssertEqual(owner.document.nodes.count,original.nodes.count+1)
        XCTAssertEqual(owner.document.definitions.count,original.definitions.count+1)
        let address=try XCTUnwrap(owner.selection.nodeAddresses.first)
        let node=try XCTUnwrap(owner.document.nodes.first{$0.address==address})
        XCTAssertEqual(node.x,450);XCTAssertEqual(node.y,220)
        XCTAssertEqual(node.name,"Separable Conv2d")
        guard case .modelBlock(let pin,_)=node.payload else {return XCTFail("Expected Block")}
        XCTAssertEqual(owner.document.definitions.first{$0.reference==pin}?.body.nodes.count,2)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,original.nodes)
        XCTAssertEqual(owner.document.definitions,original.definitions)
        owner.setExecutionBusy(true)
        XCTAssertFalse(adapter.canAddType(entry.id))
        adapter.addNode(type:entry.id,x:0,y:0)
        XCTAssertEqual(owner.document.nodes,original.nodes)
    }

}
