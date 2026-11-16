import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowDataNodeContextTests:XCTestCase {
    private var latestLibrary:ProjectLibrary?
    private let data=WorkflowNodeAddress(domain:.data,localID:"shared")
    private let model=WorkflowNodeAddress(domain:.model,localID:"shared")
    private func make(operatorID: String = "kwip.data.sourceCSV")throws->(WorkflowAuthoringStore,WorkflowSourceProbe) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        latestLibrary=library
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:data,name:"Measurements",payload:.dataOperator(operatorID:operatorID,config:operatorID == DataFlowSourceOperators.fileInput ? FileInputConfiguration.newDefaults : ["separator":.string(",")])),
            .init(address:model,name:"Model",payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":false]))])
        _=try library.saveSource(.workflow(doc),id:doc.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[data]),lease:owner.lease()))
        return(owner,WorkflowSourceProbe(owner:owner))
    }
    func testFileInputDraftAppliesAtomicallyAndExplicitSourceActionsUseCurrentLease() throws {
        let (owner, sources) = try make(operatorID: DataFlowSourceOperators.fileInput)
        let context = try WorkflowDataNodeContext(owner: owner, address: data, sources: sources)
        let original = try XCTUnwrap(context.node?.config)
        var next = original; next["sourceMode"] = .string("directory"); next["extensions"] = .array([.string("txt")])
        XCTAssertTrue(context.configure(try FileInputConfiguration.validate(next), expected: original))
        XCTAssertEqual(sources.actions, [], "Applying controls cannot silently read the original folder.")
        XCTAssertTrue(context.chooseSource()); XCTAssertTrue(context.refreshSource())
        XCTAssertEqual(sources.actions, ["choose", "inspect"])
        XCTAssertEqual(context.node?.config["sourceMode"], .string("directory"))
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.chooseSource()); XCTAssertFalse(context.refreshSource())
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.undo(lease: owner.lease()))
        XCTAssertFalse(context.isCurrent)
        XCTAssertFalse(context.configure(next, expected: original))
        XCTAssertEqual(owner.document.nodes[0].payload, .dataOperator(operatorID: DataFlowSourceOperators.fileInput, config: original))
    }
    func testOwnCommitsRenewLeaseWithoutChangingFieldIdentityOrOtherPendingFields()throws {
        let(owner,sources)=try make(),context=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        let identity=context.contextIdentity,origin=context.numericOrigin(key:"uncommitted")
        context.setNumericPending(origin:origin,key:"uncommitted",pending:true)
        XCTAssertTrue(owner.hasPendingDataEdits)
        let config=context.node!.config
        XCTAssertTrue(context.configure(["separator":.string("\t")],expected:config))
        XCTAssertTrue(context.isCurrent);XCTAssertEqual(context.contextIdentity,identity)
        XCTAssertEqual(context.numericOrigin(key:"uncommitted"),origin)
        XCTAssertTrue(owner.hasPendingDataEdits,"Committing one value must not release another field's draft.")
        XCTAssertTrue(context.rename("Renamed data"))
        XCTAssertEqual(owner.document.nodes[0].name,"Renamed data")
        XCTAssertEqual(owner.document.nodes[1].name,"Model")
        XCTAssertFalse(context.configure([:],expected:config))
        XCTAssertTrue(context.isCurrent);XCTAssertTrue(owner.hasPendingDataEdits)
        XCTAssertFalse(context.configure(["unsupportedSourceField":.string("rejected")],expected:context.node!.config))
        XCTAssertTrue(context.isCurrent,"A rejected owner transaction must leave this editor usable.")
        XCTAssertTrue(owner.hasPendingDataEdits)
        context.setNumericPending(origin:origin,key:"uncommitted",pending:false)
        XCTAssertFalse(owner.hasPendingDataEdits)
    }
    func testSelectionAwayBackExternalEditAndUndoRetireCapturedAuthoritySynchronously()throws {
        let(owner,sources)=try make(),context=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        let config=context.node!.config
        XCTAssertTrue(owner.select(.init(nodeAddresses:[model]),lease:owner.lease()))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[data]),lease:owner.lease()))
        XCTAssertFalse(context.isCurrent)
        XCTAssertFalse(context.configure([:],expected:config))
        let second=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        XCTAssertNotNil(owner.apply(.rename(address:model,name:"External change"),lease:owner.lease()))
        XCTAssertFalse(second.isCurrent)
        let third=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(third.isCurrent)
    }
    func testSourceProtocolIsReadOnlyDuringRenderingAndActionsRequireExactLiveSession()throws {
        let(owner,sources)=try make(),context=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        XCTAssertEqual(context.upstreamSchema.map(\.name),[" x, ","target"])
        _=context.sourceDisplayName;_=context.outputStatus;_=context.outputs
        XCTAssertEqual(sources.actions,[])
        XCTAssertTrue(context.chooseSource());XCTAssertEqual(sources.actions,["choose"])
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.refreshSource());XCTAssertFalse(context.chooseSource())
        owner.setExecutionBusy(false)
        XCTAssertTrue(context.setSeparator("\t",expected:context.node!.config))
        XCTAssertEqual(sources.actions,["choose","inspect"])
        XCTAssertEqual(sources.lastLease,owner.lease())
        owner.invalidate()
        XCTAssertFalse(context.refreshSource());XCTAssertFalse(context.isCurrent)
        XCTAssertFalse(owner.hasPendingDataEdits)
    }
    func testFailedPersistencePreservesConfigurationContextAndPendingDraftThenAllowsRetry()throws {
        let(owner,sources)=try make(),context=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        let original=context.node!.config,origin=context.numericOrigin(key:"other")
        context.setNumericPending(origin:origin,key:"other",pending:true)
        let url=try XCTUnwrap(latestLibrary).documentURL(id:owner.projectID)
        let bytes=try Data(contentsOf:url)
        try FileManager.default.removeItem(at:url)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertFalse(context.configure(["separator":.string("\t")],expected:original))
        XCTAssertEqual(context.node?.config,original)
        XCTAssertTrue(context.isCurrent);XCTAssertTrue(owner.hasPendingDataEdits)
        XCTAssertNotNil(owner.error)
        try FileManager.default.removeItem(at:url);try bytes.write(to:url,options:.atomic)
        XCTAssertTrue(context.configure(["separator":.string("\t")],expected:original))
        XCTAssertTrue(context.isCurrent);XCTAssertTrue(owner.hasPendingDataEdits)
    }
    func testAggregateDraftAppliesAtomicallyAndUndoRetiresAuthority()throws {
        let(owner,sources)=try make()
        let address=WorkflowNodeAddress(domain:.data,localID:"aggregate")
        let initial=try XCTUnwrap(owner.dataRegistry.operatorDefinition(id:"kwip.data.aggregate")).defaultConfig
        XCTAssertNotNil(owner.apply(.insert(.init(address:address,name:"",payload:.dataOperator(operatorID:"kwip.data.aggregate",config:initial))),lease:owner.lease()))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
        var config=initial
        config["groupBy"] = .array([.string(" x, ")])
        config["aggregations"] = .array([.object(["op":.string("countRows"),"output":.string("My count")])])
        config["nullKeys"] = .string("drop");config["maxRows"] = .integer(123)
        XCTAssertTrue(context.applyTableConfiguration(config,expected:initial),context.statusMessage)
        XCTAssertEqual(context.node?.config,config)
        var invalid=config;invalid["aggregations"] = .array([])
        XCTAssertFalse(context.applyTableConfiguration(invalid,expected:config))
        XCTAssertEqual(context.node?.config,config)
        XCTAssertFalse(context.applyTableConfiguration(initial,expected:initial))
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.applyTableConfiguration(config,expected:config))
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertFalse(context.isCurrent)
        XCTAssertFalse(context.applyTableConfiguration(config,expected:initial))
        let restored=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
        XCTAssertEqual(restored.node?.config,initial)
        XCTAssertEqual(sources.actions,[])
    }
    func testJoinDraftAppliesAtomicallyAndUndoRetiresAuthority()throws {
        let(owner,sources)=try make()
        let join=WorkflowNodeAddress(domain:.data,localID:"join")
        let definition=try XCTUnwrap(owner.dataRegistry.operatorDefinition(id:"kwip.data.join"))
        let initial=definition.defaultConfig
        XCTAssertNotNil(owner.apply(.insert(.init(address:join,name:"",payload:.dataOperator(operatorID:"kwip.data.join",config:initial))),lease:owner.lease()))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[join]),lease:owner.lease()))
        let context=try WorkflowDataNodeContext(owner:owner,address:join,sources:sources)
        XCTAssertEqual(context.upstreamSchema(forInput:"left").map(\.name),[" x, "])
        XCTAssertEqual(context.upstreamSchema(forInput:"right").map(\.name),["target"])
        var config=initial
        config["leftKeys"] = .array([.string(" x, ")]);config["rightKeys"] = .array([.string("target")])
        config["mode"] = .string("full");config["maxRows"] = .integer(123)
        XCTAssertTrue(context.applyTableConfiguration(config,expected:initial),context.statusMessage)
        XCTAssertEqual(context.node?.config,config)
        var invalid=config;invalid["rightKeys"] = .array([])
        XCTAssertFalse(context.applyTableConfiguration(invalid,expected:config))
        XCTAssertEqual(context.node?.config,config)
        XCTAssertFalse(context.applyTableConfiguration(initial,expected:initial))
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.applyTableConfiguration(config,expected:config))
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(context.isCurrent)
        let restored=try WorkflowDataNodeContext(owner:owner,address:join,sources:sources)
        XCTAssertEqual(restored.node?.config,initial,"One undo restores the entire applied draft.")
        XCTAssertEqual(sources.actions,[],"Schema guidance and draft apply must not read a source.")
    }
    func testForeignSourceSessionIsRejectedAtConstruction()throws {
        let(owner,_)=try make(),(_,foreignSources)=try make()
        XCTAssertThrowsError(try WorkflowDataNodeContext(owner:owner,address:data,sources:foreignSources))
        XCTAssertFalse(owner.hasPendingDataEdits)
    }
    func testLatestExplicitSourceActionStatusReplacesEarlierSavedMessage()throws {
        let(owner,sources)=try make(),context=try WorkflowDataNodeContext(owner:owner,address:data,sources:sources)
        XCTAssertTrue(context.rename("Saved first"))
        sources.actionSucceeds=false
        XCTAssertFalse(context.chooseSource())
        XCTAssertEqual(context.statusMessage,"Source permission denied")
    }
    func testGateRejectsForeignOrStaleRegistrationAndContextCleanupIsIsolated()throws {
        let(owner,_)=try make(),first=UUID(),second=UUID(),lease=owner.lease()
        XCTAssertTrue(owner.setDataEditPending(contextID:first,address:data,key:"one",pending:true,lease:lease))
        XCTAssertTrue(owner.setDataEditPending(contextID:second,address:data,key:"two",pending:true,lease:lease))
        owner.clearDataEdits(contextID:first);XCTAssertTrue(owner.hasPendingDataEdits)
        XCTAssertFalse(owner.setDataEditPending(contextID:first,address:model,key:"bad",pending:true,lease:lease))
        XCTAssertNotNil(owner.apply(.rename(address:data,name:"Advance"),lease:lease))
        XCTAssertFalse(owner.setDataEditPending(contextID:first,address:data,key:"late",pending:true,lease:lease))
        owner.clearDataEdits(contextID:second);XCTAssertFalse(owner.hasPendingDataEdits)
    }
}

@MainActor private final class WorkflowSourceProbe:WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage="Recorded source guidance"
    var changes:AnyPublisher<Void,Never> {Empty(completeImmediately:false).eraseToAnyPublisher()}
    var actionSucceeds=true
    var actions:[String]=[]
    var lastLease:WorkflowEditorLease?
    init(owner:WorkflowAuthoringStore) {projectID=owner.projectID;sessionID=owner.sessionID}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] {[.init(name:" x, ",dtype:"string"),.init(name:"target",dtype:"string")]}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument,inputPort:String)->[DataColumnSchema] {inputPort == "left" ? [.init(name:" x, ",dtype:"string")]:inputPort == "right" ? [.init(name:"target",dtype:"string")]:[]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {"measurements.csv"}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {actions.append("choose");lastLease=lease;statusMessage=actionSucceeds ? "Source chosen" : "Source permission denied";return actionSucceeds}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {actions.append("inspect");lastLease=lease;return true}
}
