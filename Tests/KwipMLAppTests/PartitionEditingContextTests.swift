import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class PartitionEditingContextTests:XCTestCase {
    func testAllCurrentPartitionStrategiesAndFormulaAreAtomicUndoableAndRejectStaleBusyAuthority()throws {
        for (op,strategy) in [("kwip.data.partition","random"),("kwip.data.partition","entity"),("kwip.data.partition","time"),("kwip.data.formula","formula")] {
            let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:directory)}
            let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let registry=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
            let initial=try XCTUnwrap(registry.operatorDefinition(id:op)).defaultConfig
            let address=WorkflowNodeAddress(domain:.data,localID:"partition")
            let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Chosen population",payload:.dataOperator(operatorID:op,config:initial))])
            let library=ProjectLibrary(directory:directory)
            _=try library.saveSource(.workflow(document),id:document.projectID)
            let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:registry)
            XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
            let sources=PartitionSourceProbe(owner:owner)
            let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
            var formula=FormulaDraft.defaults
            if case .array(var operations)=formula["operations"],case .object(var first)=operations[0] {first["expression"] = .string("1");operations[0] = .object(first);formula["operations"] = .array(operations)}
            let next:[String:JSONValue]
            switch strategy {
            case "formula":next=formula
            case "random":next=["strategy":.string("random"),"seed":43,"validationPercent":20,"testPercent":20]
            case "entity":next=["strategy":.string("entity"),"seed":42,"validationPercent":20,"testPercent":20,"groupBy":.array([.string(" entity, ")])]
            default:next=["strategy":.string("time"),"timeColumn":.string("time"),"timeEncoding":.string("iso8601Offset"),"trainingEnd":.string("2026-01-01T00:00:00Z"),"validationEnd":.string("2026-02-01T00:00:00Z"),"embargoMilliseconds":1000,"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")]
            }
            XCTAssertTrue(context.applyTableConfiguration(next,expected:initial),context.statusMessage)
            XCTAssertEqual(context.node?.config,next)
            XCTAssertEqual(sources.actions,0)
            var invalid=next;invalid[strategy == "time" ? "embargoMilliseconds":"testPercent"] = .integer(-1)
            XCTAssertFalse(context.applyTableConfiguration(invalid,expected:next))
            XCTAssertEqual(context.node?.config,next)
            owner.setExecutionBusy(true)
            XCTAssertFalse(context.applyTableConfiguration(next,expected:next))
            owner.setExecutionBusy(false)
            XCTAssertFalse(context.applyTableConfiguration(next,expected:initial))
            XCTAssertTrue(owner.undo(lease:owner.lease()))
            XCTAssertFalse(context.isCurrent)
            XCTAssertFalse(context.applyTableConfiguration(next,expected:initial))
            let restored=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
            XCTAssertEqual(restored.node?.config,initial,"One undo restores all partition fields.")
            XCTAssertEqual(sources.actions,0,"Rendering and applying recorded schema cannot acquire sources.")
        }
    }
}

@MainActor private final class PartitionSourceProbe:WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage="Recorded columns"
    var actions=0
    var changes:AnyPublisher<Void,Never> {Empty(completeImmediately:false).eraseToAnyPublisher()}
    init(owner:WorkflowAuthoringStore) {projectID=owner.projectID;sessionID=owner.sessionID}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] {[.init(name:" entity, ",dtype:"string",nullable:false),.init(name:"time",dtype:"string",nullable:false)]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {nil}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {actions+=1;return false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {actions+=1;return false}
}
