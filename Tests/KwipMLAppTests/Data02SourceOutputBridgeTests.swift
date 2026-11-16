import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class Data02SourceOutputBridgeTests: XCTestCase {
    func testOutputDestinationRoutesExactOwnerLeaseWithoutRenderingActions() throws {
        let (owner,sources,address)=try fixture(output:true)
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
        XCTAssertEqual(context.outputDestinationSummary,"Test destination on this Mac")
        XCTAssertEqual(sources.calls,0)
        XCTAssertTrue(context.chooseOutputDestination())
        XCTAssertEqual(sources.calls,1); XCTAssertEqual(sources.receivedLease,owner.lease())
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.chooseOutputDestination()); XCTAssertEqual(sources.calls,1)
        owner.setExecutionBusy(false)
        owner.invalidate()
        XCTAssertFalse(context.chooseOutputDestination()); XCTAssertEqual(sources.calls,1)
    }
    func testSourceNodeCannotInvokeOutputPicker() throws {
        let (owner,sources,address)=try fixture(output:false)
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
        XCTAssertFalse(context.chooseOutputDestination())
        XCTAssertEqual(sources.calls,0)
        XCTAssertTrue(context.outputDestinationSummary.contains("Select a table output"))
    }
    private func fixture(output:Bool) throws -> (WorkflowAuthoringStore,Data02DestinationSources,WorkflowNodeAddress) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at:directory) }
        let library=ProjectLibrary(directory:directory)
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let id=output ? "kwip.data.outputDelimited" : "kwip.data.sourceCSV"
        let config=try XCTUnwrap(registry.operatorDefinition(id:id)).defaultConfig
        let address=WorkflowNodeAddress(domain:.data,localID:"node")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Named component",payload:.dataOperator(operatorID:id,config:config))])
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        return (owner,Data02DestinationSources(owner:owner),address)
    }
}

@MainActor private final class Data02DestinationSources: WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage="Output destination ready"
    var changes:AnyPublisher<Void,Never> { Empty(completeImmediately:false).eraseToAnyPublisher() }
    var calls=0
    var receivedLease:WorkflowEditorLease?
    init(owner:WorkflowAuthoringStore) {projectID=owner.projectID; sessionID=owner.sessionID}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] {[]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {nil}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
    func outputDestinationSummary(for address:WorkflowNodeAddress)->String {"Test destination on this Mac"}
    func chooseOutputDestination(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {calls+=1; receivedLease=lease; return true}
}
