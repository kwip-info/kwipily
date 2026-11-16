import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class Data02ColumnOwnerTests: XCTestCase {
    func testColumnsApplyIsAtomicKeepsPendingAndUndoRestoresOriginal() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:directory) }
        let library=ProjectLibrary(directory:directory)
        let address=WorkflowNodeAddress(domain:.data,localID:"convert")
        var initial=try ColumnsConfigurationDraft(configuration:ColumnsConfigurationDraft.defaults)
        initial.add(source:"amount");initial.update("amount") {$0.targetType="float64"};initial.missingPolicy="error"
        let original=initial.configuration
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Convert amounts",payload:.dataOperator(operatorID:"kwip.data.columns",config:original))])
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let sources=Data02ColumnSources(owner:owner)
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
        let origin=context.numericOrigin(key:"rows")
        context.setNumericPending(origin:origin,key:"rows",pending:true)
        var draft=try ColumnsConfigurationDraft(configuration:original)
        draft.update("amount") {$0.targetType="int64"};draft.missingPolicy="preserve"
        let changed=try draft.validatedConfiguration(schema:context.upstreamSchema)
        XCTAssertTrue(context.applyTableConfiguration(changed,expected:original),context.statusMessage)
        XCTAssertEqual(context.node?.config,changed)
        XCTAssertTrue(owner.hasPendingInspectorEdits,"Applying one transaction does not silently clear another pending editor registration.")
        context.setNumericPending(origin:origin,key:"rows",pending:false)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        var invalid=changed;invalid["rows"] = .array([.object(["source":.string("absent"),"included":.bool(true),"outputName":.string("amount"),"targetType":.string("float64")])])
        XCTAssertFalse(context.applyTableConfiguration(invalid,expected:changed))
        XCTAssertEqual(context.node?.config,changed)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(context.isCurrent)
        let restored=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources)
        XCTAssertEqual(restored.node?.config,original)
        XCTAssertFalse(restored.applyTableConfiguration(changed,expected:changed))
    }
}

@MainActor private final class Data02ColumnSources: WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage="Recorded source"
    var changes:AnyPublisher<Void,Never> { Empty(completeImmediately:false).eraseToAnyPublisher() }
    init(owner:WorkflowAuthoringStore) { projectID=owner.projectID; sessionID=owner.sessionID }
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] { [.init(name:"amount",dtype:"string")] }
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {nil}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
}
