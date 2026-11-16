import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCapturedOutputContextTests:XCTestCase {
    private let address=WorkflowNodeAddress(domain:.data,localID:"same")
    private func fixture()throws->(WorkflowAuthoringStore,OutputSourceProbe,OutputSnapshotProbe) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:address,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:.init(domain:.model,localID:"same"),payload:.modelPrimitive(type:"input",configurationVersion:2,config:["width":2]))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        _=owner.select(.init(nodeAddresses:[address]),lease:owner.lease())
        return(owner,OutputSourceProbe(owner),OutputSnapshotProbe(owner))
    }
    private func snapshot(_ owner:WorkflowAuthoringStore,address:WorkflowNodeAddress?=nil,graph:Bool=true,bindings:Bool=true,phase:WorkflowExecutionPhase = .preparing)->WorkflowCapturedDataOutput {
        .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address ?? self.address,runID:UUID(uuidString:"11111111-1111-1111-1111-111111111111")!,requestIdentity:String(repeating:"a",count:64),capturedRevision:7,bindingRevision:3,phase:phase,matchesCurrentConstruction:graph,matchesCurrentBindings:bindings,ports:["table":.object(["kind":.string("table"),"preview":.object(["rowCount":100,"partition":.string("training"),"columns":.array([.object(["name":.string(" x, "),"dtype":.string("float64"),"nullable":false])]),"rows":.array([.array([.number(1.5)])]),"previewTruncated":true])])])
    }
    func testCanvasCaptureLabelsRespectQualifiedIdentityStalenessAndRetirement() throws {
        let (owner,_,outputs)=try fixture()
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        canvas.outputProvider=outputs
        let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address])
        XCTAssertEqual(canvas.previewStatus(for:id),"Not evaluated")
        XCTAssertEqual(canvas.canvasEvidenceLabel(nodeID:id),"No captured output for this component.")
        outputs.value=snapshot(owner,phase:.completed)
        XCTAssertEqual(canvas.previewStatus(for:id),"Captured")
        XCTAssertEqual(canvas.canvasEvidenceLabel(nodeID:id),"Captured output · current sources not revalidated.")
        outputs.value=snapshot(owner,phase:.failed)
        XCTAssertEqual(canvas.previewStatus(for:id),"Partial capture")
        XCTAssertEqual(canvas.canvasEvidenceLabel(nodeID:id),"Partial captured output · inspect run status in Results.")
        outputs.value=snapshot(owner,graph:false,phase:.completed)
        XCTAssertEqual(canvas.previewStatus(for:id),"Stale")
        XCTAssertEqual(canvas.canvasEvidenceLabel(nodeID:id),"Captured output is stale · rerun to reflect current inputs.")
        outputs.value=snapshot(owner,address:.init(domain:.model,localID:"same"),phase:.completed)
        XCTAssertEqual(canvas.previewStatus(for:id),"Not evaluated")
        XCTAssertEqual(canvas.canvasEvidenceLabel(nodeID:id),"No captured output for this component.")
        outputs.value=snapshot(owner,phase:.completed)
        canvas.invalidate()
        XCTAssertEqual(canvas.previewStatus(for:id),"Not evaluated")
        XCTAssertEqual(canvas.canvasEvidenceLabel(nodeID:id),"No captured output for this component.")
    }
    func testCapturedTypedOutputBridgesWithoutSourceActionsAndRetainsRequestProvenance()throws {
        let(owner,sources,outputs)=try fixture();outputs.value=snapshot(owner)
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources,outputs:outputs)
        let result=try XCTUnwrap(context.outputs?["table"] as? [String:Any])
        let preview=try XCTUnwrap(result["preview"] as? [String:Any])
        XCTAssertEqual(preview["rowCount"] as? Int,100)
        XCTAssertEqual((preview["rows"] as? [[Double]])?.first,[1.5])
        XCTAssertTrue(context.outputStatus.lowercased().contains("partial"))
        XCTAssertTrue(context.outputStatus.lowercased().contains("not revalidated"))
        XCTAssertTrue(context.outputDisclosure?.contains(String(repeating:"a",count:64)) == true)
        XCTAssertEqual(sources.actions,0)
    }
    func testSameLocalIDWrongDomainAndForeignProviderCannotLeakOutput()throws {
        let(owner,sources,outputs)=try fixture()
        outputs.value=snapshot(owner,address:.init(domain:.model,localID:"same"))
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources,outputs:outputs)
        XCTAssertNil(context.outputs)
        let(_,_,foreign)=try fixture()
        XCTAssertThrowsError(try WorkflowDataNodeContext(owner:owner,address:address,sources:sources,outputs:foreign))
        _=owner.select(.init(),lease:owner.lease())
        outputs.value=snapshot(owner)
        XCTAssertNil(context.outputs)
    }
    func testStaleGraphAndBindingReasonsDoNotRelabelCapturedFailureAsComplete()throws {
        let(owner,sources,outputs)=try fixture()
        outputs.value=snapshot(owner,graph:false,bindings:false,phase:.failed)
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources,outputs:outputs)
        XCTAssertNotNil(context.outputs,"Historical sampled values remain inspectable.")
        let label=context.outputStatus.lowercased()
        XCTAssertTrue(label.contains("stale"));XCTAssertTrue(label.contains("binding"));XCTAssertTrue(label.contains("failed"))
        XCTAssertFalse(label.contains("completed"));XCTAssertNil(context.retainedRows)
        XCTAssertTrue(context.outputDisclosure?.contains("7") == true)
        XCTAssertEqual(sources.actions,0)
    }
    func testOutputProviderPublicationUpdatesContextWithoutChangingEditingAuthority()throws {
        let(owner,sources,outputs)=try fixture()
        let context=try WorkflowDataNodeContext(owner:owner,address:address,sources:sources,outputs:outputs)
        let identity=context.contextIdentity
        var notifications=0
        let observation=context.objectWillChange.sink {notifications+=1};defer {observation.cancel()}
        outputs.value=snapshot(owner,phase:.cancelled);outputs.changed.send()
        XCTAssertGreaterThan(notifications,0)
        XCTAssertEqual(context.contextIdentity,identity);XCTAssertTrue(context.isCurrent)
        XCTAssertTrue(context.outputStatus.lowercased().contains("cancelled"))
    }
    func testContextPreservesCapturePopulationAndNewerRunVersusRefreshOrdering() throws {
        let (owner, sources, outputs) = try fixture()
        let prefix = SourceInspectionResult(columns: ["prefix"], rows: [["earlier inspection"]], rowCount: 1,
            sourceSHA256: String(repeating: "b", count: 64),
            parsing: .init(validatedRows: 1, previewRows: 1, sampleOnly: true, options: CSVParsingConfiguration.legacyDefaults))
        sources.preview = prefix; sources.completedAt = Date(timeIntervalSince1970: 10)
        var captured = snapshot(owner, phase: .completed)
        captured.capturedAt = Date(timeIntervalSince1970: 20)
        captured.dataCompleteness = "Sampled working data · first records"
        outputs.value = captured
        let context = try WorkflowDataNodeContext(owner: owner, address: address, sources: sources, outputs: outputs)
        XCTAssertEqual(context.outputCapturedAt, captured.capturedAt)
        XCTAssertEqual(context.outputPopulationLabel, captured.dataCompleteness)
        XCTAssertTrue(context.outputStatus.contains("Sampled working data"))
        let output = try XCTUnwrap(context.outputs?["table"] as? [String: Any])
        let run = try XCTUnwrap(output["preview"] as? [String: Any])
        func presentation() -> TableInputOutputPresentation {
            let newer = context.outputCapturedAt.map { $0 > (context.sourcePreviewCompletedAt ?? .distantPast) } ?? false
            return .init(run: run, runIsCurrent: context.outputIsCurrent, preview: context.sourceInspectionPreview, preferNewerRun: newer)
        }
        XCTAssertEqual(presentation().label, "Run output")
        sources.completedAt = Date(timeIntervalSince1970: 30)
        XCTAssertEqual(presentation().label, "Preview", "A later explicit refresh must supersede the earlier run again.")
        XCTAssertEqual(sources.actions, 0)
    }

}

@MainActor private final class OutputSnapshotProbe:WorkflowDataOutputs {
    let projectID:String;let sessionID:UUID
    let changed=PassthroughSubject<Void,Never>()
    var changes:AnyPublisher<Void,Never> {changed.eraseToAnyPublisher()}
    var value:WorkflowCapturedDataOutput?
    init(_ owner:WorkflowAuthoringStore) {projectID=owner.projectID;sessionID=owner.sessionID}
    func snapshot(for address:WorkflowNodeAddress)->WorkflowCapturedDataOutput? {value}
}
@MainActor private final class OutputSourceProbe:WorkflowDataSources {
    let projectID:String;let sessionID:UUID
    var isBusy=false;var statusMessage="Recorded source schema";var actions=0
    var preview: SourceInspectionResult?
    var completedAt: Date?
    func sourceInspectionPreview(for address: WorkflowNodeAddress) -> SourceInspectionResult? { preview }
    func sourceInspectionCompletedAt(for address: WorkflowNodeAddress) -> Date? { completedAt }
    var changes:AnyPublisher<Void,Never> {Empty(completeImmediately:false).eraseToAnyPublisher()}
    init(_ owner:WorkflowAuthoringStore) {projectID=owner.projectID;sessionID=owner.sessionID}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] {[]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {"source.csv"}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {actions+=1;return false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {actions+=1;return false}
}
