import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelOutputPreflightDiagnosticsTests: XCTestCase {
    @MainActor func testMismatchAppearsOnBoundaryAndSelectedDiagnosticsThenClearsAfterRepair() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = ProjectLibrary(directory: root)
        let document = try ModelOutputPracticeExample.document(device: "cpu")
        _ = try library.saveSource(.workflow(document))
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models = try ComponentRegistry.builtIn()
        let data = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: models, dataRegistry: data)
        let canvas = try WorkflowCanvasAdapter(owner: owner, modelRegistry: models, dataRegistry: data)
        defer { canvas.invalidate() }
        await settle(canvas)
        let id = try XCTUnwrap(canvas.projection.displayIDByAddress[ModelOutputPracticeExample.address])
        var config = try ModelOutputConfiguration([:]); config.checkExpectedOutput = true
        config.expectedSampleRank = "exact"; config.expectedSampleDimensions = [3]
        XCTAssertTrue(canvas.configure(nodeID: id, config: config.values, lease: owner.lease()))
        await settle(canvas)
        let findings = WorkflowSelectionPresentation.configurationIssues(canvas.validation.issues, displayIDs: [id])
        XCTAssertTrue(findings.contains { $0.nodeID == id && $0.message.contains("expected 3") && $0.message.contains("observed 2") })
        XCTAssertFalse(canvas.canvasPortDescriptor(nodeID: id, port: "in", direction: .input).declared.isEmpty)
        config.expectedSampleDimensions = [2]
        XCTAssertTrue(canvas.configure(nodeID: id, config: config.values, lease: owner.lease()))
        await settle(canvas)
        XCTAssertTrue(WorkflowSelectionPresentation.configurationIssues(canvas.validation.issues, displayIDs: [id]).isEmpty)
    }
    @MainActor func testNamedInputsWithoutTrainMappingReachBoundaryAndTrainConfiguration() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let document=try ModelOutputPracticeExample.document(device:"cpu")
        _=try library.saveSource(.workflow(document))
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        _ = try owner.migrateRetiredToolsForAuthoring()
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        defer {canvas.invalidate()}
        await settle(canvas)
        let input=try WorkflowModelInput.boundary(for:ModelOutputPracticeExample.address,in:owner.document)
        XCTAssertTrue(canvas.apply(.setModelInput(definition:ModelOutputPracticeExample.address,type:"input",configuration:["features":2],destinations:WorkflowModelInput.destinations(of:try XCTUnwrap(input),in:owner.document),inputName:"input"),lease:owner.lease()),canvas.statusMessage)
        XCTAssertThrowsError(try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:16))
        await settle(canvas)
        for address in [ModelOutputPracticeExample.address,WorkflowExamples.trainAddress] {
            let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address])
            XCTAssertTrue(canvas.validation.issues.contains {$0.nodeID==id && $0.message.contains("named tensor mapping")})
            XCTAssertNotNil(canvas.modelConfigurationIssue(for:address))
        }
        XCTAssertNotNil(canvas.currentRunConfigurationFinding)
        XCTAssertTrue(owner.select(.init(nodeAddresses:[ModelOutputPracticeExample.address]),lease:owner.lease()))
        let definition=try WorkflowControlNodeContext(owner:owner,address:ModelOutputPracticeExample.address)
        let output=try WorkflowModelNodeContext(authority:definition.authority)
        XCTAssertTrue(output.authority.configure(output.configuration,expected:output.configuration))
        XCTAssertTrue(definition.isCurrent)
        XCTAssertTrue(definition.rename("Output boundary"))
        XCTAssertTrue(output.isCurrent)
        // Recorded evidence still reconstructs the original request, including its failure.
        XCTAssertNoThrow(try WorkflowCompiler.compileHistoricalEvidence(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:16))
        let snapshot=try WorkflowCompiler.modelSnapshot(owner.document,definition:ModelOutputPracticeExample.address,modelRegistry:models,dataRegistry:data)
        let plan=NamedInputPlan(inputs:[.init(name:"input",nodeID:try XCTUnwrap(snapshot.graph.namedInputs?["input"]),sampleShape:[2],columns:["signal","context"])],roles:[.init(name:"target",dtype:"int64",sampleShape:[],columns:["label"])])
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let train=try WorkflowControlNodeContext(owner:owner,address:WorkflowExamples.trainAddress)
        XCTAssertTrue(train.configureMapping(["namedInputPlan":.object(plan.values),"featureColumns":.array([.string("signal"),.string("context")]),"targetColumn":.string("label")],expected:train.configuration),train.statusMessage)
        await settle(canvas)
        XCTAssertNoThrow(try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:16))
        for address in [ModelOutputPracticeExample.address,WorkflowExamples.trainAddress] {XCTAssertNil(canvas.modelConfigurationIssue(for:address))}
        XCTAssertNil(canvas.currentRunConfigurationFinding)
    }
    @MainActor private func settle(_ canvas: WorkflowCanvasAdapter) async {
        for _ in 0..<200 {
            if !canvas.isInferringTensors { return }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTFail("Model preflight did not finish within the bounded wait")
    }
}
