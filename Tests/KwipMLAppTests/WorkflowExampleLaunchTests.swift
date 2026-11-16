import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExampleLaunchTests: XCTestCase {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @MainActor func testExampleCreatesFreshWorkflowAndKeepsLocalBindingOutOfPortableSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(), workspace.statusMessage)
        let first = try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(workspace.activeProjectID,first.owner.projectID)
        let path = try XCTUnwrap(first.bindings.path(for:WorkflowExamples.sourceAddress))
        XCTAssertEqual(URL(fileURLWithPath:path).lastPathComponent,WorkflowExamples.observationsResourceName)
        XCTAssertTrue(path.hasPrefix(root.appendingPathComponent("example-sources").path + "/"), "Example bindings must survive app bundle replacement")
        let rows = try String(contentsOfFile:path,encoding:.utf8).split(whereSeparator: \.isNewline)
        XCTAssertEqual(rows.count,65)
        let bytes = try WorkflowCodec.encode(first.owner.document)
        XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains(path))
        XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("sourceBindings"))
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:first.owner.projectID),.workflow(first.owner.document))
        XCTAssertNotNil(workspace.workflowHistory)
        XCTAssertEqual(workspace.activeSection,"Build")
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let second = try XCTUnwrap(workspace.workflowSession)
        XCTAssertNotEqual(first.owner.projectID,second.owner.projectID)
        XCTAssertTrue(first.isRetired)
        XCTAssertNotEqual(second.bindings.path(for:WorkflowExamples.sourceAddress),path)
        XCTAssertTrue(workspace.beginWorkflowProject(id:first.owner.projectID),workspace.statusMessage)
        let restored = try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(restored.bindings.path(for:WorkflowExamples.sourceAddress),path)
        XCTAssertEqual(try String(contentsOfFile:path,encoding:.utf8).split(whereSeparator: \.isNewline).count,65)
        let compiled = try WorkflowCompiler.compile(second.owner.document,train:WorkflowExamples.trainAddress,
            modelRegistry:second.owner.modelRegistry,dataRegistry:second.owner.dataRegistry,contractVersion:15)
        XCTAssertEqual(compiled.settings.device,"cpu")
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:first.owner.projectID),.workflow(first.owner.document))
    }
    @MainActor func testPartitionExamplesLoadPersistAndReopenWithExplicitEvaluation()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        for temporal in [false,true] {
            XCTAssertTrue(temporal ? workspace.loadTemporalSplitWorkflowExample(device:"cpu") : workspace.loadGroupSplitWorkflowExample(device:"cpu"),workspace.statusMessage)
            let host=try XCTUnwrap(workspace.workflowSession),document=host.owner.document
            XCTAssertEqual(document.schemaVersion,18)
            let frozen=try XCTUnwrap(document.nodes.first {if case .frozenModel=$0.payload{return true};return false})
            let evaluate=try XCTUnwrap(document.nodes.first {if case .evaluateTest=$0.payload{return true};return false})
            XCTAssertEqual(frozen.payload,.frozenModel(config:[:]))
            let path=try XCTUnwrap(host.bindings.path(for:WorkflowExamples.sourceAddress))
            XCTAssertTrue(FileManager.default.fileExists(atPath:path))
            XCTAssertFalse(String(decoding:try WorkflowCodec.encode(document),as:UTF8.self).contains(path))
            let training=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry,contractVersion:15)
            XCTAssertEqual(training.settings.device,"cpu")
            let evaluation=try WorkflowEvaluationCompiler.compile(document,evaluate:evaluate.address,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry)
            XCTAssertEqual(evaluation.testEndpoint,.init(node:"split",port:"test"))
            XCTAssertTrue(workspace.loadPreparedWorkflowExample(),workspace.statusMessage)
            XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
            let reopened=try XCTUnwrap(workspace.workflowSession)
            XCTAssertEqual(reopened.owner.document,document)
            XCTAssertEqual(reopened.bindings.path(for:WorkflowExamples.sourceAddress),path)
        }
    }
    @MainActor func testBusyOrInvalidDeviceDoesNotReplaceCurrentProject() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample())
        let host = try XCTUnwrap(workspace.workflowSession), id = workspace.activeProjectID
        XCTAssertFalse(workspace.loadPreparedWorkflowExample(device:"cuda"))
        XCTAssertTrue(workspace.workflowSession === host);XCTAssertEqual(workspace.activeProjectID,id)
        workspace.isPreparingData = true
        XCTAssertFalse(workspace.loadPreparedWorkflowExample())
        XCTAssertTrue(workspace.workflowSession === host);XCTAssertFalse(host.isRetired)
        workspace.isPreparingData = false
    }
    @MainActor func testOptInBundledExampleGeneratesCodeRunsAndReloadsVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false)
    }
    @MainActor func testOptInTensorExampleUsesUniversalCanvasAndVerifiedHistory()async throws {
        try await verifyExample(tensorLayout:true)
    }
    @MainActor func testOptInGELUExamplesUseUniversalCanvasAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,geluApproximation:false)
        try await verifyExample(tensorLayout:false,geluApproximation:true)
    }
    @MainActor func testOptInGuidedCourseStagesProduceExecutableWorkflow()async throws {
        try await verifyExample(tensorLayout:false,guided:true)
    }
    @MainActor func testOptInSoftmaxExampleCodeTrainingAndVerifiedHistory()async throws {
        try await verifyExample(tensorLayout:false,softmax:true)
    }
    @MainActor func testOptInEmbeddingExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,embedding:true)
    }
    @MainActor func testOptInLayerNormExamplesCodeTrainingAndVerifiedHistory()async throws {
        for whole in [false,true] {try await verifyExample(tensorLayout:false,layerNorm:whole)}
    }
    @MainActor func testOptInMatrixExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,matrix:true)
    }
    @MainActor func testOptInBilinearExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,bilinear:true)
    }
    @MainActor func testOptInLazyLinearExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,lazyLinear:true)
    }
    @MainActor func testOptInConv1dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,conv1d:true)
    }
    @MainActor func testOptInConv2dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,conv2d:true)
    }
    @MainActor func testOptInConv3dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,conv3d:true)
    }
    @MainActor func testOptInConvTranspose1dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,convTranspose1d:true)
    }
    @MainActor func testOptInConvTranspose2dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,convTranspose2d:true)
    }
    @MainActor func testOptInConvTranspose3dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,convTranspose3d:true)
    }
    @MainActor func testOptInDepthwiseConv2dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,depthwiseConv2d:true)
    }
    @MainActor func testOptInAvgPool1dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,componentID: "avg_pool1d")
    }
    @MainActor func testOptInMaxPool3dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,componentID: "max_pool3d")
    }
    @MainActor func testOptInMaxPool2dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,maxPool2d:true)
    }
    @MainActor func testOptInMaxPool1dExampleCodeTrainingAndVerifiedHistory() async throws {
        try await verifyExample(tensorLayout:false,maxPool1d:true)
    }
    @MainActor func testOptInEmbeddingBagExampleUsesCurrentTrainCaptureAndVerifiedHistory()async throws {
        try await verifyExample(tensorLayout:false,embeddingBag:true)
    }
    @MainActor private func verifyExample(tensorLayout:Bool,geluApproximation:Bool?=nil,guided:Bool=false,softmax:Bool=false,layerNorm:Bool?=nil,embedding:Bool=false,embeddingBag:Bool=false,matrix:Bool=false,bilinear:Bool=false,lazyLinear:Bool=false,conv1d:Bool=false,conv2d:Bool=false,conv3d:Bool=false,convTranspose1d:Bool=false,convTranspose2d:Bool=false,convTranspose3d:Bool=false,depthwiseConv2d:Bool=false,maxPool1d:Bool=false,maxPool2d:Bool=false,componentID:String?=nil)async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual bundled example execution.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace = WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        if guided {
            workspace.startCourse()
            let course=try XCTUnwrap(workspace.workflowSession)
            let count=try WorkflowGuidedBuildPlan.parallelClassifier().steps.count
            for _ in 1..<count {
                let progress=try XCTUnwrap(course.owner.document.guidedBuild)
                XCTAssertTrue(course.advanceGuidedCourse(expected:progress),course.canvas.statusMessage)
            }
            XCTAssertEqual(course.owner.document.guidedBuild?.completedSteps,count)
        } else if embeddingBag {
            XCTAssertTrue(workspace.loadComponentWorkflowExample("embedding_bag",device:"cpu"),workspace.statusMessage)
        } else if let componentID {
            XCTAssertTrue(workspace.loadComponentWorkflowExample(componentID,device:"cpu"),workspace.statusMessage)
        } else if maxPool2d {
            XCTAssertTrue(workspace.loadMaxPool2dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if maxPool1d {
            XCTAssertTrue(workspace.loadMaxPool1dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if depthwiseConv2d {
            XCTAssertTrue(workspace.loadDepthwiseConv2dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if convTranspose3d {
            XCTAssertTrue(workspace.loadConvTranspose3dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if convTranspose2d {
            XCTAssertTrue(workspace.loadConvTranspose2dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if convTranspose1d {
            XCTAssertTrue(workspace.loadConvTranspose1dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if conv3d {
            XCTAssertTrue(workspace.loadConv3dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if conv2d {
            XCTAssertTrue(workspace.loadConv2dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if conv1d {
            XCTAssertTrue(workspace.loadConv1dWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if lazyLinear {
            XCTAssertTrue(workspace.loadLazyLinearWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if bilinear {
            XCTAssertTrue(workspace.loadBilinearWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if matrix {
            XCTAssertTrue(workspace.loadMatrixWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if embedding {
            XCTAssertTrue(workspace.loadEmbeddingWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if let layerNorm {
            XCTAssertTrue(workspace.loadLayerNormWorkflowExample(normalizeWholeSample:layerNorm,device:"cpu"),workspace.statusMessage)
        } else if softmax {
            XCTAssertTrue(workspace.loadSoftmaxWorkflowExample(device:"cpu"),workspace.statusMessage)
        } else if let geluApproximation {
            XCTAssertTrue(workspace.loadGELUWorkflowExample(approximateTanh:geluApproximation,device:"cpu"),workspace.statusMessage)
        } else {
            XCTAssertTrue(tensorLayout ? workspace.loadTensorLayoutWorkflowExample(device:"cpu") : workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        }
        XCTAssertEqual(workspace.projectSections,["Build","History","Code","Settings"])
        let host = try XCTUnwrap(workspace.workflowSession), original = host.owner.document
        // These component proofs inspect the standalone model implementation.
        workspace.workflowCodeScope = .model
        let codeReady = expectation(description:"Actual export before execution")
        let codeObservation = workspace.$isGeneratingCode.drop(while: {!$0}).filter{!$0}.prefix(1).sink{_ in codeReady.fulfill()}
        defer {codeObservation.cancel()}
        workspace.refreshDraftCode()
        XCTAssertTrue(workspace.isGeneratingCode,workspace.draftCodeError ?? workspace.statusMessage)
        await fulfillment(of:[codeReady],timeout:30)
        XCTAssertNil(workspace.draftCodeError)
        XCTAssertTrue(workspace.draftCode.contains("class Model"))
        XCTAssertTrue(workspace.draftCode.contains("nn.Linear"))
        if let componentID {
            XCTAssertTrue(workspace.poolingRuntimeCapabilities.contains(componentID))
            XCTAssertTrue(workspace.draftCode.contains(componentID + "("))
            XCTAssertTrue(workspace.draftCode.contains(componentID == "avg_pool1d" ? "count_include_pad=True" : "return_indices=True"))
        }
        if maxPool2d {
            XCTAssertTrue(workspace.poolingRuntimeCapabilities.contains("max_pool2d"))
            XCTAssertTrue(workspace.draftCode.contains("max_pool2d("))
            XCTAssertTrue(workspace.draftCode.contains("return_indices=True"))
        }
        if maxPool1d {
            XCTAssertTrue(workspace.poolingRuntimeCapabilities.contains("max_pool1d"))
            XCTAssertTrue(workspace.draftCode.contains("max_pool1d("))
            XCTAssertTrue(workspace.draftCode.contains("return_indices=True"))
        }
        if depthwiseConv2d {
            XCTAssertTrue(workspace.draftCode.contains("nn.Conv2d("))
            XCTAssertTrue(workspace.draftCode.contains("groups=2"))
            XCTAssertNotNil(workspace.depthwiseConv2dRuntimeCapability)
            let graph = ExampleFlows.depthwiseConv2dClassifier()
            XCTAssertEqual(GraphValidator.validate(graph, registry: workspace.registry).tensorLayouts["convolution"]?.sample.dimensions, [4, 4, 5])
        }
        if convTranspose3d {
            XCTAssertTrue(workspace.draftCode.contains("nn.ConvTranspose3d("))
            XCTAssertTrue(workspace.draftCode.contains("output_padding=(0, 0, 0)"))
            XCTAssertNotNil(workspace.convTranspose3dRuntimeCapability)
            let graph = ExampleFlows.convTranspose3dClassifier()
            XCTAssertEqual(GraphValidator.validate(graph, registry: workspace.registry).tensorLayouts["convolution"]?.sample.dimensions, [4, 3, 5, 7])
        }
        if convTranspose2d {
            XCTAssertTrue(workspace.draftCode.contains("nn.ConvTranspose2d("))
            XCTAssertTrue(workspace.draftCode.contains("output_padding=(1, 0)"))
            XCTAssertNotNil(workspace.convTranspose2dRuntimeCapability)
        }
        if convTranspose1d {
            XCTAssertTrue(workspace.draftCode.contains("nn.ConvTranspose1d("))
            XCTAssertTrue(workspace.draftCode.contains("output_padding=(1,)"))
            XCTAssertNotNil(workspace.convTranspose1dRuntimeCapability)
        }
        if conv3d {
            XCTAssertTrue(workspace.draftCode.contains("nn.Conv3d("))
            XCTAssertTrue(workspace.draftCode.contains("groups=2"))
            XCTAssertNotNil(workspace.conv3dRuntimeCapability)
        }
        if conv2d {
            XCTAssertTrue(workspace.draftCode.contains("nn.Conv2d("))
            XCTAssertTrue(workspace.draftCode.contains("groups=2"))
            XCTAssertNotNil(workspace.conv2dRuntimeCapability)
        }
        if conv1d {
            XCTAssertTrue(workspace.draftCode.contains("nn.Conv1d("))
            XCTAssertTrue(workspace.draftCode.contains("padding='same'"))
            XCTAssertTrue(workspace.draftCode.contains("groups=2"))
            XCTAssertNotNil(workspace.conv1dRuntimeCapability)
        }
        if lazyLinear {
            XCTAssertTrue(workspace.draftCode.contains("nn.LazyLinear(4, bias=True)"))
            XCTAssertTrue(workspace.draftCode.contains("nn.LazyLinear(2, bias=True)"))
        }
        if let geluApproximation {
            XCTAssertTrue(workspace.draftCode.contains("torch.nn.functional.gelu("))
            XCTAssertTrue(workspace.draftCode.contains(geluApproximation ? "approximate='tanh'" : "approximate='none'"))
            let compiled=try WorkflowCompiler.compile(original,train:WorkflowExamples.trainAddress,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry,contractVersion:15)
            XCTAssertEqual(compiled.model.graph.nodes.first(where:{$0.type == "gelu"})?.config["approximate_tanh"],.bool(geluApproximation))
        }
        if bilinear {
            XCTAssertTrue(workspace.draftCode.contains("nn.Bilinear("))
            XCTAssertNotNil(workspace.bilinearRuntimeCapability)
        }
        if matrix {
            XCTAssertTrue(workspace.draftCode.contains("torch.matmul("))
            XCTAssertNotNil(workspace.matrixMultiplyRuntimeCapability)
        }
        if embedding {
            XCTAssertTrue(workspace.draftCode.contains("Embedding"))
            XCTAssertNotNil(workspace.embeddingRuntimeCapability)
            XCTAssertEqual(URL(fileURLWithPath:try XCTUnwrap(host.bindings.path(for:WorkflowExamples.sourceAddress))).lastPathComponent,"embedding-observations.csv")
        }
        if layerNorm != nil {XCTAssertTrue(workspace.draftCode.contains("nn.LayerNorm"));XCTAssertNotNil(workspace.layerNormRuntimeCapability)}
        if softmax {
            XCTAssertTrue(workspace.draftCode.contains("torch.softmax("))
            XCTAssertTrue(workspace.draftCode.contains("dim=-1"))
            XCTAssertNotNil(workspace.softmaxRuntimeCapability)
        }
        if tensorLayout {XCTAssertTrue(workspace.draftCode.contains(".reshape("));XCTAssertTrue(workspace.draftCode.contains(".transpose("));XCTAssertTrue(workspace.draftCode.contains(".flatten("))}
        XCTAssertTrue(host.outcomes.isEmpty,"Code generation is independent of training")
        let terminal = expectation(description:"Bundled example actual training")
        let observation = host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in terminal.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        await fulfillment(of:[terminal],timeout:45)
        if workspace.isBusy {host.cancel()}
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        XCTAssertNil(outcome.recordError)
        XCTAssertEqual(host.execution?.metrics.count,3)
        XCTAssertTrue(host.execution?.metrics.allSatisfy{$0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.accuracy.map(\.isFinite) == true} == true)
        XCTAssertFalse(workspace.isBusy);XCTAssertNil(workspace.cancelDataPreparation)
        XCTAssertEqual(host.owner.document,original)
        let listing = try WorkflowHistoryReader.list(root:root,projectID:host.owner.projectID,limit:1000)
        XCTAssertEqual(listing.entries.count,1);XCTAssertTrue(listing.issues.isEmpty)
        let reference = try XCTUnwrap(listing.entries.first?.reference)
        guard case .verified(let verified) = WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Actual example archive failed verification")}
        XCTAssertEqual(verified.outcome.status,.completed)
        let cache = try XCTUnwrap(verified.preparationCacheReport)
        XCTAssertEqual(cache.policy, "off")
        XCTAssertEqual(cache.executedNodeIDs.count, (embedding || embeddingBag) ? 3 : 6, "Training-side Views are captured alongside Train preparation")
        XCTAssertTrue(cache.reusedNodeIDs.isEmpty)
        XCTAssertEqual(verified.capture.projectID,host.owner.projectID)
        XCTAssertEqual(verified.capture.train,WorkflowExamples.trainAddress)
        XCTAssertTrue(try verified.capture.plan.matches(outcome.request.plan, compilerContractVersion: verified.capture.compilerContractVersion))
        let content = try XCTUnwrap(verified.modelContent)
        XCTAssertEqual(content.metrics.count,3);XCTAssertTrue(content.code.contains("class Model"))
        if maxPool1d || maxPool2d || componentID != nil {
            let live = try XCTUnwrap(host.modelSnapshot(for:.init(domain:.model,localID:"pool")))
            let displayed = try XCTUnwrap(host.canvas.projection.displayIDByAddress[.init(domain:.model,localID:"pool")])
            XCTAssertEqual(host.canvas.previewStatus(for:displayed),"Captured")
            XCTAssertTrue(live.matchesCurrentConstruction);XCTAssertTrue(live.matchesCurrentBindings)
            if componentID == "avg_pool1d" {
                guard case .object(let values) = live.ports["out"],
                      case .object(let saved) = content.activationPreviews,
                      case .object(let pool) = saved["pool"] else { return XCTFail("Missing average pooling captures") }
                XCTAssertNil(live.ports["indices"])
                XCTAssertEqual(values["dtype"],.string("float32"))
                XCTAssertEqual(values["values"],pool["values"])
            } else {
                guard case .object(let indices) = live.ports["indices"],
                      case .object(let saved) = content.activationPreviews,
                      case .object(let pool) = saved["pool"],case .object(let outputs) = pool["outputs"] else { return XCTFail("Missing live/saved pooling captures") }
                XCTAssertEqual(indices["dtype"],.string("int64"))
                XCTAssertEqual(outputs["indices"],.object(indices.filter { $0.key != "captureStatus" }))
            }
        }

        if tensorLayout || geluApproximation != nil || guided || softmax || layerNorm != nil || embedding || embeddingBag || matrix || bilinear || lazyLinear || conv1d || conv2d || conv3d || convTranspose1d || convTranspose2d || convTranspose3d || depthwiseConv2d || maxPool1d || maxPool2d || componentID != nil {XCTAssertNil(verified.capture.plan.modelProvenance)} else {XCTAssertNotNil(verified.capture.plan.modelProvenance)}
        XCTAssertNotNil(verified.outcome.preparationManifestSHA256);XCTAssertNotNil(verified.outcome.modelManifestSHA256)
        XCTAssertEqual(verified.capture.sources.map(\.address),[WorkflowExamples.sourceAddress])
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:host.owner.projectID),.workflow(original))
        let trainingAddress = WorkflowNodeAddress(domain: .data, localID: (embedding || embeddingBag) ? "split" : "fit")
        let captured = try XCTUnwrap(host.snapshot(for: trainingAddress))
        XCTAssertEqual(captured.runID, outcome.runID)
        XCTAssertEqual(captured.requestIdentity, outcome.request.requestIdentity)
        XCTAssertEqual(captured.capturedRevision, original.revision)
        XCTAssertEqual(captured.phase, .completed)
        XCTAssertTrue(captured.matchesCurrentConstruction)
        XCTAssertTrue(captured.matchesCurrentBindings)
        guard case .object(let table) = captured.ports["training"],
              case .object(let preview) = table["preview"] else { return XCTFail("Expected captured training preview") }
        XCTAssertEqual(preview["rowCount"], .integer(embedding ? 64 : 32))
        XCTAssertNil(host.snapshot(for: .init(domain: .model, localID: (embedding || embeddingBag) ? "split" : "fit")))
        XCTAssertNil(host.snapshot(for: .init(domain: .data, localID: "not-executed")))
        XCTAssertTrue(host.bindings.setPath(nil, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        XCTAssertFalse(try XCTUnwrap(host.snapshot(for: trainingAddress)).matchesCurrentBindings)
        XCTAssertTrue(host.owner.rename("Edited after captured run", lease: host.owner.lease()))
        XCTAssertTrue(try XCTUnwrap(host.snapshot(for: trainingAddress)).matchesCurrentConstruction,
            "A project label does not change the captured computation")
        XCTAssertNotNil(host.owner.apply(.configure(address: WorkflowExamples.sourceAddress,
            config: ["separator": .string(";")]), lease: host.owner.lease()))
        let earlier = try XCTUnwrap(host.snapshot(for: trainingAddress))
        XCTAssertFalse(earlier.matchesCurrentConstruction)
        if maxPool1d || maxPool2d || componentID != nil {
            let stale = try XCTUnwrap(host.modelSnapshot(for:.init(domain:.model,localID:"pool")))
            XCTAssertFalse(stale.matchesCurrentConstruction);XCTAssertFalse(stale.matchesCurrentBindings)
            let displayed = try XCTUnwrap(host.canvas.projection.displayIDByAddress[.init(domain:.model,localID:"pool")])
            XCTAssertEqual(host.canvas.previewStatus(for:displayed),"Stale")
        }

        XCTAssertEqual(earlier.ports, captured.ports)
        XCTAssertEqual(earlier.requestIdentity, captured.requestIdentity)
        workspace.newProject()
        if maxPool1d || maxPool2d || componentID != nil { XCTAssertNil(host.modelSnapshot(for:.init(domain:.model,localID:"pool"))) }
        XCTAssertNil(host.snapshot(for: trainingAddress), "Retired sessions must not supply current inspector evidence")
    }
}
