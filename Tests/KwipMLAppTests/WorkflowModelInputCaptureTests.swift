import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowModelInputCaptureTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    // Independent contract fixture; this is not a claim of runtime discovery.
    private let descriptor = try! JSONDecoder().decode(JSONValue.self,from:Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8))
    private func edited()throws->(WorkflowDocument,ComponentRegistry,DataFlowOperatorRegistry) {
        let model=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let original=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let definition=try XCTUnwrap(original.nodes.first {if case .modelDefinition=$0.payload{return true};return false})
        let input=try XCTUnwrap(WorkflowModelInput.boundary(for:definition.address,in:original))
        guard case .modelPrimitive(let type,_,let config)=input.payload else {throw DataFlowContractError("Missing input fixture")}
        let updated=try WorkflowEditing.apply(.setModelInput(definition:definition.address,type:type,configuration:config,destinations:WorkflowModelInput.destinations(of:input,in:original)),to:original,selection:.init(),modelRegistry:model,dataRegistry:data).document
        XCTAssertEqual(updated.schemaVersion,14)
        XCTAssertEqual(updated.modelInputs,[definition.address.localID:input.address.localID])
        return (updated,model,data)
    }
    private func write(_ root:URL)throws->(WorkflowArchiveCapture,WorkflowDocument,URL) {
        let(document,model,data)=try edited()
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:12)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        let request=WorkflowExecutionRequest(lease:.init(projectID:document.projectID,sessionID:UUID(),generation:0),document:document,authoringDocument:document,experiment:nil,experimentInputs:nil,train:plan.train,plan:plan,
            modelRegistrySHA256:try WorkflowArchiveCodec.registryHash(model),dataRegistrySHA256:try WorkflowArchiveCodec.registryHash(data),runtime:runtime,runtimeEvidence:try WorkflowRuntimeEvidence.capture(runtime),requestIdentity:String(repeating:"a",count:64),priorCache:nil,gradientArguments:[],checkpointParent:nil,executionBudget:nil,sampleCapture:nil,preparationCacheLimitBytes:nil,capturedModelRegistry:model,derivativeCapture:nil,moduleCompositionDescriptor:descriptor)
        let source=try WorkflowCapturedSource.capture(repository.appendingPathComponent("examples/dataflow/observations.csv"))
        let directory=root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let result=try WorkflowArchiveCaptureWriter.write(request:request,runID:UUID(),sources:[WorkflowExamples.sourceAddress:source],directory:directory,modelRegistry:model,dataRegistry:data)
        let capture=try WorkflowArchiveCodec.decodeCapture(Data(contentsOf:directory.appendingPathComponent(result.reference.path)))
        XCTAssertTrue(try capture.plan.matches(plan,compilerContractVersion:12))
        return (capture,document,directory)
    }
    func testEditedSchemaFourteenPublishesExactOwnedSourceUnderExistingCompilerAndCaptureContracts()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let(capture,document,directory)=try write(root)
        XCTAssertEqual(WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:14),12)
        XCTAssertEqual(capture.schemaVersion,14);XCTAssertEqual(capture.compilerContractVersion,12);XCTAssertEqual(capture.captureFormatVersion,8)
        let source=try Data(contentsOf:directory.appendingPathComponent(capture.workflow.path))
        XCTAssertEqual(WorkflowArchiveCodec.hash(source),capture.workflow.sha256)
        XCTAssertEqual(try WorkflowCodec.decode(source),document,"History must retain canonical ownership, not a stripped worker projection")
        let provenance=try XCTUnwrap(capture.plan.modelProvenance)
        XCTAssertEqual(provenance.sourceCompositeDocument.schemaVersion,2,"Ownership alone must preserve legacy reusable model source")
        XCTAssertEqual(provenance.nativeExpansionContractVersion,1)
        XCTAssertTrue(document.modelSourceVersions?.values.contains(2) == true)
        XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(capture)),capture)
        XCTAssertEqual(capture.moduleCompositionDescriptor,descriptor)
    }
    func testSchemaFourteenCannotAcquireOlderCompilerFormatOrLoseCompositionEvidence()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let(capture,_,_)=try write(root)
        var wrong=capture;wrong.compilerContractVersion=11
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong))
        wrong=capture;wrong.captureFormatVersion=7
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong))
        wrong=capture;wrong.moduleCompositionDescriptor=nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong))
        wrong=capture;wrong.schemaVersion=15
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong))
        var payload=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encodeCapture(capture)) as? [String:Any])
        var plan=try XCTUnwrap(payload["plan"] as? [String:Any])
        var provenance=try XCTUnwrap(plan["modelProvenance"] as? [String:Any])
        provenance["nativeExpansionContractVersion"]=2
        plan["modelProvenance"]=provenance;payload["plan"]=plan
        XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:payload)),"Source2 cannot claim source3/native2 expansion")
    }
    @MainActor func testEditedProjectStillRequiresQualifiedCompositionDiscoveryBeforeRequestCapture()throws {
        let(document,model,data)=try edited()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:model,dataRegistry:data)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime)) {
            XCTAssertTrue($0.localizedDescription.contains("verify native module composition"),$0.localizedDescription)
        }
    }
}
