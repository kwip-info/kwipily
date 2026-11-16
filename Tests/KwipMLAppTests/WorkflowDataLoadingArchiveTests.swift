import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowDataLoadingArchiveTests:XCTestCase {
    private func plan()throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu");document.schemaVersion=13
        var plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:12)
        plan.datasetRecipe = .init(configuration:try .init(values:DatasetConfiguration.defaults),source:.init(domain:.control,localID:"reader"))
        plan.loaderRecipe = .init(configuration:try .init(values:DataLoaderConfiguration.defaults),source:.init(domain:.control,localID:"batches"))
        return plan
    }
    func testSelectedReaderRecipesSurviveCapturedProjection()throws {
        let plan=try plan(),capture=try WorkflowArchivePlan(plan,compilerContractVersion:12)
        let bytes=try WorkflowArchiveCodec.encode(capture)
        let obj=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        let dataset=try XCTUnwrap(obj["datasetRecipe"] as? [String:Any])
        let loader=try XCTUnwrap(obj["loaderRecipe"] as? [String:Any])
        XCTAssertEqual((dataset["source"] as? [String:Any])?["localID"] as? String,"reader")
        XCTAssertEqual((loader["source"] as? [String:Any])?["localID"] as? String,"batches")
        let decoded=try JSONDecoder().decode(WorkflowArchivePlan.self,from:bytes)
        XCTAssertTrue(try decoded.matches(plan,compilerContractVersion:12))
        XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:11))
    }
    func testReaderSourceCannotImpersonateTrainOrOtherRecipe()throws {
        var plan=try plan()
        plan.datasetRecipe = .init(configuration:plan.datasetRecipe!.configuration,source:plan.train)
        XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:12))
        plan=try self.plan()
        plan.loaderRecipe = .init(configuration:plan.loaderRecipe!.configuration,source:plan.datasetRecipe!.source)
        XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:12))
    }
    func testReaderCaptureFormatRoundTripsAndRefusesOlderFormat()throws {
        let plan=try plan(),pin=String(repeating:"a",count:64)
        let descriptor = try! JSONDecoder().decode(JSONValue.self, from: Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8))
        func file(_ name:String)->WorkflowArchiveFile {.init(path:name,bytes:10,sha256:pin)}
        var capture=WorkflowArchiveCapture(schemaVersion:13,compilerContractVersion:12,projectID:plan.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,revision:plan.revision,train:plan.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),sources:plan.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},plan:try WorkflowArchivePlan(plan,compilerContractVersion:12),captureFormatVersion:8,moduleCompositionDescriptor:descriptor)
        let bytes=try WorkflowArchiveCodec.encodeCapture(capture)
        XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(bytes),capture)
        capture.captureFormatVersion=7
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(capture))
        capture.captureFormatVersion=8;capture.compilerContractVersion=11
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(capture))
    }
    func testReaderWorkflowHasItsOwnCompilerContract() {XCTAssertEqual(WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:13),12)}
    func testAbsentRecipeLegacyCaptureOmitsNewFields()throws {
        var plan=try plan();plan.datasetRecipe=nil;plan.loaderRecipe=nil
        let capture=try WorkflowArchivePlan(plan,compilerContractVersion:11)
        let obj=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(capture)) as? [String:Any])
        XCTAssertNil(obj["datasetRecipe"]);XCTAssertNil(obj["loaderRecipe"])
    }
}
