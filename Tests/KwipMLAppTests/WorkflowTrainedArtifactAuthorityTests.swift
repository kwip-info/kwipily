import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowTrainedArtifactAuthorityTests:XCTestCase {
    private func plan(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:max(2,WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:document.schemaVersion)))
    }
    func testScientificIdentityIgnoresOnlyLayoutLabelsAndRevision()throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let original=try WorkflowTrainedArtifactAuthority.scientificIdentity(plan(document))
        document.revision += 1;document.name="Renamed workflow"
        for i in document.nodes.indices {document.nodes[i].name="Relabeled";document.nodes[i].x += 100;document.nodes[i].y -= 30}
        XCTAssertEqual(try WorkflowTrainedArtifactAuthority.scientificIdentity(plan(document)),original)
    }
    func testScientificIdentityBindsTrainingSeedModelAndPreparationConfiguration()throws {
        let original=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let identity=try WorkflowTrainedArtifactAuthority.scientificIdentity(plan(original))
        for field in ["seed","learningRate"] {
            var changed=original
            let index=try XCTUnwrap(changed.nodes.firstIndex {if case .runSetup=$0.payload{return true};return false})
            guard case .runSetup(var config)=changed.nodes[index].payload else {return XCTFail("Missing setup")}
            config[field]=field=="seed" ? .integer(123):.number(0.123)
            changed.nodes[index].payload = .runSetup(config:config)
            XCTAssertNotEqual(try WorkflowTrainedArtifactAuthority.scientificIdentity(plan(changed)),identity,field)
        }
        var changed=original
        let split=try XCTUnwrap(changed.nodes.firstIndex {if case .dataOperator("kwip.data.partition",_)=$0.payload{return true};return false})
        guard case .dataOperator(let type,var config)=changed.nodes[split].payload else {return XCTFail("Missing partition")}
        config["seed"]=123;changed.nodes[split].payload = .dataOperator(operatorID:type,config:config)
        XCTAssertNotEqual(try WorkflowTrainedArtifactAuthority.scientificIdentity(plan(changed)),identity)
    }
    @MainActor func testPrivateTrainReferenceAloneDoesNotAuthorizeMissingArchiveOrSavedAlias()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let document=try WorkflowTrainedModelSource.promoteAuthoring(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        let bindings=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
        let pin=String(repeating:"a",count:64)
        let reference=WorkflowFrozenModelReference(projectID:document.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,outcomeSHA256:pin,modelManifestSHA256:pin,preparationManifestSHA256:pin,checkpointSHA256:pin)
        XCTAssertTrue(bindings.setReference(reference,for:WorkflowExamples.trainAddress,lease:owner.lease(),expectedRevision:0))
        let endpoint=WorkflowEndpoint(address:WorkflowExamples.trainAddress,port:"model")
        for source in [WorkflowTrainedModelSource.train(endpoint),.savedCheckpoint(endpoint)] {
            do {_ = try await WorkflowTrainedArtifactAuthority.resolve(source:source,owner:owner,bindings:bindings,root:root);XCTFail("Unverified or mis-typed provider acquired authority")}
            catch {XCTAssertEqual(bindings.reference(for:WorkflowExamples.trainAddress),reference,"Refusal must preserve selected evidence")}
        }
    }
}
