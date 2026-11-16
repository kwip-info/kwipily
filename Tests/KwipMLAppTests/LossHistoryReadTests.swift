import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LossHistoryReadTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func temporaryRoot()throws->URL {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        return URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
    }
    private struct Saved {
        let task:LossLearningTask
        let reference:WorkflowHistoryReference
        let directory:URL
        let recipe:LossRecipe
        let outputShape:[Int]
        let registryHash:String
        let workerRunID:String
    }
    private func capture(_ tasks:[LossLearningTask],root:URL)async throws->[Saved] {
        let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"].map {URL(fileURLWithPath:$0)} ?? repository
        let workspace=WorkspaceStore(rootOverride:development,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertEqual(workspace.lossRuntimeCapabilities.componentIDs,LossConfiguration.componentIDs)
        var saved:[Saved]=[]
        for task in tasks {
            let document=try WorkflowExamples.preparedLossObjective(task:task,device:"cpu")
            _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
            XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
            let host=try XCTUnwrap(workspace.workflowSession)
            let resource=try XCTUnwrap(BundledLossComponents.entries[task.componentID]?.learning?.sourceResource)
            let source=root.appendingPathComponent(task.rawValue+".csv")
            try FileManager.default.copyItem(at:repository.appendingPathComponent("Sources/KwipMLApp/Resources/"+resource+".csv"),to:source)
            try workspace.sourceAccess.remember(source)
            XCTAssertTrue(host.bindings.setPath(source.path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
            let finished=expectation(description:"Actual captured objective "+task.rawValue)
            let subscription=host.$outcomes.dropFirst().filter {!$0.isEmpty}.prefix(1).sink {_ in finished.fulfill()}
            XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
            await fulfillment(of:[finished],timeout:45)
            subscription.cancel()
            if workspace.isBusy {host.cancel()}
            let outcome=try XCTUnwrap(host.outcomes.first)
            XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage);XCTAssertNil(outcome.recordError)
            let directoryID=try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent))
            let recipe=try XCTUnwrap(outcome.request.plan.lossRecipe)
            let shape=try WorkflowMappingRequirements.outputContract(outcome.request.plan.model.graph,registry:try XCTUnwrap(outcome.request.capturedModelRegistry)).layout.sample.dimensions
            let recorded=try WorkflowArchiveCodec.decodeOutcome(Data(contentsOf:outcome.directory.appendingPathComponent("outcome.json")))
            saved.append(.init(task:task,reference:.init(projectID:document.projectID,directoryID:directoryID),directory:outcome.directory,recipe:recipe,outputShape:shape,registryHash:outcome.request.modelRegistrySHA256,workerRunID:try XCTUnwrap(recorded.workerRunID)))
            host.retire();workspace.retireWorkflowSession()
        }
        return saved
    }
    func testActualColdObjectiveReadRevalidatesPinnedBytesAndRefusesMissingArtifact()async throws {
        let root=try temporaryRoot()
        defer {try? FileManager.default.removeItem(at:root)}
        let actual=try await capture([.mse],root:root)
        let item=try XCTUnwrap(actual.first)
        let before=try XCTUnwrap(WorkflowHistoryReader.loadModelContent(item.reference,root:root))
        XCTAssertNotNil(before.objectiveEvaluation)
        let predictions=item.directory.appendingPathComponent("training/artifacts/predictions.json")
        let original=try Data(contentsOf:predictions)
        try Data("{}".utf8).write(to:predictions)
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(item.reference,root:root))
        guard case .unavailable=WorkflowHistoryReader.load(item.reference,root:root) else {return XCTFail("Changed actual prediction bytes must invalidate History")}
        try original.write(to:predictions)
        XCTAssertNotNil(try WorkflowHistoryReader.loadModelContent(item.reference,root:root)?.objectivePredictionPreview)
        let objective=item.directory.appendingPathComponent("training/artifacts/objective_recipe.py")
        try FileManager.default.removeItem(at:objective)
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(item.reference,root:root))
    }
}


/// Positive cold-reader and export guards consume the same exact current inline CPU captures.
/// Negative History/byte-corruption tests above retain their own independent capture and root.
@MainActor enum SavedObjectiveHistoryAssertions {
    static func verify(task:LossLearningTask,reference:WorkflowHistoryReference,directory:URL,
                       recipe:LossRecipe,outputShape:[Int],registryHash:String,workerRunID:String,root:URL)throws {
        guard case .verified(let archived)=WorkflowHistoryReader.load(reference,root:root) else {XCTFail("Cold actual objective archive must verify: "+task.rawValue);return}
        XCTAssertEqual(archived.plan.lossRecipe,recipe)
        XCTAssertEqual(archived.capture.modelRegistrySHA256,registryHash)
        let content=try XCTUnwrap(WorkflowHistoryReader.loadModelContent(reference,root:root))
        XCTAssertEqual(content.objectiveContext?.recipe,recipe)
        XCTAssertEqual(content.objectiveContext?.runID,workerRunID)
        XCTAssertEqual(content.objectiveContext?.outputSampleShape,outputShape)
        XCTAssertEqual(content.objectivePredictionPreview?.runID,workerRunID)
        XCTAssertEqual(content.objectivePredictionPreview?.source,recipe.source)
        XCTAssertEqual(content.objectivePredictionPreview?.rows.count,16)
        XCTAssertEqual(content.objectivePredictionPreview?.rows.first?.output.shape,outputShape)
        XCTAssertEqual(content.objectiveEvaluation?.componentID,task.componentID)
        XCTAssertEqual(content.objectiveEvaluation?.observations,16)
        XCTAssertNil(content.predictionPreview,"Typed objective outputs cannot become legacy binary rows")
        XCTAssertEqual(content.metrics.map(\.epoch),[1,2,3])
        if [.crossEntropy,.nll].contains(task) {XCTAssertNotNil(content.metrics.last?.accuracy)} else {XCTAssertNil(content.metrics.last?.accuracy)}
        XCTAssertEqual(content.objectiveModelCode,try String(contentsOf:directory.appendingPathComponent("training/artifacts/objective_recipe.py"),encoding:.utf8))
        XCTAssertEqual(content.code,try String(contentsOf:directory.appendingPathComponent("training/artifacts/model.py"),encoding:.utf8))
        XCTAssertEqual(content.outcome.workerRunID,workerRunID)
    }
}
