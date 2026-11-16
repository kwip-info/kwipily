import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainExperimentInlineContextTests:XCTestCase {
    private func fixture()throws->(WorkflowAuthoringStore,WorkflowControlNodeContext,WorkflowNodeAddress) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let setup=try XCTUnwrap(document.edges.first {$0.to.address == WorkflowExamples.trainAddress && $0.to.port == "configuration"}?.from.address)
        return (owner,try .init(owner:owner,address:WorkflowExamples.trainAddress),setup)
    }
    func testTrainEditsOnlyItsBoundSetupAndPreservesTrainSelectionAndUndo()throws {
        let (owner,context,setup)=try fixture()
        let trainBefore=context.configuration
        let expected=try XCTUnwrap(context.experimentConfiguration)
        let spec=try WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[42,43])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertEqual(context.experimentTrainNodes.map(\.address),[WorkflowExamples.trainAddress])
        XCTAssertTrue(context.setExperimentSpecification(spec,expected:expected))
        XCTAssertEqual(context.configuration,trainBefore)
        XCTAssertEqual(context.experimentSpecification,spec)
        XCTAssertEqual(owner.selection.nodeAddresses,[WorkflowExamples.trainAddress])
        guard case .runSetup(let saved)=owner.document.nodes.first(where:{$0.address == setup})?.payload else {return XCTFail("Missing setup")}
        XCTAssertNotNil(saved["experiment"])
        XCTAssertFalse(context.setExperimentSpecification(nil,expected:expected),"A stale setup draft must not clear newly saved combinations.")
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(context.isCurrent)
    }
    func testSelectionDepartureRefusesExperimentEditsAndPreview()throws {
        let (owner,context,setup)=try fixture()
        let expected=try XCTUnwrap(context.experimentConfiguration)
        let spec=try WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[42,43])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[setup]),lease:owner.lease()))
        XCTAssertFalse(context.setExperimentSpecification(spec,expected:expected))
        XCTAssertThrowsError(try context.previewExperiment(spec,train:WorkflowExamples.trainAddress,expected:expected))
        XCTAssertNil(context.experimentConfiguration)
    }
}
