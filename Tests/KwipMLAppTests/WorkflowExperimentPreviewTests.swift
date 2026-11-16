import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExperimentPreviewTests:XCTestCase {
    private func fixture()throws->(WorkflowAuthoringStore,WorkflowControlNodeContext) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let setup=try XCTUnwrap(document.nodes.first {if case .runSetup=$0.payload{return true};return false}).address
        _=owner.select(.init(nodeAddresses:[setup]),lease:owner.lease())
        return(owner,try .init(owner:owner,address:setup))
    }
    private var spec:WorkflowExperimentSpecification {
        .init(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
    }
    func testPreviewResolvesConnectedTrainWithoutMutation()throws {
        let(owner,context)=try fixture(),before=owner.document,lease=owner.lease()
        XCTAssertEqual(context.experimentTrainNodes.map(\.address),[WorkflowExamples.trainAddress])
        let preview=try context.previewExperiment(spec,train:WorkflowExamples.trainAddress,expected:context.configuration)
        XCTAssertEqual(preview.variants.map {$0.execution.settings.seed},[7,8])
        XCTAssertEqual(preview.totalEpochs,6)
        XCTAssertEqual(owner.document,before);XCTAssertEqual(owner.lease(),lease);XCTAssertFalse(owner.canUndo)
    }
    func testWrongTrainBusyStaleAndRetiredContextsCannotPreview()throws {
        let(owner,context)=try fixture(),expected=context.configuration
        XCTAssertThrowsError(try context.previewExperiment(spec,train:.init(domain:.control,localID:"other"),expected:expected))
        owner.setExecutionBusy(true)
        XCTAssertThrowsError(try context.previewExperiment(spec,train:WorkflowExamples.trainAddress,expected:expected))
        owner.setExecutionBusy(false)
        XCTAssertTrue(context.setRunValue("seed",value:9,expected:expected))
        XCTAssertThrowsError(try context.previewExperiment(spec,train:WorkflowExamples.trainAddress,expected:expected))
        context.retire()
        XCTAssertTrue(context.experimentTrainNodes.isEmpty)
        XCTAssertThrowsError(try context.previewExperiment(spec,train:WorkflowExamples.trainAddress,expected:context.configuration))
    }
}
