import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowTrainedExportReviewTests:XCTestCase {
    func testAllEightObjectiveSummariesUseCapturedOutputGeometryWithoutInventingClasses() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        for task in LossLearningTask.allCases {
            let document=try PreparedWorkflowExample.component(task.componentID).document(device:"cpu")
            let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:12)
            let recipe=try XCTUnwrap(plan.lossRecipe)
            let contract=try TrainedModelInputContract(graph:plan.model.graph,registry:registry,recipe:recipe)
            let expectedShape=task == .ctc ? "[N, 3, 3]":task == .triplet ? "[N, 3, 2]":([LossLearningTask.crossEntropy,.nll,.kl].contains(task) ? "[N, 3]":"[N, 2]")
            let summary=contract.outputReviewSummary
            XCTAssertTrue(summary.contains(expectedShape),"\(task.componentID): \(summary)")
            XCTAssertTrue(summary.contains("float32"),task.componentID)
            XCTAssertTrue(summary.contains("numeric model values"),task.componentID)
            XCTAssertFalse(summary.contains("logits"),task.componentID)
            XCTAssertFalse(summary.contains("class index"),task.componentID)
        }
    }
    func testLegacyTwoLogitReviewCopyIsUnchanged() throws {
        let contract=try TrainedModelInputContract(graph:ExampleFlows.parallelClassifier,registry:.builtIn(),columns:["x","y"],target:"label")
        XCTAssertEqual(contract.outputReviewSummary,"Output contains two logits and a class index per row.")
    }
}
