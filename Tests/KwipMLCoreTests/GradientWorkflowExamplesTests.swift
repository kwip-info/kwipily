import Foundation
import XCTest
@testable import KwipMLCore

final class GradientWorkflowExamplesTests:XCTestCase {
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:13)
    }
    func testFourGenuineFactoriesCompileTypedMSEAndSourceQualifiedPolicies() throws {
        for task in GradientLearningTask.allCases {
            let document=try GradientWorkflowExamples.document(task:task,device:"cpu")
            XCTAssertEqual(document.schemaVersion,15)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
            let plan=try compile(document)
            XCTAssertEqual(plan.lossRecipe?.configuration.componentID,"loss_mse")
            XCTAssertEqual(plan.featureColumns,["x0","x1"])
            XCTAssertNil(plan.targetColumn)
            XCTAssertEqual(plan.gradientPolicyRecipe?.configuration,try GradientPolicyConfiguration(values:task.gradientValues))
            let expectedTrainability=try TrainabilityConfiguration(values:task.trainabilityValues)
            if expectedTrainability.owners.isEmpty {
                XCTAssertNil(plan.trainabilityRecipe,"Default layer trainability needs no invented recipe envelope.")
            } else {
                XCTAssertEqual(plan.trainabilityRecipe?.configuration,expectedTrainability)
                XCTAssertEqual(plan.trainabilityRecipe?.source,WorkflowExamples.trainAddress)
                XCTAssertTrue(document.nodes.contains {$0.parameterSettings != nil},"Explicit trainability remains owned by the layers.")
            }
            XCTAssertEqual(plan.gradientPolicyRecipe?.source,WorkflowExamples.trainAddress)
            XCTAssertEqual(plan.settings.batchSize,2)
            XCTAssertEqual(plan.trainingMode,task == .nextRun ? .fineTune:.fresh)
            XCTAssertEqual(Set(plan.model.graph.nodes.map(\.type)),["input","linear","output"])
            XCTAssertEqual(plan.model.graph.nodes.filter {$0.type=="linear"}.count,2)
        }
    }
    func testActualSourceRowsAndUnfabricatedSavedParentPreparation() throws {
        let rows=try GradientWorkflowExamples.sourceCSV().split(separator:"\n").map {String($0).split(separator:",").map(String.init)}
        XCTAssertEqual(rows.count,9);XCTAssertEqual(rows[0],["x0","x1","y0"])
        XCTAssertEqual(Set(rows.dropFirst().map {$0[0]}).count,8)
        XCTAssertTrue(rows.dropFirst().flatMap {$0}.allSatisfy {Double($0)?.isFinite==true})
        let parent=try compile(GradientWorkflowExamples.document(task:.nextRun,device:"cpu",parentPreparation:true))
        XCTAssertEqual(parent.trainingMode,.fresh)
        XCTAssertEqual(parent.trainabilityRecipe?.configuration,try TrainabilityConfiguration(values:GradientLearningTask.keepSettings.trainabilityValues))
        XCTAssertEqual(parent.optimizerRecipe?.configuration.values["momentum"],.number(0.5))
        XCTAssertThrowsError(try GradientWorkflowExamples.document(task:.selectedInfluence,parentPreparation:true))
        XCTAssertThrowsError(try GradientWorkflowExamples.sourceCSV(widths:[9,9,9]))
    }
}
