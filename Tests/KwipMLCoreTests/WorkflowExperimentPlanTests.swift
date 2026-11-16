import XCTest
@testable import KwipMLCore

final class WorkflowExperimentPlanTests:XCTestCase {
    private func build(_ axes:[WorkflowExperimentAxis], runs:Int=32, epochs:Int=1000, source:WorkflowDocument?=nil)throws->WorkflowExperimentPlan {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowExperimentPlan.build(source ?? WorkflowExamples.preparedParallelClassifier(device:"cpu"),train:WorkflowExamples.trainAddress,
            axes:axes,budget:.init(maximumRuns:runs,maximumTotalEpochs:epochs),modelRegistry:.builtIn(),
            dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testCartesianOrderAndIsolationPreserveDeclaredDataAndModel()throws {
        let source=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),original=source
        let plan=try build([.init(field:"seed",values:[7,8]),.init(field:"batchSize",values:[16,32])],source:source)
        XCTAssertEqual(plan.variants.map {[$0.execution.settings.seed,$0.execution.settings.batchSize]},[[7,16],[7,32],[8,16],[8,32]])
        XCTAssertEqual(plan.totalEpochs,12)
        XCTAssertEqual(source,original)
        XCTAssertTrue(plan.variants.allSatisfy {$0.execution.dataflow == plan.variants[0].execution.dataflow})
        XCTAssertTrue(plan.variants.allSatisfy {$0.execution.model.semanticSHA256 == plan.variants[0].execution.model.semanticSHA256})
        XCTAssertEqual(plan.variants.map(\.ordinal),[1,2,3,4])
        XCTAssertEqual(plan.variants[3].overrides,["seed":8,"batchSize":32])
        XCTAssertEqual(try build(plan.axes,source:source),plan)
    }
    func testInvalidDimensionsAndBudgetsRejectRatherThanTruncate()throws {
        for axes:[WorkflowExperimentAxis] in [[],[.init(field:"seed",values:[])],[.init(field:"unknown",values:[1])],
            [.init(field:"seed",values:[1,1])],[.init(field:"seed",values:[1]),.init(field:"seed",values:[2])],
            [.init(field:"batchSize",values:[0])],[.init(field:"seed",values:[.number(1.5)])],
            [.init(field:"learningRate",values:[1,.number(1.0)])]] {
            XCTAssertThrowsError(try build(axes))
        }
        let axes:[WorkflowExperimentAxis]=[.init(field:"seed",values:[1,2]),.init(field:"epochs",values:[3,4])]
        XCTAssertThrowsError(try build(axes,runs:3))
        XCTAssertThrowsError(try build(axes,epochs:13))
        XCTAssertEqual(try build(axes,epochs:14).totalEpochs,14)
        XCTAssertThrowsError(try build(axes,runs:33))
    }
    func testContinuationIsNotSilentlyConvertedToFreshTraining()throws {
        var source=try WorkflowCodec.migrateToVersion3(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        let index=try XCTUnwrap(source.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        source.nodes[index].payload = .train(config:["trainingMode":.string("exactResume")])
        XCTAssertThrowsError(try build([.init(field:"seed",values:[1,2])],source:source))
    }
}
