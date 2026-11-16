import XCTest
@testable import KwipMLCore

final class WorkflowPredictionPreparationTests:XCTestCase {
    func testBranchesCaptureFullPredictionsWithoutCopyingTrainingOrItsControls() throws {
        var document=try WorkflowExamples.preparedEmbeddingClassifier()
        let train=WorkflowNodeAddress(domain:.control,localID:"train"),view=WorkflowNodeAddress(domain:.data,localID:"predictions-view")
        document.nodes.append(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
        let endpoint=WorkflowEndpoint(address:train,port:"predictions")
        document.edges.append(.init(id:"inspect-predictions",from:endpoint,to:.init(address:view,port:"table"),kind:.table))
        let plan=try WorkflowPredictionPreparation(document:document,pins:[endpoint:String(repeating:"a",count:64)])
        XCTAssertEqual(plan.document.nodes.count,2)
        XCTAssertTrue(plan.document.nodes.allSatisfy {$0.address.domain == .data})
        XCTAssertEqual(plan.sources.values.first,endpoint)
        XCTAssertEqual(plan.document.edges.first?.to.address,view)
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        XCTAssertNoThrow(try WorkflowDataPreparationPlan.compile(plan.document,registry:data))
        XCTAssertNoThrow(try WorkflowCompiler.compile(document,train:train,modelRegistry:.builtIn(),dataRegistry:data,contractVersion:15))
        XCTAssertThrowsError(try WorkflowPredictionPreparation(document:document,pins:[:]),"Missing completed evidence never implies retraining or a bounded preview.")
        let ports=try WorkflowPortContracts.resolve(document.nodes.first {$0.address==train}!,document:document,modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(ports.outputs.first {$0.id=="predictions"}?.kind,.table)
    }
}
