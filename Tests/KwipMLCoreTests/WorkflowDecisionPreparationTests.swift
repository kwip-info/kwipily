import XCTest
@testable import KwipMLCore
final class WorkflowDecisionPreparationTests:XCTestCase {
 func testDecisionBranchPreservesModelPinEdgeButNeverTrainingInputs()throws {
  let document=try TypedDecisionPracticeExample.document(),train=TypedDecisionPracticeExample.train,decision=TypedDecisionPracticeExample.decision,view=WorkflowNodeAddress(domain:.data,localID:"decisions")
  let plan=try WorkflowDecisionPreparation(document:document,consumer:decision)
  XCTAssertTrue(plan.document.nodes.contains {$0.address==train})
  XCTAssertFalse(plan.document.edges.contains {$0.to.address==train})
  XCTAssertFalse(plan.document.nodes.contains {if case .saveModel=$0.payload{return true};return false})
  XCTAssertEqual(WorkflowDecisionPreparation.consumer(in:document,for:view),decision)
  XCTAssertTrue(try WorkflowTrainedModelSource.resolve(in:document,consumer:decision).isTrain)
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
  XCTAssertNoThrow(try WorkflowDataPreparationPlan.compile(plan.document,registry:registry,allowTrainingTerminals:true))
  XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(plan.document,registry:registry))
 }
}
