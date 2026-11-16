import XCTest
@testable import KwipMLCore
final class TypedDecisionConfigurationTests:XCTestCase {
 func testStableQuestionsAndNoSilentDuplicateOptions()throws {
  var set=TypedDecisionConfiguration.defaultQuestionSet
  try TypedDecisionConfiguration.validateQuestionSet(set)
  set["revision"] = .bool(true)
  XCTAssertThrowsError(try TypedDecisionConfiguration.validateQuestionSet(set))
 }
 func testTaskMatchedLocalAndHubFileAdmission()throws {
  XCTAssertTrue(ModelHubRequest.allowedAssetFile("encoder/config.json"))
  XCTAssertTrue(ModelHubRequest.allowedAssetFile("tokenizer/tokenizer.json"))
  XCTAssertTrue(ModelHubRequest.allowedFile("typed-decisions/encoder/config.json"))
  for name in ["../../tokenizer.json","encoder/model.py","encoder/../config.json","something/encoder/config.json"] {XCTAssertFalse(ModelHubRequest.allowedAssetFile(name))}
 }
 func testDecideConfigurationRefusesInvalidStateAndKeepsAllScores()throws {
  var config=TypedDecisionConfiguration.defaults
  try TypedDecisionConfiguration.validate(config)
  config["stateColumns"] = .array([.string("text"),.string("text")])
  XCTAssertThrowsError(try TypedDecisionConfiguration.validate(config))
 }
 func testPracticeUsesRealTrainAndStandardViewsAndPreservesRoundtrip()throws {
  let document=try TypedDecisionPracticeExample.document()
  let encoded=try JSONEncoder().encode(document)
  XCTAssertEqual(try JSONDecoder().decode(WorkflowDocument.self,from:encoded),document)
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
  XCTAssertNoThrow(try WorkflowFineTuningPlan.compile(document,train:TypedDecisionPracticeExample.train,dataRegistry:registry))
  let plan=try WorkflowDecisionPreparation(document:document,consumer:TypedDecisionPracticeExample.decision)
  XCTAssertNoThrow(try WorkflowDataPreparationPlan.compile(plan.document,registry:registry,allowTrainingTerminals:true))
  XCTAssertThrowsError(try WorkflowCodePlan.compile(document,modelRegistry:.builtIn(),dataRegistry:registry),"Unqualified portable Code must not substitute causal training or omit Decide")
  XCTAssertEqual(document.nodes.filter{if case .dataOperator(let id,_)=($0.payload){return id=="kwip.data.view"};return false}.count,2)
 }

}
