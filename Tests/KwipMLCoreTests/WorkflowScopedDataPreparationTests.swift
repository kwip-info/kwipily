import XCTest
@testable import KwipMLCore
final class WorkflowScopedDataPreparationTests:XCTestCase {
 func testSelectedDecisionKeepsProviderWithoutTrainingOrDownstreamConsumers()throws {
  let original=try TypedDecisionPracticeExample.document()
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
  let plan=try WorkflowScopedDataPreparation(document:original,target:TypedDecisionPracticeExample.decision,registry:registry)
  XCTAssertTrue(plan.document.nodes.contains{$0.address==TypedDecisionPracticeExample.train})
  XCTAssertFalse(plan.document.edges.contains{$0.to.address==TypedDecisionPracticeExample.train})
  XCTAssertFalse(plan.document.nodes.contains{if case .saveModel=$0.payload{return true};return false})
  XCTAssertFalse(plan.document.nodes.contains{$0.address.localID=="decisions"})
  XCTAssertEqual(plan.document.revision,original.revision)
  XCTAssertTrue(plan.apiSources.isEmpty)
  XCTAssertThrowsError(try WorkflowScopedDataPreparation(document:original,target:TypedDecisionPracticeExample.train,registry:registry))
 }
 func testCapturedTargetPinsOnlyItsIncomingClosure()throws {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
  let train=WorkflowNodeAddress(domain:.control,localID:"train"),view=WorkflowNodeAddress(domain:.data,localID:"view"),other=WorkflowNodeAddress(domain:.data,localID:"other")
  let endpoint=WorkflowEndpoint(address:train,port:"predictions")
  let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:train,payload:.train(config:[:])),.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),.init(address:other,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:[:]))],edges:[.init(id:"view",from:endpoint,to:.init(address:view,port:"table"),kind:.table),.init(id:"other",from:endpoint,to:.init(address:other,port:"table"),kind:.table)])
  XCTAssertThrowsError(try WorkflowScopedDataPreparation(document:document,target:view,registry:registry)){XCTAssertTrue($0 is WorkflowCapturedTableRequirement)}
  let plan=try WorkflowScopedDataPreparation(document:document,target:view,registry:registry,capturedPin:{received in XCTAssertEqual(received,endpoint);return String(repeating:"a",count:64)})
  XCTAssertEqual(plan.document.nodes.count,2)
  XCTAssertFalse(plan.document.nodes.contains{$0.address==train || $0.address==other})
  XCTAssertTrue(plan.permissionDocument.nodes.contains{$0.address==train})
  XCTAssertFalse(plan.permissionDocument.nodes.contains{$0.address==other})
  XCTAssertEqual(Set(plan.capturedSources.values),[endpoint]);XCTAssertTrue(plan.apiSources.isEmpty)
 }
 func testWriteTargetIsExplicitAndInspectionCannotTraverseWrite()throws {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
  let graph=try DataFlowCodec.decodeGraph(Data(contentsOf:root.appendingPathComponent("runner/dataflow_examples/data02-output.json")))
  var document=WorkflowDocument(projectID:UUID().uuidString,name:"Scoped writes",nodes:graph.nodes.map{WorkflowNode(address:.init(domain:.data,localID:$0.id),payload:.dataOperator(operatorID:$0.operatorID,config:$0.config))},edges:graph.edges.enumerated().map{index,edge in WorkflowEdge(id:"e-\(index)",from:.init(address:.init(domain:.data,localID:edge.from.node),port:edge.from.port),to:.init(address:.init(domain:.data,localID:edge.to.node),port:edge.to.port),kind:.table)})
  let output=WorkflowNodeAddress(domain:.data,localID:"tool")
  XCTAssertEqual(try WorkflowScopedDataPreparation(document:document,target:output,registry:registry).document.nodes.count,3)
  XCTAssertEqual(try WorkflowScopedDataPreparation(document:document,target:.init(domain:.data,localID:"source"),registry:registry).document.nodes.count,1)
  let view=WorkflowNodeAddress(domain:.data,localID:"after-write")
  document.nodes.append(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
  document.edges.append(.init(id:"after-write",from:.init(address:output,port:"table"),to:.init(address:view,port:"table"),kind:.table))
  XCTAssertThrowsError(try WorkflowScopedDataPreparation(document:document,target:view,registry:registry)){error in XCTAssertTrue(error.localizedDescription.contains("upstream Table Output"))}
 }
}

extension WorkflowScopedDataPreparationTests {
 func testJoinRequiresBothConnectedCapturesButNeverSiblingProducerOrWrite()throws {
  let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
  let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
  let left=WorkflowNodeAddress(domain:.control,localID:"left"),right=WorkflowNodeAddress(domain:.control,localID:"right"),other=WorkflowNodeAddress(domain:.control,localID:"other")
  let invalid=WorkflowNodeAddress(domain:.data,localID:"missing-required-inputs")
  let join=WorkflowNodeAddress(domain:.data,localID:"join"),output=WorkflowNodeAddress(domain:.data,localID:"ungranted-sibling")
  let l=WorkflowEndpoint(address:left,port:"predictions"),r=WorkflowEndpoint(address:right,port:"predictions")
  let template=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")
  let joinPayload=try XCTUnwrap(template.nodes.first{$0.address.localID=="join"}).payload
  let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
   .init(address:left,payload:.train(config:[:])),.init(address:right,payload:.train(config:[:])),.init(address:other,payload:.train(config:[:])),
   .init(address:invalid,payload:joinPayload),.init(address:join,payload:joinPayload),.init(address:output,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:[:]))
  ],edges:[.init(id:"left",from:l,to:.init(address:join,port:"left"),kind:.table),.init(id:"right",from:r,to:.init(address:join,port:"right"),kind:.table),.init(id:"sibling",from:.init(address:other,port:"predictions"),to:.init(address:output,port:"table"),kind:.table)])
  XCTAssertThrowsError(try WorkflowScopedDataPreparation(document:document,target:invalid,registry:registry))
  for missing in [l,r] {
   XCTAssertThrowsError(try WorkflowScopedDataPreparation(document:document,target:join,registry:registry,capturedPin:{endpoint in
    XCTAssertNotEqual(endpoint.address,other)
    if endpoint==missing {throw WorkflowCapturedTableRequirement()}
    return String(repeating:"a",count:64)
   })){XCTAssertTrue($0 is WorkflowCapturedTableRequirement)}
  }
  var requested=Set<WorkflowEndpoint>()
  let plan=try WorkflowScopedDataPreparation(document:document,target:join,registry:registry,capturedPin:{endpoint in requested.insert(endpoint);return String(repeating:endpoint==l ? "a":"b",count:64)})
  XCTAssertEqual(requested,[l,r]);XCTAssertEqual(Set(plan.capturedSources.values),[l,r])
  XCTAssertFalse(plan.permissionDocument.nodes.contains{$0.address==other || $0.address==output})
  XCTAssertEqual(Set(plan.permissionDocument.nodes.map(\.address)),[left,right,join])
  XCTAssertFalse(plan.document.nodes.contains{$0.address.domain == .control})
 }
}
