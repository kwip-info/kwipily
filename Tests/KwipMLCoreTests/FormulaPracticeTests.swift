import XCTest
@testable import KwipMLCore
final class FormulaPracticeTests:XCTestCase {
    func testPracticeRoundTripsCompilesAndCarriesFormulaThroughTrainingLineage()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try FormulaPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        let graph=try WorkflowDataPreparationPlan.compile(document,registry:registry)
        XCTAssertEqual(graph.nodes.filter{$0.operatorID == "kwip.data.formula"}.count,1)
        XCTAssertTrue(FileManager.default.fileExists(atPath:try FormulaPracticeExample.observationsURL().path))
        var training=try WorkflowExamples.preparedClassifier(projectID:UUID().uuidString,device:"cpu",model:nil)
        let address=WorkflowNodeAddress(domain:.data,localID:"formula")
        let target=try XCTUnwrap(training.nodes.first{$0.address.localID == "training"})
        let edgeIndex=try XCTUnwrap(training.edges.firstIndex{$0.to.address == target.address && $0.to.port == "table"})
        let parent=training.edges[edgeIndex].from
        let parentKind=training.edges[edgeIndex].kind
        training.nodes.append(.init(address:address,name:"Constant feature",payload:.dataOperator(operatorID:"kwip.data.formula",config:["operations":.array([.object(["id":.string("one"),"expression":.string("1"),"outputs":.array([.object(["name":.string("constant"),"dtype":.string("int64"),"write":.string("new")])]),"nullPolicy":.string("preserve")])]) ])))
        training.edges[edgeIndex].from = .init(address:address,port:"table")
        training.edges[edgeIndex].kind = .table
        training.edges.append(.init(from:parent,to:.init(address:address,port:"table"),kind:parentKind))
        _=try WorkflowCompiler.compile(training,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry)
    }
}
