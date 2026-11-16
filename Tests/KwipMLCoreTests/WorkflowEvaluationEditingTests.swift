import XCTest
@testable import KwipMLCore

final class WorkflowEvaluationEditingTests:XCTestCase {
    func testEvaluationEditsPreserveLabelsConfigurationAndTypedConnections()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let frozen=WorkflowNodeAddress(domain:.control,localID:"saved"),evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        var document=WorkflowDocument(projectID:UUID().uuidString)
        func edit(_ command:WorkflowCommand)throws {document=try WorkflowEditing.apply(command,to:document,modelRegistry:model,dataRegistry:data).document}
        try edit(.insert(.init(address:frozen,name:"Chosen model",payload:.frozenModel(config:[:]))))
        XCTAssertEqual(document.schemaVersion,5,"Explicit component insertion upgrades and persists in the same edit")
        try edit(.insert(.init(address:evaluate,name:"Final holdout",payload:.evaluateTest(config:[:]))))
        try edit(.configure(address:evaluate,config:["device":.string("cpu"),"batchSize":17,"previewLimit":3]))
        try edit(.connect(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model")))
        XCTAssertEqual(document.edges.first?.kind,.frozenModel)
        XCTAssertEqual(document.nodes[1].name,"Final holdout")
        XCTAssertEqual(document.nodes[1].payload,.evaluateTest(config:["device":.string("cpu"),"batchSize":17,"previewLimit":3]))
        let prior=document
        XCTAssertThrowsError(try edit(.configure(address:frozen,config:["path":.string("/private/model.pt")])))
        XCTAssertEqual(document,prior)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
    }
}
