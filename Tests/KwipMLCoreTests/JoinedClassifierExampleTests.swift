import XCTest
@testable import KwipMLCore

final class JoinedClassifierExampleTests:XCTestCase {
    func testJoinedExampleCompilesTwoSourcesBeforePartitionAndRoundTrips()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let document=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(Set(plan.sourceAddresses),[WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress])
        XCTAssertEqual(plan.featureColumns,["signal","context"])
        XCTAssertEqual(plan.targetColumn,"label")
        XCTAssertEqual(plan.settings.epochs,3)
        XCTAssertEqual(plan.settings.device,"cpu")
        let join=try XCTUnwrap(document.nodes.first {$0.address.localID == "join"})
        guard case .dataOperator("kwip.data.join",let config)=join.payload else {return XCTFail("Expected join")}
        XCTAssertEqual(config["cardinality"],.string("oneToOne"))
        XCTAssertEqual(config["leftKeys"],.array([.string("observation_id")]))
        XCTAssertEqual(Set(document.edges.filter {$0.to.address == join.address}.map(\.from.address)),Set(plan.sourceAddresses))
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        XCTAssertEqual(PreparedWorkflowExample.joined.sourceResources.count,2)
    }
}
