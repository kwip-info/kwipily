import XCTest
@testable import KwipMLCore

final class AggregatedClassifierExampleTests:XCTestCase {
    func testRepeatedMeasurementExampleCompilesBeforeSplitAndKeepsLabelsOutOfGrouping()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let document=try WorkflowExamples.preparedAggregatedClassifier(device:"cpu")
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.sourceAddresses,[WorkflowExamples.sourceAddress])
        XCTAssertEqual(plan.featureColumns,["signal","context"])
        XCTAssertEqual(plan.targetColumn,"label")
        XCTAssertEqual(plan.settings.epochs,3)
        let aggregate=try XCTUnwrap(document.nodes.first {$0.address.localID == "aggregate"})
        guard case .dataOperator("kwip.data.aggregate",let config)=aggregate.payload else {return XCTFail("Expected aggregation")}
        XCTAssertEqual(config["groupBy"],.array([.string("subject_id")]))
        XCTAssertEqual(config["nullKeys"],.string("error"))
        guard case .array(let reductions)=config["aggregations"] else {return XCTFail("Expected reducers")}
        XCTAssertEqual(reductions.count,5)
        XCTAssertEqual(document.edges.first {$0.to.address == aggregate.address}?.from.address.localID,"cast")
        XCTAssertEqual(document.edges.first {$0.from.address == aggregate.address}?.to.address.localID,"split")
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        XCTAssertEqual(PreparedWorkflowExample.aggregated.sourceResources.first?.resource,"repeated-measurements")
    }
}
