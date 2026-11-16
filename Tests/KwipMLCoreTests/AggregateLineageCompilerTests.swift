import XCTest
@testable import KwipMLCore

final class AggregateLineageCompilerTests:XCTestCase {
    private let aggregate=WorkflowNodeAddress(domain:.data,localID:"summary")
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func document()throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")
        let index=try XCTUnwrap(document.edges.firstIndex {$0.from.address.localID == "join" && $0.to.address.localID == "cast"})
        let edge=document.edges.remove(at:index)
        document.nodes.append(.init(address:aggregate,payload:.dataOperator(operatorID:"kwip.data.aggregate",config:["groupBy":.array([.string("label")]),"nullKeys":.string("error"),"maxRows":.integer(1000),"aggregations":.array([.object(["op":.string("mean"),"column":.string("signal"),"nullPolicy":.string("error"),"output":.string("signal")]),.object(["op":.string("mean"),"column":.string("context"),"nullPolicy":.string("error"),"output":.string("context")])])])))
        document.edges.append(.init(id:"joined-to-summary",from:edge.from,to:.init(address:aggregate,port:"table"),kind:.table))
        document.edges.append(.init(id:"summary-to-cast",from:.init(address:aggregate,port:"table"),to:edge.to,kind:.table))
        return document
    }
    func testAggregationCreatesPopulationOriginWhileCapturingEveryPhysicalSource()throws {
        let plan=try compile(document())
        XCTAssertEqual(Set(plan.sourceAddresses),[WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress])
        XCTAssertTrue(plan.dataflow.nodes.contains {$0.id == "summary" && $0.operatorID == "kwip.data.aggregate"})
    }
    func testPartitionedRowsCannotBeRegroupedBeforeTraining()throws {
        for role in ["training","validation","test"] {
            var value=try document()
            let i=try XCTUnwrap(value.edges.firstIndex {$0.to.address == aggregate})
            let edge=value.edges.remove(at:i),split=WorkflowNodeAddress(domain:.data,localID:"early-split")
            value.nodes.append(.init(address:split,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25])))
            value.edges.append(.init(id:"to-early-split",from:edge.from,to:.init(address:split,port:"table"),kind:.table))
            value.edges.append(.init(id:"partition-to-summary",from:.init(address:split,port:role),to:edge.to,kind:.table))
            XCTAssertThrowsError(try compile(value)) {error in XCTAssertTrue(error.localizedDescription.contains("before splitting"),error.localizedDescription)}
        }
    }
}
