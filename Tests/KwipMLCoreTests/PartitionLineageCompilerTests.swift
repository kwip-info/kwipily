import XCTest
@testable import KwipMLCore

final class PartitionLineageCompilerTests:XCTestCase {
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func document(_ type:String)throws->WorkflowDocument {
        var d=try WorkflowExamples.preparedAggregatedClassifier(device:"cpu")
        let index=try XCTUnwrap(d.nodes.firstIndex {$0.address.localID == "split"})
        let config:[String:JSONValue]=type == "entity" ? ["strategy":.string("entity"),"groupBy":.array([.string("subject_id")]),"seed":.integer(42),"validationPercent":.integer(20),"testPercent":.integer(20)] : ["strategy":.string("time"),"timeColumn":.string("observed_at"),"timeEncoding":.string("epochMilliseconds"),"trainingEnd":.integer(1000),"validationEnd":.integer(2000),"embargoMilliseconds":.integer(0),"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")]
        d.nodes[index].payload = .dataOperator(operatorID:"kwip.data.partition",config:type == "random" ? PartitionPracticeExample.configuration:config)
        for i in d.edges.indices where d.edges[i].from.address == d.nodes[index].address {d.edges[i].kind = .table}
        return d
    }
    func testNewSplitRolesCompileWithoutIncludingTestInTrainingMapping()throws {
        for type in ["entity","time","random"] {
            let plan=try compile(document(type))
            XCTAssertTrue(plan.dataflow.nodes.contains {$0.operatorID == "kwip.data.partition"})
            XCTAssertEqual(plan.sourceAddresses,[WorkflowExamples.sourceAddress])
        }
    }
    func testTestCannotBeUsedForFittingOrValidationSelection()throws {
        for type in ["entity","time","random"] {
            for target in ["training","validation"] {
                var d=try document(type)
                let index=try XCTUnwrap(d.edges.firstIndex {$0.from.address.localID == "split" && $0.to.address.localID == "fit" && $0.to.port == target})
                d.edges[index].from.port="test"
                XCTAssertThrowsError(try compile(d)) {error in
                    let message=error.localizedDescription
                    XCTAssertFalse(message.contains("no reviewed"),message)
                    XCTAssertTrue(message.contains("training") || message.contains("Training") || message.contains("partition roles"),message)
                }
            }
        }
    }
}
