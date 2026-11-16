import XCTest
@testable import KwipMLCore

final class JoinLineageCompilerTests:XCTestCase {
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testJoinedPopulationKeepsBothPhysicalSourcesAndExcludesUnconnectedSources()throws {
        var document=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")
        document.nodes.append(.init(address:.init(domain:.data,localID:"unrelated"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])))
        let plan=try compile(document)
        XCTAssertEqual(Set(plan.sourceAddresses),[WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress])
        XCTAssertFalse(plan.dataflow.nodes.contains {$0.id == "unrelated"})
    }
    func testJoinRejectsPartitionedInputOnEitherSide()throws {
        for side in ["left","right"] {
            for role in ["training","validation","test"] {
                var document=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")
                let index=try XCTUnwrap(document.edges.firstIndex {$0.to.address.localID == "join" && $0.to.port == side})
                let incoming=document.edges.remove(at:index)
                let split=WorkflowNodeAddress(domain:.data,localID:"premature-split")
                document.nodes.append(.init(address:split,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25])))
                document.edges.append(.init(id:"source-to-early-split",from:incoming.from,to:.init(address:split,port:"table"),kind:.table))
                document.edges.append(.init(id:"early-partition-to-join",from:.init(address:split,port:role),to:incoming.to,kind:.table))
                XCTAssertThrowsError(try compile(document)) { error in
                    XCTAssertTrue(error.localizedDescription.contains("before splitting"),error.localizedDescription)
                }
            }
        }
    }
}
