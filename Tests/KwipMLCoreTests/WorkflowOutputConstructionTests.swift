import XCTest
@testable import KwipMLCore
final class WorkflowOutputConstructionTests:XCTestCase {
    func testNewPredictionBranchPreservesProviderButUpstreamEditsDoNot()throws {
        let source=try WorkflowModelBoundary.consolidate(WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu")))
        var current=source
        let view=WorkflowNodeAddress(domain:.data,localID:"new-view"),train=WorkflowExamples.trainAddress
        current.nodes.append(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
        current.edges.append(.init(id:"predictions-view",from:.init(address:train,port:"predictions"),to:.init(address:view,port:"table"),kind:.table));current.revision+=1
        XCTAssertTrue(WorkflowOutputConstruction.matches(current,captured:source,provider:train))
        let index=try XCTUnwrap(current.nodes.firstIndex {$0.address==train})
        current.nodes[index].payload = .train(config:["runSetup":.object(["epochs":99])])
        XCTAssertFalse(WorkflowOutputConstruction.matches(current,captured:source,provider:train))
        current=source;current.edges.removeAll {$0.to.address==train && $0.to.port=="model"}
        XCTAssertFalse(WorkflowOutputConstruction.matches(current,captured:source,provider:train))
    }
}
