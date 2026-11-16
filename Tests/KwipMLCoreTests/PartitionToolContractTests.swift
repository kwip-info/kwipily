import XCTest
@testable import KwipMLCore

final class PartitionToolContractTests:XCTestCase {
    private var root:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testPracticeCompilesAndAuditHasNoCanvasPort()throws {
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let definition=try XCTUnwrap(registry.operators.first{$0.id == "kwip.data.partition"})
        XCTAssertEqual(definition.authoringOutputs.map(\.id),["training","validation","test"])
        XCTAssertEqual(definition.outputs.last?.inspectionOnly,true)
        XCTAssertTrue(definition.authoringOutputs.allSatisfy {$0.kind == "table"})
        let document=try PartitionPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        _=try WorkflowDataPreparationPlan.compile(document,registry:registry)
        var invalid=document
        invalid.edges[1].from.port="exclusions"
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(invalid,registry:registry))
    }
}
