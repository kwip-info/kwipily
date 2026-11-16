import XCTest
@testable import KwipMLCore

final class EmbeddingExampleTests:XCTestCase {
    func testNewExampleUsesCompactNonoverlappingLanes() throws {
        let document = try WorkflowExamples.preparedEmbeddingClassifier()
        XCTAssertLessThanOrEqual(document.nodes.map(\.x).max() ?? 0, 1500)
        XCTAssertLessThanOrEqual(document.nodes.map(\.y).max() ?? 0, 400)
        for (index, node) in document.nodes.enumerated() {
            for other in document.nodes.dropFirst(index + 1) {
                XCTAssertTrue(abs(node.x - other.x) >= 200 || abs(node.y - other.y) >= 130,
                    "Example cards overlap: \(node.name), \(other.name)")
            }
        }
        let byAddress = Dictionary(uniqueKeysWithValues: document.nodes.map { ($0.address, $0) })
        for edge in document.edges {
            let from = try XCTUnwrap(byAddress[edge.from.address])
            let to = try XCTUnwrap(byAddress[edge.to.address])
            XCTAssertLessThan(from.x, to.x, "Example connections should progress left to right")
        }
    }
    func testPreparedLookupUsesExactIntegerColumnsWithoutScaling()throws {
        let registry=try ComponentRegistry.builtIn()
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let document=try WorkflowExamples.preparedEmbeddingClassifier(device:"cpu")
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.model.graph.nodes.map(\.type).sorted(),["index_input","embedding","flatten","linear","output"].sorted())
        XCTAssertEqual(try WorkflowMappingRequirements.inputContract(plan.model.graph,registry:registry).dtype,.int64)
        XCTAssertFalse(document.nodes.contains {node in
            if case .dataOperator(let id,_)=node.payload {return id.contains("Scaler") || id.hasSuffix("transform")};return false
        })
        let cast=try XCTUnwrap(document.nodes.first {$0.address.localID == "cast"})
        guard case .dataOperator(let operatorID,let configuration)=cast.payload else {return XCTFail("Missing Columns")}
        XCTAssertEqual(operatorID,"kwip.data.columns")
        guard case .array(let rows)=configuration["rows"] else {return XCTFail("Missing ordered column rows")}
        let expected:[JSONValue]=["signal","context","label"].map {name in
            .object(["source":.string(name),"included":.bool(true),"outputName":.string(name),"targetType":.string("int64")])
        }
        XCTAssertEqual(rows,expected)
        XCTAssertEqual(configuration["castFailure"],.string("error"))
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
    }
}
