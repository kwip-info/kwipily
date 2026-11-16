import XCTest
@testable import KwipMLCore

final class WorkflowBlockInsertionTests: XCTestCase {
    private var definition: CompositeDefinition { CompositeExamples.parallelClassifier.definitions[0] }
    private func node(_ id: String = "block") -> WorkflowNode {
        .init(address: .init(domain: .model, localID: id), name: "My feature block", x: 40, y: 80,
              payload: .modelBlock(reference: definition.reference, overrides: ["branchWidth": 4]))
    }
    private func apply(_ command: WorkflowCommand, to document: WorkflowDocument) throws -> WorkflowEditResult {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowEditing.apply(command, to: document, modelRegistry: .builtIn(),
            dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testAtomicInsertionAndIdenticalPinReusePreserveOriginalBaseline() throws {
        let baseline = WorkflowDocument(projectID: UUID().uuidString)
        let result = try apply(.insertBlock(node(), definition: definition), to: baseline)
        XCTAssertEqual(result.document.nodes, [node()])
        XCTAssertEqual(result.document.definitions, [definition])
        XCTAssertEqual(result.document.revision, baseline.revision + 1)
        XCTAssertEqual(result.selection.nodeAddresses, [node().address])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(result.document)), result.document)
        let second = try apply(.insertBlock(node("second"), definition: definition), to: result.document)
        XCTAssertEqual(second.document.definitions, [definition])
        XCTAssertEqual(second.document.nodes.count, 2)
        XCTAssertTrue(baseline.nodes.isEmpty); XCTAssertTrue(baseline.definitions.isEmpty)
    }
    func testInvalidPinPayloadAndPositionRefuseEntireTransaction() throws {
        let baseline = WorkflowDocument(projectID: UUID().uuidString, definitions: [definition])
        var different = definition; different.title = "Conflicting pin"
        XCTAssertThrowsError(try apply(.insertBlock(node(), definition: different), to: baseline))
        var invalid = node(); invalid.payload = .modelBlock(reference: .init("other"), overrides: [:])
        XCTAssertThrowsError(try apply(.insertBlock(invalid, definition: definition), to: baseline))
        invalid = node(); invalid.payload = .modelPrimitive(type: "linear", configurationVersion: 2, config: [:])
        XCTAssertThrowsError(try apply(.insertBlock(invalid, definition: definition), to: baseline))
        invalid = node(); invalid.x = .infinity
        XCTAssertThrowsError(try apply(.insertBlock(invalid, definition: definition), to: baseline))
        var broken = definition; broken.outputs[0].source.componentID = "missing"
        XCTAssertThrowsError(try apply(.insertBlock(node(), definition: broken), to: WorkflowDocument(projectID: UUID().uuidString)))
        XCTAssertTrue(baseline.nodes.isEmpty); XCTAssertEqual(baseline.definitions, [definition])
    }
    func testNodeAndDefinitionBoundsAndDuplicateAddressAreAtomic() throws {
        let original = WorkflowDocument(projectID: UUID().uuidString)
        let inserted = try apply(.insertBlock(node(), definition: definition), to: original).document
        XCTAssertThrowsError(try apply(.insertBlock(node(), definition: definition), to: inserted))
        var full = original
        full.definitions = (0..<64).map { index in
            var copy = definition; copy.reference = .init("pin-\(index)"); return copy
        }
        XCTAssertThrowsError(try apply(.insertBlock(node(), definition: definition), to: full))
        XCTAssertEqual(full.definitions.count, 64); XCTAssertTrue(full.nodes.isEmpty)
        var modelFull = original
        modelFull.nodes = (0..<128).map { .init(address: .init(domain: .model, localID: "node-\($0)"),
            payload: .modelPrimitive(type: "relu", configurationVersion: 2, config: [:])) }
        XCTAssertThrowsError(try apply(.insertBlock(node(), definition: definition), to: modelFull))
        XCTAssertTrue(modelFull.definitions.isEmpty)
    }
}
