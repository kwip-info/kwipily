import XCTest
@testable import KwipMLCore

final class SeparableConv2dBlockTests: XCTestCase {
    func testInspectableDefinitionExposesAllSettingsAndDerivesIntermediateChannels() throws {
        let definition = try SeparableConv2dBlock.definition()
        XCTAssertEqual(definition.body.nodes.map(\.type), ["depthwise_conv2d", "conv2d"])
        XCTAssertEqual(definition.body.nodes.map(\.id), ["depthwise", "pointwise"])
        XCTAssertEqual(definition.body.edges.count, 1)
        XCTAssertEqual(definition.parameters.count, 11)
        XCTAssertEqual(definition.inputs, [.init("in", targets: [.init("depthwise", "in")])])
        XCTAssertEqual(definition.outputs, [.init("out", source: .init("pointwise", "out"))])
        try CompositeDefinitionValidation.validate(definition, definitions: [definition], registry: .builtIn())
        let body = try CompositeConfigurationResolver.resolve(definition,
            overrides: ["channel_multiplier": 3, "depthwise_bias": false, "pointwise_bias": true], definitions: [], registry: .builtIn())
        XCTAssertEqual(body.nodes[0].config["channel_multiplier"], 3)
        XCTAssertEqual(body.nodes[1].config["in_channels"], 6)
        XCTAssertEqual(body.nodes[1].config["kernel_size"], .array([1, 1]))
        XCTAssertEqual(body.nodes[1].config["groups"], 1)
        XCTAssertEqual(body.nodes[0].config["bias"], false)
        XCTAssertEqual(body.nodes[1].config["bias"], true)
        XCTAssertEqual(definition.body.nodes[1].config["in_channels"], 4)
    }
    func testExpansionKeepsIndependentStageOriginsAndStableIDsThroughMultiplierEdits() throws {
        let source = try CompositeExamples.separableConv2dClassifier()
        let before = try CompositeExpander.expand(source, registry: .builtIn())
        XCTAssertTrue(GraphValidator.validate(before.graph, registry: try .builtIn()).isValid)
        if let path = ProcessInfo.processInfo.environment["KWIPML_SEPARABLE_FIXTURE_DIR"] {
            let directory = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(before.graph).write(to: directory.appendingPathComponent("graph.json"), options: .atomic)
            try encoder.encode(CompositeSourceSnapshot(project: source, expansion: before)).write(to: directory.appendingPathComponent("source.json"), options: .atomic)
        }
        let children = before.nodeOrigins.filter { $0.value.instancePath == ["convolution"] }
        XCTAssertEqual(Set(children.values.map(\.localNodeID)), ["depthwise", "pointwise"])
        var edited = source; edited.root.instances[0].overrides["channel_multiplier"] = 3
        let after = try CompositeExpander.expand(edited, registry: .builtIn())
        XCTAssertEqual(Set(before.nodeOrigins.keys), Set(after.nodeOrigins.keys))
        let pointID = try XCTUnwrap(children.first { $0.value.localNodeID == "pointwise" }?.key)
        XCTAssertEqual(after.graph.nodes.first { $0.id == pointID }?.config["in_channels"], 6)
        XCTAssertEqual(after.graph.nodes.first { $0.id == pointID }?.config["out_channels"], 4)
    }
    func testPreparedWorkflowPreservesBlockAndRoundTripsAllFieldMetadata() throws {
        let workflow = try WorkflowExamples.preparedSeparableConv2dClassifier(projectID: "11111111-1111-1111-1111-111111111111", device: "cpu")
        XCTAssertEqual(workflow.definitions.count, 1)
        XCTAssertEqual(workflow.nodes.count, 18)
        let node = try XCTUnwrap(workflow.nodes.first { $0.address == .init(domain: .model, localID: "convolution") })
        guard case .modelBlock(let reference, _) = node.payload else { return XCTFail("Expected inspectable Block, not flattened authoring") }
        XCTAssertEqual(reference, workflow.definitions[0].reference)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)), workflow)
    }
}
