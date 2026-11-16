import Foundation

/// A transparent composition: spatial filtering followed by learned channel mixing.
public enum SeparableConv2dBlock {
    public static let reference = CompositeDefinitionReference("kwip.separable-conv2d")
    public static let defaultValues: [String: JSONValue] = [
        "in_channels": 2, "channel_multiplier": 2, "out_channels": 4,
        "kernel_size": .array([2, 3]), "stride": .array([1, 1]), "dilation": .array([1, 1]),
        "depthwise_bias": true, "pointwise_bias": true, "padding_policy": .string("same"),
        "padding": .array([0, 0]), "padding_mode": .string("zeros")]

    public static func definition(values: [String: JSONValue] = defaultValues,
                                  reference: CompositeDefinitionReference = reference) throws -> CompositeDefinition {
        let config = try SeparableConv2dConfiguration(values: values)
        var parameters = DepthwiseConv2dComponent.definition.fields.map { original -> CompositeParameter in
            var field = original
            if field.key == "bias" { field.key = "depthwise_bias"; field.title = "Depthwise bias" }
            field.defaultValue = config.values[field.key]!
            return .init(field, targets: [.init("depthwise", original.key)])
        }
        for key in ["out_channels", "bias"] {
            guard var field = Conv2dComponent.definition.fields.first(where: { $0.key == key }) else {
                throw GraphConfigurationError("The pointwise convolution contract is unavailable.")
            }
            if key == "bias" { field.key = "pointwise_bias"; field.title = "Pointwise bias" }
            field.defaultValue = config.values[field.key]!
            parameters.append(.init(field, targets: [.init("pointwise", key)]))
        }
        return .init(reference: reference, title: "Separable Conv2d", body: .init(nodes: [
            .init(id: "depthwise", type: "depthwise_conv2d", name: "Independent spatial filters", x: 80, y: 160, config: config.depthwise.values),
            .init(id: "pointwise", type: "conv2d", name: "Mix channels at each position", x: 370, y: 160, config: config.pointwise.values)
        ], edges: [.init(id: "spatial-to-channel", source: "depthwise", target: "pointwise")]),
        inputs: [.init("in", targets: [.init("depthwise", "in")])],
        outputs: [.init("out", source: .init("pointwise", "out"))], parameters: parameters,
        integerProducts: [.init(lhs: "in_channels", rhs: "channel_multiplier", targets: [.init("pointwise", "in_channels")])])
    }
}

extension CompositeExamples {
    public static func separableConv2dClassifier() throws -> CompositeProjectDocument {
        let original = ExampleFlows.depthwiseConv2dClassifier()
        let definition = try SeparableConv2dBlock.definition()
        return .init(name: "Separable constructed-grid classifier", root: .init(
            nodes: original.nodes.filter { $0.id != "convolution" },
            instances: [.init(id: "convolution", definition: definition.reference, name: "Spatial filtering and channel mixing", x: 730, y: 380)],
            edges: original.edges), definitions: [definition], execution: original.execution)
    }
}

extension WorkflowExamples {
    public static func preparedSeparableConv2dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        var workflow = try preparedDepthwiseConv2dClassifier(projectID: projectID, device: device)
        let definition = try SeparableConv2dBlock.definition()
        workflow.name = "Prepare and train separable constructed-grid classifier"
        workflow.definitions = [definition]
        guard let index = workflow.nodes.firstIndex(where: { $0.address == .init(domain: .model, localID: "convolution") }) else {
            throw GraphConfigurationError("The prepared spatial example has no convolution position.")
        }
        workflow.nodes[index].name = "Spatial filtering and channel mixing"
        workflow.nodes[index].payload = .modelBlock(reference: definition.reference, overrides: [:])
        // New examples reserve the expanded two-stage container footprint. Saved
        // projects retain their authored positions; opening them never relayouts source.
        let containerX=workflow.nodes[index].x
        for nodeIndex in workflow.nodes.indices where workflow.nodes[nodeIndex].address.domain == .model && workflow.nodes[nodeIndex].x>containerX {
            workflow.nodes[nodeIndex].x+=1000
        }
        if let index = workflow.nodes.firstIndex(where: { $0.address == .init(domain: .control, localID: "model") }) {
            workflow.nodes[index].name = "Separable constructed-grid classifier architecture"
        }
        return workflow
    }
}
