import Foundation

/// Independent spatial filters per input channel; groups and output channels are derived.
public enum DepthwiseConv2dComponent {
    public static let definition = ComponentDefinition(
        id: "depthwise_conv2d", title: "Depthwise Conv2d", category: "Layers", inputs: ["in"], outputs: ["out"], fields: [
            .init(key: "in_channels", title: "Input channels", kind: .integer, defaultValue: 2, min: 1, max: 512),
            .init(key: "channel_multiplier", title: "Filters per input channel", kind: .integer, defaultValue: 1, min: 1, max: 512),
            .init(key: "kernel_size", title: "Kernel [height, width]", kind: .integerList, defaultValue: .array([3, 3]), min: 1, max: 4096, minItems: 2, maxItems: 2),
            .init(key: "stride", title: "Stride [height, width]", kind: .integerList, defaultValue: .array([1, 1]), min: 1, max: 4096, minItems: 2, maxItems: 2),
            .init(key: "dilation", title: "Dilation [height, width]", kind: .integerList, defaultValue: .array([1, 1]), min: 1, max: 4096, minItems: 2, maxItems: 2),
            .init(key: "bias", title: "Learn bias", kind: .boolean, defaultValue: true, min: nil, max: nil),
            .init(key: "padding_policy", title: "Padding policy", kind: .choice, defaultValue: .string("explicit"), min: nil, max: nil,
                choices: [.init(value: "explicit", title: "Explicit amount"), .init(value: "valid", title: "Valid · no padding"), .init(value: "same", title: "Same spatial size")]),
            .init(key: "padding", title: "Padding [height, width]", kind: .integerList, defaultValue: .array([0, 0]), min: 0, max: 4096, minItems: 2, maxItems: 2),
            .init(key: "padding_mode", title: "Padding mode", kind: .choice, defaultValue: .string("zeros"), min: nil, max: nil,
                choices: [.init(value: "zeros", title: "Zeros"), .init(value: "reflect", title: "Reflect"), .init(value: "replicate", title: "Replicate"), .init(value: "circular", title: "Circular")])
        ])
}
