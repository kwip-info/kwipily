import Foundation

/// An authored specialization of Conv2d; each input channel has independent filters.
public struct DepthwiseConv2dConfiguration: Equatable, Sendable {
    public let inputChannels: Int
    public let channelMultiplier: Int
    public let convolution: Conv2dConfiguration
    public var parameterCount: Int { convolution.parameterCount }

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["in_channels", "channel_multiplier", "kernel_size", "stride", "dilation", "bias", "padding_policy", "padding", "padding_mode"],
              let channels = values["in_channels"]?.integerValue, (1...512).contains(channels),
              let multiplier = values["channel_multiplier"]?.integerValue, (1...512).contains(multiplier),
              channels * multiplier <= 512 else {
            throw GraphConfigurationError("DepthwiseConv2d requires its nine authored settings, input channels and channel multiplier from 1–512, and at most 512 derived output channels. Groups and output channels are derived, not editable fields.")
        }
        var derived = values
        derived.removeValue(forKey: "channel_multiplier")
        derived["in_channels"] = .integer(Int64(channels))
        derived["groups"] = .integer(Int64(channels))
        derived["out_channels"] = .integer(Int64(channels * multiplier))
        convolution = try Conv2dConfiguration(values: derived)
        inputChannels = channels; channelMultiplier = multiplier
    }
    public var values: [String: JSONValue] {
        var authored = convolution.values
        authored.removeValue(forKey: "groups")
        authored.removeValue(forKey: "out_channels")
        authored["channel_multiplier"] = .integer(Int64(channelMultiplier))
        return authored
    }
    public func resolved(input: BatchedTensorLayout) throws -> Conv2dConfiguration.Resolved {
        try convolution.resolved(input: input)
    }
}
