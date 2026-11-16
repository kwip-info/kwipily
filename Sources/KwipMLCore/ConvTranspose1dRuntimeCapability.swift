import Foundation

/// Exact channels-first convolution and preallocation contract.
public struct ConvTranspose1dRuntimeCapability: Equatable {
    private init() {}
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096 else { throw GraphConfigurationError("ConvTranspose1d capability exceeds 4096 bytes.") }
        let expected: JSONValue = .object([
            "version": .object(["major": 1, "minor": 0, "patch": 0]), "adapterID": .string("kwip.runner.graph-v2.conv_transpose1d"),
            "graphSchemaVersions": .array([2]), "sampleRank": 2, "preservesBatch": true, "graphDTypes": .array([.string("float32")]),
            "spatialParameterKind": .string("scalar"), "maximumSampleElements": 100_000_000, "maximumUncroppedSampleElements": 100_000_000,
            "channelMinimum": 1, "channelMaximum": 512, "lengthMinimum": 1, "lengthMaximum": 4096,
            "kernelMinimum": 1, "kernelMaximum": 4096, "strideMinimum": 1, "strideMaximum": 4096,
            "dilationMinimum": 1, "dilationMaximum": 4096, "groupMinimum": 1, "groupMaximum": 512,
            "outputPaddingMinimum": 0, "outputPaddingMaximum": 4096,
            "paddingMinimum": 0, "paddingMaximum": 4096, "maximumParameters": 2_000_000, "maximumMultiplyAdds": 128_000_000,
            "configurationKeys": .array(["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "dilation", "groups", "bias"].map { .string($0) }),
            "paddingMode": .string("zeros"), "outputPaddingRule": .string("lessThanStrideOrDilation"),
            "supportsDynamicOutputSize": false, "biasModes": .array([false, true])
        ])
        guard try JSONDecoder().decode(JSONValue.self, from: data) == expected else {
            throw GraphConfigurationError("ConvTranspose1d capability differs from the reviewed shape, padding, dtype or allocation limits.")
        }
        return Self()
    }
}
