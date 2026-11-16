import Foundation

/// Exact deferred dense layer contract; shape validation precedes allocation and optimizer creation.
public struct LazyLinearRuntimeCapability: Equatable {
    private init() {}
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096 else {
            throw GraphConfigurationError("LazyLinear capability exceeds 4096 bytes.")
        }
        let expected: JSONValue = .object([
            "version": .object(["major": 1, "minor": 0, "patch": 0]),
            "adapterID": .string("kwip.runner.graph-v2.lazy-linear"),
            "graphSchemaVersions": .array([2]),
            "maximumRank": 8, "preservesBatch": true, "minimumSampleRank": 1,
            "maximumSampleElements": 100_000_000, "maximumParameters": 2_000_000,
            "inputFeatureMinimum": 1, "inputFeatureMaximum": 4096,
            "outputFeatureMinimum": 1, "outputFeatureMaximum": 512,
            "biasModes": .array([false, true]), "graphDTypes": .array([.string("float32")]),
            "inputWidthInferred": true,
            "materializationPolicy": .string("validatedShapeBeforeOptimizer")
        ])
        guard try JSONDecoder().decode(JSONValue.self, from: data) == expected else {
            throw GraphConfigurationError("LazyLinear capability differs from the reviewed version, shape, dtype, parameter or materialization contract.")
        }
        return Self()
    }
}
