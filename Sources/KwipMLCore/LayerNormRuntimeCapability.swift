import Foundation

/// Reviewed float32, trailing-sample LayerNorm execution domain.
public struct LayerNormRuntimeCapability: Equatable {
    private init() {}
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096 else { throw GraphConfigurationError("Layer normalization capability exceeds 4096 bytes.") }
        let expected: JSONValue = .object([
            "version": .object(["major": 1, "minor": 0, "patch": 0]),
            "adapterID": .string("kwip.runner.graph-v2.layer_norm"),
            "graphSchemaVersions": .array([2]), "maximumRank": 8, "preservesBatch": true,
            "epsilonMinimum": .number(LayerNormConfiguration.epsilonRange.lowerBound),
            "epsilonMaximum": .number(LayerNormConfiguration.epsilonRange.upperBound),
            "elementwiseAffineModes": .array([false, true]), "biasModes": .array([false, true]),
            "varianceCorrection": 0, "usesCurrentInputStatistics": true
        ])
        guard try JSONDecoder().decode(JSONValue.self, from: data) == expected else {
            throw GraphConfigurationError("Layer normalization capability differs from the reviewed version, shape, epsilon or affine/statistics policy.")
        }
        return Self()
    }
}
