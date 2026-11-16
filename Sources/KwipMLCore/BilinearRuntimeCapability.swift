import Foundation

/// Exact dense two-input layer contract; no implicit observation or sample broadcast.
public struct BilinearRuntimeCapability: Equatable {
    private init() {}
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096 else { throw GraphConfigurationError("Bilinear capability exceeds 4096 bytes.") }
        let expected: JSONValue = .object([
            "version": .object(["major": 1, "minor": 0, "patch": 0]),
            "adapterID": .string("kwip.runner.graph-v2.bilinear"), "graphSchemaVersions": .array([2]),
            "maximumRank": 8, "preservesBatch": true, "minimumSampleRank": 1,
            "sampleBroadcasting": false, "maximumSampleElements": 100_000_000,
            "maximumParameters": 2_000_000, "featureMinimum": 1, "featureMaximum": 512,
            "biasModes": .array([false, true]), "graphDTypes": .array([.string("float32")])
        ])
        guard try JSONDecoder().decode(JSONValue.self, from: data) == expected else {
            throw GraphConfigurationError("Bilinear capability differs from the reviewed version, type, shape, parameter or bias contract.")
        }
        return Self()
    }
}
