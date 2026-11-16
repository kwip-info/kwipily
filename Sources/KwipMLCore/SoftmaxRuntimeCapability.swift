import Foundation

/// Additive discovery for explicit-axis, dense, batch-preserving normalization.
public struct SoftmaxRuntimeCapability:Equatable {
    private init() {}
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096 else {throw GraphConfigurationError("Softmax capability exceeds 4096 bytes.")}
        let expected:JSONValue = .object([
            "version":.object(["major":1,"minor":0,"patch":0]),
            "adapterID":.string("kwip.runner.graph-v2.softmax"),
            "graphSchemaVersions":.array([2]),
            "maximumRank":8,"preservesBatch":true
        ])
        guard try JSONDecoder().decode(JSONValue.self,from:data) == expected else {
            throw GraphConfigurationError("Softmax capability differs from the reviewed version, schema, rank or batch policy.")
        }
        return Self()
    }
}
