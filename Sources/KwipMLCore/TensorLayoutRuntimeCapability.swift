import Foundation

/// Verified additive worker support. The historical classifier contract alone is insufficient.
public struct TensorLayoutRuntimeCapability:Equatable {
    private init() {}
    public static let operations:Set<String>=["reshape","transpose","flatten"]
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096 else {throw GraphConfigurationError("Tensor layout capability exceeds 4096 bytes.")}
        let expected:JSONValue = .object([
            "version":.object(["major":1,"minor":0,"patch":0]),
            "adapterID":.string("kwip.runner.graph-v2.tensor-layout"),
            "graphSchemaVersions":.array([2]),
            "operations":.array([.string("reshape"),.string("transpose"),.string("flatten")]),
            "maximumRank":8,"preservesBatch":true
        ])
        guard try JSONDecoder().decode(JSONValue.self,from:data) == expected else {
            throw GraphConfigurationError("Tensor layout capability differs from the reviewed version, operations, rank or batch policy.")
        }
        return Self()
    }
}
