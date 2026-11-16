import Foundation

/// Reviewed dense matrix adapter; vectors/scalars are deliberately not advertised.
public struct MatrixMultiplyRuntimeCapability:Equatable {
    private init() {}
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096 else {throw GraphConfigurationError("Matrix capability exceeds 4096 bytes.")}
        let expected:JSONValue = .object([
            "version":.object(["major":1,"minor":0,"patch":0]),
            "adapterID":.string("kwip.runner.graph-v2.matrixMultiply"),
            "graphSchemaVersions":.array([2]),"maximumRank":8,"preservesBatch":true,
            "minimumSampleRank":2,"sampleBroadcasting":true,
            "maximumOutputElements":100_000_000,"graphDTypes":.array([.string("float32")])
        ])
        guard try JSONDecoder().decode(JSONValue.self,from:data) == expected else {
            throw GraphConfigurationError("Matrix capability differs from the reviewed version, type, rank, broadcasting or allocation policy.")
        }
        return Self()
    }
}
