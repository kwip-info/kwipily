import Foundation

/// Explicit extension to the baseline float classifier execution contract.
/// Decoding does not itself expose components or establish application readiness.
public struct EmbeddingRuntimeCapability:Equatable {
    private init() {}
    public var mpsMaximumNormExplanation:String {
        "Maximum-norm renormalization uses CPU fallback on Apple GPU in this runtime. Lookup and supported training remain on the selected device."
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096 else {throw GraphConfigurationError("Embedding capability exceeds 4096 bytes.")}
        let expected:JSONValue = .object([
            "version":.object(["major":1,"minor":0,"patch":0]),
            "adapterID":.string("kwip.runner.graph-v2.embedding"),"graphSchemaVersions":.array([2]),
            "inputDType":.string("int64"),"weightDType":.string("float32"),
            "maximumInputSampleRank":6,"maximumOutputRank":8,
            "maximumPreparedColumns":128,"maximumParameters":2_000_000,
            "preservesBatch":true,"outOfRangePolicy":.string("reject"),
            "paddingGradient":.string("suppressed"),"frequencyScaling":true,
            "frozenWeightsMayRenormalize":true,"trainingOptimizer":.string("Adam"),
            "trainableSparseGradients":false,"previewCacheSchemaForMaxNorm":2,
            "mpsMaxNormExecution":.string("cpuFallback")
        ])
        guard try JSONDecoder().decode(JSONValue.self,from:data)==expected else {
            throw GraphConfigurationError("Embedding capability differs from the reviewed dtype, shape, optimizer, cache or device policy.")
        }
        return Self()
    }
}
