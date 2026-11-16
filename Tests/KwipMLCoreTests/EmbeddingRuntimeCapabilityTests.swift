import XCTest
@testable import KwipMLCore

final class EmbeddingRuntimeCapabilityTests:XCTestCase {
    private let descriptor = #"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.embedding","graphSchemaVersions":[2],"inputDType":"int64","weightDType":"float32","maximumInputSampleRank":6,"maximumOutputRank":8,"maximumPreparedColumns":128,"maximumParameters":2000000,"preservesBatch":true,"outOfRangePolicy":"reject","paddingGradient":"suppressed","frequencyScaling":true,"frozenWeightsMayRenormalize":true,"trainingOptimizer":"Adam","trainableSparseGradients":false,"previewCacheSchemaForMaxNorm":2,"mpsMaxNormExecution":"cpuFallback"}"#
    func testStrictLookupDomainAndDisclosure()throws {
        let data=Data(descriptor.utf8)
        let capability=try EmbeddingRuntimeCapability.decode(data)
        XCTAssertTrue(capability.mpsMaximumNormExplanation.contains("CPU"))
        let original=try JSONDecoder().decode([String:JSONValue].self,from:data)
        for key in original.keys {
            var missing=original;missing.removeValue(forKey:key)
            XCTAssertThrowsError(try EmbeddingRuntimeCapability.decode(JSONEncoder().encode(missing)),key)
            var wrong=original;wrong[key] = .null
            XCTAssertThrowsError(try EmbeddingRuntimeCapability.decode(JSONEncoder().encode(wrong)),key)
        }
        var extra=original;extra["assumeNativeGPU"] = true
        XCTAssertThrowsError(try EmbeddingRuntimeCapability.decode(JSONEncoder().encode(extra)))
        XCTAssertThrowsError(try EmbeddingRuntimeCapability.decode(Data(repeating:32,count:4097)))
    }
}
