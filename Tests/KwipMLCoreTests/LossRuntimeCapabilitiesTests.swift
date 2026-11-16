import XCTest
@testable import KwipMLCore

final class LossRuntimeCapabilitiesTests:XCTestCase {
    /// Literal reviewed worker descriptor, independent of the decoder/generated value under test.
    func descriptor(_ id:String)throws->Data {
        let value:[String:Any] = ["version":"1.0.0","adapterID":"kwip.runner.objective."+id,"recipeSchemaVersions":[1],"configurationVersion":1,"predictionDTypes":["float32"],"devices":["cpu","mps"],"predictionRanks":[2,8],"sampleDimensionMaximum":4096,"classMinimum":2,"classMaximum":512,"classAxisMinimum": -7,"classAxisMaximum":7,"excludesBatchClassAxis":true,"reductions":id == "loss_kl_div" ? ["none","mean","sum","batchmean"]:["none","mean","sum"],"scalarizations":["mean","sum"],"targetRoles":id == "loss_ctc" ? ["target","input_lengths","target_lengths"]:id == "loss_triplet_margin" ? []:id == "loss_bce_with_logits" ? ["target","weight","pos_weight"]:["target"],"maximumTensorElements":100000000,"maximumWorkElements":100000000,"recipeMaximumBytes":32768,"fixedWeights":"buffers","mappedWeights":"preparedData","batchBroadcast":false]
        return try JSONSerialization.data(withJSONObject:value)
    }
    func testAbsentDescriptorsGrantNoExplicitObjectiveSupport() {
        XCTAssertTrue(LossRuntimeCapabilities().componentIDs.isEmpty)
        for id in LossConfiguration.componentIDs {XCTAssertFalse(LossRuntimeCapabilities().contains(id))}
    }
    func testExactPublishedStringVersionGrantsOnlySelectedObjective()throws {
        for id in LossConfiguration.componentIDs {
            XCTAssertEqual(BundledLossComponents.entries[id]?.published,true,"Actual publication is a prerequisite, never a test-local grant")
            let bytes=try descriptor(id)
            XCTAssertNoThrow(try LossRuntimeCapabilities.verify(bytes,componentID:id))
            XCTAssertEqual(LossRuntimeCapabilities(descriptors:[id:bytes]).componentIDs,[id])
        }
    }
    func testEveryDescriptorPolicyAndVersionTypeMustMatchExactly()throws {
        for id in LossConfiguration.componentIDs {
            let bytes=try descriptor(id),literal=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
            for key in literal.keys {
                var bad=literal;bad.removeValue(forKey:key)
                XCTAssertThrowsError(try LossRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject:bad),componentID:id),id+" missing "+key)
            }
            for (key,value) in [("version",["major":1,"minor":0,"patch":0] as Any),("batchBroadcast",true),("configurationVersion",true),("foreign",1)] {
                var bad=literal;bad[key]=value
                XCTAssertFalse(LossRuntimeCapabilities(descriptors:[id:try JSONSerialization.data(withJSONObject:bad)]).contains(id))
            }
            XCTAssertFalse(LossRuntimeCapabilities(descriptors:["foreign":bytes]).contains(id))
            XCTAssertThrowsError(try LossRuntimeCapabilities.verify(bytes,componentID:"foreign"))
            XCTAssertThrowsError(try LossRuntimeCapabilities.verify(Data(repeating:32,count:8193),componentID:id))
            XCTAssertThrowsError(try LossRuntimeCapabilities.verify(Data("null".utf8),componentID:id))
        }
    }
}
