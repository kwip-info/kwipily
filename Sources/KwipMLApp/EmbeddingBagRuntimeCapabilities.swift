import Foundation
import KwipMLCore

/// Independent optional discovery; ordinary Embedding never grants bag-coordinate support.
struct EmbeddingBagRuntimeCapabilities: Equatable {
    static let operations: Set<String> = ["bag_input", "embedding_bag"]
    private(set) var componentIDs: Set<String> = []
    init(descriptors: [String: Data] = [:]) {
        for id in Self.operations {
            if let data=descriptors[id],(try? Self.verify(data,componentID:id)) != nil { componentIDs.insert(id) }
        }
    }
    func contains(_ id: String) -> Bool { componentIDs.contains(id) }
    var isComplete: Bool { Self.operations.isSubset(of: componentIDs) }
    static func verify(_ data: Data, componentID: String) throws {
        guard data.count <= 8192, operations.contains(componentID),
              try JSONDecoder().decode(JSONValue.self,from:data) == expected(componentID) else {
            throw GraphConfigurationError("Bag support differs from its reviewed coordinate types, whole-bag bounds, learned table or device policy.")
        }
    }
    private static func expected(_ id:String) -> JSONValue {
        var value:[String:JSONValue] = [
            "version":.object(["major":1,"minor":0,"patch":0]),
            "adapterID":.string("kwip.runner.graph-v2."+id), "graphSchemaVersions":.array([2]),
            "packetSchemaVersion":1, "layouts":.array([.string("fixed"),.string("packed")]),
            "coordinateDTypes":.object(["indices":.string("int64"),"offsets":.string("int64"),"per_sample_weights":.string("float32")]),
            "maximumItemsPerBag":128,"maximumBags":4096,"maximumItems":1_000_000,
            "maximumOutputElements":2_000_000,"integerDerivatives":.string("unavailable"),
            "sourceWeights":.string("realPairedInput"),"batching":.string("wholeBags"),"emptyPackedBags":.string("preserve")
        ]
        if id == "embedding_bag" {
            value.merge([
                "modes":.array([.string("mean"),.string("sum"),.string("max")]),"weightedModes":.array([.string("sum")]),
                "outputDType":.string("float32"),"maximumParameters":2_000_000,"maximumStaticTableElements":100_000_000,
                "paddingGradient":.string("suppressed"),"frozenWeightsMayRenormalize":true,"trainableSparseAdam":false,
                "mpsMaxNormExecution":.string("reject"),"mpsTrainableFrequencyScaling":.string("reject"),"sourceWeightsDerivatives":.string("sumOnly")
            ]) {_,new in new}
        }
        return .object(value)
    }
}
