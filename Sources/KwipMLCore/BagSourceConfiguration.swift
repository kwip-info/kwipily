import Foundation

/// Bag coordinates describe a packet shared by several ports, not a dense sample tensor.
public struct BagSourceConfiguration: Equatable, Sendable {
    public static let defaults: [String:JSONValue] = ["layout":.string("fixed"),"maximum_items_per_bag":4,"per_sample_weights":false,"include_last_offset":false]
    public let layout:String
    public let maximumItemsPerBag:Int
    public let usesPerSampleWeights:Bool
    public let includeLastOffset:Bool
    public init(values:[String:JSONValue]) throws {
        guard Set(values.keys)==Set(Self.defaults.keys),
              let layout=values["layout"]?.stringValue,["fixed","packed"].contains(layout),
              let maximum=values["maximum_items_per_bag"]?.integerValue,(1...128).contains(maximum),
              let weighted=values["per_sample_weights"]?.boolValue,
              let terminal=values["include_last_offset"]?.boolValue else {
            throw GraphConfigurationError("Bag input requires fixed or packed layout, an integer maximum of 1–128 items per bag, and Boolean weight and terminal-offset switches. Supply all four controls.")
        }
        self.layout=layout;maximumItemsPerBag=maximum
        usesPerSampleWeights=weighted;includeLastOffset=terminal
    }
    public var values:[String:JSONValue] {
        ["layout":.string(layout),"maximum_items_per_bag":.integer(Int64(maximumItemsPerBag)),"per_sample_weights":.bool(usesPerSampleWeights),"include_last_offset":.bool(includeLastOffset)]
    }
    public var outputPorts:[String] {
        ["indices"]+(layout == "packed" ? ["offsets"]:[])+(usesPerSampleWeights ? ["per_sample_weights"]:[])
    }
}

public struct BagPortContract: Equatable {
    public enum Role:String,Equatable {case indices,offsets;case perSampleWeights="per_sample_weights"}
    public let sourceNodeID:String
    public let configuration:BagSourceConfiguration
    public let role:Role
    public var dtype:ContractDType {role == .perSampleWeights ? .float32:.int64}
    public init(sourceNodeID:String,configuration:BagSourceConfiguration,role:Role) {
        self.sourceNodeID=sourceNodeID;self.configuration=configuration;self.role=role
    }
}
