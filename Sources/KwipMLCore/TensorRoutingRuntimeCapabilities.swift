import Foundation

/// Support is discovered per operator; absent or changed descriptors never grant execution.
public struct TensorRoutingRuntimeCapabilities:Equatable {
    public static let operations=TensorRoutingConfiguration.componentIDs
    public private(set) var componentIDs:Set<String>=[]
    public init(descriptors:[String:Data]=[:]) {
        for id in Self.operations {
            if let data=descriptors[id],(try? Self.verify(data,componentID:id)) != nil {componentIDs.insert(id)}
        }
    }
    public func contains(_ id:String)->Bool {componentIDs.contains(id)}
    public static func verify(_ data:Data,componentID:String)throws {
        guard data.count<=4096,operations.contains(componentID),let entry=BundledTensorRoutingComponents.entries[componentID],entry.published,
              try JSONDecoder().decode(JSONValue.self,from:data) == entry.runtimeContract else {
            throw GraphConfigurationError("Routing support differs from its reviewed named-part, selector, observation or bounded-notation policy.")
        }
    }
}
