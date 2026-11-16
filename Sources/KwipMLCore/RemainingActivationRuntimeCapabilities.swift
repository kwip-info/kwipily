import Foundation

/// Optional exact operator discovery never falls back to static palette presence.
public struct RemainingActivationRuntimeCapabilities:Equatable {
    public static let operations=RemainingActivationConfiguration.componentIDs
    public private(set) var componentIDs:Set<String>=[]
    public init(descriptors:[String:Data]=[:]) {
        for id in Self.operations {
            if let data=descriptors[id],(try? Self.verify(data,componentID:id)) != nil {componentIDs.insert(id)}
        }
    }
    public func contains(_ id:String)->Bool {componentIDs.contains(id)}
    public static func verify(_ data:Data,componentID:String)throws {
        guard data.count<=4096,operations.contains(componentID),let entry=BundledActivationComponents.entries[componentID],entry.published,
              try JSONDecoder().decode(JSONValue.self,from:data) == entry.runtimeContract else {
            throw GraphConfigurationError("Activation support differs from its reviewed learned-state, sample-axis or seeded train/evaluation policy.")
        }
    }
}
