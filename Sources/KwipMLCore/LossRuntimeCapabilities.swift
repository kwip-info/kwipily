import Foundation

/// Optional objective discovery; implicit classification does not grant an explicit recipe.
public struct LossRuntimeCapabilities:Equatable {
    public static let operations=LossConfiguration.componentIDs
    public private(set) var componentIDs:Set<String>=[]
    public init(descriptors:[String:Data]=[:]) {
        for id in Self.operations {
            if let bytes=descriptors[id],(try? Self.verify(bytes,componentID:id)) != nil {componentIDs.insert(id)}
        }
    }
    public func contains(_ id:String)->Bool {componentIDs.contains(id)}
    public static func verify(_ data:Data,componentID:String)throws {
        guard data.count<=8192,operations.contains(componentID),let entry=BundledLossComponents.entries[componentID],entry.published,
              try JSONDecoder().decode(JSONValue.self,from:data) == entry.runtimeContract else {
            throw GraphConfigurationError("Objective support differs from its reviewed recipe, typed targets, reduction, fixed weights or execution bounds.")
        }
    }
}
