import Foundation

/// Exact optional discovery; a visible component is not an execution permission.
public struct SpatialPatchRuntimeCapabilities:Equatable {
    public static let operations=SpatialPatchConfiguration.componentIDs
    public private(set) var componentIDs:Set<String>=[]
    public init(descriptors:[String:Data]=[:]) {
        for id in Self.operations {
            if let data=descriptors[id],(try? Self.verify(data,componentID:id)) != nil {componentIDs.insert(id)}
        }
    }
    public func contains(_ id:String)->Bool {componentIDs.contains(id)}
    public static func verify(_ data:Data,componentID:String)throws {
        guard data.count<=4096,operations.contains(componentID),let entry=BundledSpatialPatchComponents.entries[componentID],entry.published,
              try JSONDecoder().decode(JSONValue.self,from:data) == entry.runtimeContract else {
            throw GraphConfigurationError("Spatial patch support differs from its reviewed paired geometry, rank, order or overlap-addition policy.")
        }
    }
}
