import Foundation

/// Verified runtime support is separate from static bundle availability.
public struct PoolingRuntimeCapabilities: Equatable {
    public private(set) var componentIDs: Set<String> = []
    public init(descriptors: [String: Data] = [:]) {
        for (id, entry) in BundledPoolingComponents.entries where entry.published {
            if let data = descriptors[id], (try? Self.verify(data, componentID: id)) != nil {
                componentIDs.insert(id)
            }
        }
    }
    public func contains(_ id: String) -> Bool { componentIDs.contains(id) }
    public static func verify(_ data: Data, componentID: String) throws {
        guard data.count <= 4096, let entry = BundledPoolingComponents.entries[componentID],
              try JSONDecoder().decode(JSONValue.self, from: data) == entry.runtimeContract else {
            throw GraphConfigurationError("Pooling capability differs from its reviewed window, geometry or output types.")
        }
    }
}
