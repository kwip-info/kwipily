import Foundation

/// Per-operator discovery; absent or stale descriptors never grant support.
public struct TensorCompositionRuntimeCapabilities: Equatable {
    public private(set) var componentIDs: Set<String> = []
    public init(descriptors: [String: Data] = [:]) {
        for (id, entry) in BundledTensorCompositionComponents.entries where entry.published {
            if let data = descriptors[id], (try? Self.verify(data, componentID:id)) != nil { componentIDs.insert(id) }
        }
    }
    public func contains(_ id: String) -> Bool { componentIDs.contains(id) }
    public static func verify(_ data: Data, componentID: String) throws {
        guard data.count <= 4096, let entry = BundledTensorCompositionComponents.entries[componentID],
              try JSONDecoder().decode(JSONValue.self,from:data) == entry.runtimeContract else {
            throw GraphConfigurationError("Composition capability differs from its reviewed direction, type or observation contract.")
        }
    }
}
