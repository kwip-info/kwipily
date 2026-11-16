import Foundation

/// Optional exact discovery proves family support; each configuration still checks its device policy.
public struct AttentionRuntimeCapabilities: Equatable {
    public static let operations: Set<String> = ["multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","attention_mask","rotary_position"]
    public private(set) var componentIDs: Set<String> = []
    public init(descriptors: [String: Data] = [:]) {
        for id in Self.operations {
            if let data = descriptors[id], (try? Self.verify(data, componentID: id)) != nil { componentIDs.insert(id) }
        }
    }
    public func contains(_ id: String) -> Bool { componentIDs.contains(id) }
    public static func verify(_ data: Data, componentID: String) throws {
        guard data.count <= 4096, operations.contains(componentID),
              let entry = BundledAttentionComponents.entries[componentID], entry.published,
              try JSONDecoder().decode(JSONValue.self, from: data) == entry.runtimeContract else {
            throw GraphConfigurationError("Attention support differs from its reviewed mask convention, ordered ownership, resource budget or device-local execution policy.")
        }
    }
}
