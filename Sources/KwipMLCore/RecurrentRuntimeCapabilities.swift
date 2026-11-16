import Foundation

/// Optional exact discovery proves family support; each configuration still checks its device policy.
public struct RecurrentRuntimeCapabilities: Equatable {
    public static let operations: Set<String> = ["rnn", "lstm", "gru", "rnn_cell", "lstm_cell", "gru_cell"]
    public private(set) var componentIDs: Set<String> = []
    public init(descriptors: [String: Data] = [:]) {
        for id in Self.operations {
            if let data = descriptors[id], (try? Self.verify(data, componentID: id)) != nil { componentIDs.insert(id) }
        }
    }
    public func contains(_ id: String) -> Bool { componentIDs.contains(id) }
    public static func verify(_ data: Data, componentID: String) throws {
        guard data.count <= 4096, operations.contains(componentID),
              let entry = BundledRecurrentComponents.entries[componentID], entry.published,
              try JSONDecoder().decode(JSONValue.self, from: data) == entry.runtimeContract else {
            throw GraphConfigurationError("Recurrent support differs from its reviewed explicit-state, rank, parameter budget or CPU-only projection policy.")
        }
    }
}
