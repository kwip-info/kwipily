import Foundation

/// Exact optional runtime discovery; teaching publication never substitutes for execution proof.
public struct LifecycleRuntimeCapability: Equatable {
    public let isAvailable: Bool
    public init(descriptor: Data? = nil) {
        isAvailable = descriptor.map { (try? Self.verify($0)) != nil } ?? false
    }
    public static func verify(_ data: Data) throws {
        guard data.count <= 4096,
              try JSONDecoder().decode(JSONValue.self, from: data) == BundledLifecycleCapabilities.entry.runtimeContract else {
            throw GraphConfigurationError("Checkpoint support differs from its reviewed owned-History, CPU exact-resume or weights-only restoration policy.")
        }
    }
}
