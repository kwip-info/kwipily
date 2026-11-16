import Foundation
import KwipMLCore

/// Separate optional starting-value support; a palette entry never implies runtime admission.
struct InitializationRuntimeCapabilities: Equatable {
    let componentIDs: Set<String>
    init(descriptor: Data? = nil) {
        componentIDs = descriptor.map { (try? Self.verify($0)) != nil } == true ? ["initialization"] : []
    }
    func contains(_ id: String) -> Bool { componentIDs.contains(id) }
    static func verify(_ data: Data) throws {
        guard data.count <= 4096,
              let expected = BundledInitializationComponents.metadata("initialization")?.runtimeContract,
              try JSONDecoder().decode(JSONValue.self, from: data) == expected else {
            throw GraphConfigurationError("Initialization support differs from its reviewed owner, seed, device or restored-weight policy.")
        }
    }
}
