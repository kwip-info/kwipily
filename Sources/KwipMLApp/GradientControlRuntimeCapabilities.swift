import Foundation
import KwipMLCore

/// Training controls are admitted independently of tensor-forward adapters.
struct GradientControlRuntimeCapabilities: Equatable {
    let componentIDs: Set<String>
    init(gradientPolicy: Data? = nil, trainability: Data? = nil) {
        componentIDs = Set([("gradient_policy", gradientPolicy), ("trainability", trainability)].compactMap { id, data in
            guard let data, (try? Self.verify(data, componentID: id)) != nil else { return nil }
            return id
        })
    }
    static func verify(_ data: Data, componentID: String) throws {
        guard data.count <= 4096,
              let expected = BundledGradientControlComponents.metadata(componentID)?.runtimeContract,
              try JSONDecoder().decode(JSONValue.self, from: data) == expected else {
            throw GraphConfigurationError("Training policy support differs from its reviewed owners, devices, capture permission or update behavior.")
        }
    }
    func contains(_ id: String) -> Bool { componentIDs.contains(id) }
}
