import Foundation
import KwipMLCore

/// Readers and executable artifact export are admitted independently of forward layers.
struct DataLoadingRuntimeCapabilities: Equatable {
    let componentIDs: Set<String>
    init(dataset: Data? = nil, loader: Data? = nil) {
        componentIDs = Set([("dataset_source", dataset), ("data_loader", loader)].compactMap { id, data in
            guard let data, data.count <= 4096,
                  let expected = BundledDataLoadingComponents.entries[id]?.runtimeContract,
                  (try? JSONDecoder().decode(JSONValue.self, from: data)) == expected else { return nil }
            return id
        })
    }
    func contains(_ id: String) -> Bool { componentIDs.contains(id) }
}
struct ProgramExportRuntimeCapability: Equatable {
    let isAvailable: Bool
    init(descriptor: Data? = nil) {
        guard let descriptor, descriptor.count <= 4096 else { isAvailable = false; return }
        isAvailable = (try? JSONDecoder().decode(JSONValue.self, from: descriptor)) == BundledProgramExportCapabilities.entry.runtimeContract
    }
}
