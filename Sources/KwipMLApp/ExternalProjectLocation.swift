import Foundation

/// Local identity is independent of the portable UUID, so copied files cannot share private run state.
struct ExternalProjectLocation: Equatable {
    let id: UUID
    let projectID: String
    let rootURL: URL
}

@MainActor extension WorkspaceStore {
    var externalFlowCatalog: ExternalFlowCatalog {
        ExternalFlowCatalog(directory: applicationDirectory.appendingPathComponent("external-flow-catalog"))
    }
}
