import Foundation
import KwipMLCore
@testable import KwipMLApp

/// A retained outcome is observable before the execution queue's final cleanup.
/// Repeated-run fixtures join the actual idle boundary, rather than cancelling
/// successful completion work or waiting on an expectation for a refused start.
@MainActor func awaitWorkflowIntegrationIdle(_ workspace:WorkspaceStore,host:WorkflowWorkspaceSession,timeout:TimeInterval=10)async throws {
    let deadline=Date().addingTimeInterval(timeout)
    while workspace.isBusy || host.owner.isBusy || host.hasActiveOperation {
        guard workspace.workflowSession === host,Date()<deadline else {
            host.cancel()
            throw WorkflowArchiveError("Workflow did not return idle: "+host.statusMessage)
        }
        try await Task.sleep(for:.milliseconds(10))
    }
}
