import Foundation
import KwipMLCore

/// Root implements this hook with retained ownership and the managed supervisor.
/// Cancellation must not return until the actual owned worker terminates.
protocol WorkflowProgramExportTransport: Actor {
    func capture(source:WorkflowTrainedExportPackage,workspace:URL,runtime:RuntimeLocation,
                 configuration:ProgramExportConfiguration,executionBudget:ResolvedExecutionBudget?,budgetWindow:ExecutionBudgetWindow?)async throws->WorkflowProgramExportPackage
}
protocol WorkflowProgramExportIO: Actor {
    func stage(reference:WorkflowHistoryReference,root:URL,runtime:RuntimeLocation,expected:WorkflowProgramExportIdentity,
               configuration:ProgramExportConfiguration,executionBudget:ResolvedExecutionBudget?)async throws->WorkflowProgramExportPackage
    func publish(_ package:WorkflowProgramExportPackage,to destination:URL)async throws
}
actor WorkflowProgramExportWriter:WorkflowProgramExportIO {
    private let source:any WorkflowTrainedExportIO
    private let transport:any WorkflowProgramExportTransport
    init(transport:any WorkflowProgramExportTransport,source:any WorkflowTrainedExportIO=WorkflowTrainedExportWriter()) {
        self.transport=transport;self.source=source
    }
    func stage(reference:WorkflowHistoryReference,root:URL,runtime:RuntimeLocation,expected:WorkflowProgramExportIdentity,
               configuration:ProgramExportConfiguration,executionBudget:ResolvedExecutionBudget?)async throws->WorkflowProgramExportPackage {
        try Task.checkCancellation()
        let window=try executionBudget.map {try ExecutionBudgetWindow(policy:$0,startedAtUptime:ProcessInfo.processInfo.systemUptime)}
        let staged=try await source.stage(reference,root:root)
        defer {staged.remove()}
        try Task.checkCancellation()
        guard staged.reference == reference,staged.runID == expected.runID,
              staged.requestIdentity == expected.requestIdentity,staged.modelManifestSHA256 == expected.modelManifestSHA256,
              !staged.inputContract.isBag else {throw WorkflowArchiveError("The selected saved model changed, or uses a bag packet. Reopen History; use trained packet export for bag models.")}
        if let window {_ = try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)}
        let result=try await transport.capture(source:staged,workspace:root,runtime:runtime,configuration:configuration,executionBudget:executionBudget,budgetWindow:window)
        do {
            try Task.checkCancellation()
            guard result.identity == expected,result.configuration == configuration,
                  result.sourceManifestSHA256 == WorkflowArchiveCodec.hash(staged.manifestBytes) else {throw WorkflowArchiveError("Captured program differs from the reviewed saved model or configuration.")}
            return result
        } catch {result.remove();throw error}
    }
    func publish(_ package:WorkflowProgramExportPackage,to destination:URL)async throws {try package.publish(to:destination)}
}
