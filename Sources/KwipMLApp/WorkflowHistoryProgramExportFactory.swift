import Foundation
import KwipMLCore

/// A presentation snapshot extracted only from independently verified saved History.
struct WorkflowHistoryProgramExportSelection:Equatable {
    let reference:WorkflowHistoryReference
    let identity:WorkflowProgramExportIdentity
    let completed:Bool
    let hasModel:Bool
    let isBag:Bool
}
@MainActor enum WorkflowHistoryProgramExportFactory {
    static func validate(_ selected:WorkflowHistoryProgramExportSelection,currentReference:WorkflowHistoryReference?,active:Bool,busy:Bool,loading:Bool,capability:ProgramExportRuntimeCapability,runtimeVersion:String?,runtime:RuntimeLocation?)throws {
        guard active,!busy,!loading,currentReference == selected.reference,selected.completed,selected.hasModel,!selected.isBag,
              UUID(uuidString:selected.reference.projectID)?.uuidString == selected.reference.projectID,
              UUID(uuidString:selected.identity.runID)?.uuidString == selected.identity.runID,
              WorkflowArchiveCodec.isHash(selected.identity.requestIdentity),WorkflowArchiveCodec.isHash(selected.identity.modelManifestSHA256),
              capability.isAvailable,let runtimeVersion,runtimeVersion.utf8.count <= 128,
              runtimeVersion.range(of:#"^[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+_-]*$"#,options:.regularExpression) != nil,
              let runtime,runtime.packageVersions["torch"] == nil || runtime.packageVersions["torch"] == runtimeVersion else {
            throw WorkflowArchiveError("Select a verified completed ordinary model, wait for the workspace to be idle, and refresh executable-program runtime support. Packet models use trained model export.")
        }
    }
}
