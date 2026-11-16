import Foundation
import KwipMLCore

/// Same-project immutable identities, never absolute paths or portable node
/// configuration. Binding alone is not verification: selection and every run
/// must independently verify all pinned artifacts before publishing authority.
struct WorkflowFrozenModelReference:WorkflowPrivateTrainReference {
    let projectID:String
    let directoryID:String
    let runID:String
    let requestIdentity:String
    let outcomeSHA256:String
    let modelManifestSHA256:String
    let preparationManifestSHA256:String
    let checkpointSHA256:String
    static var storageNamespace:String {"workflow-frozen-model"}
    static var fieldNames:Set<String> {
        ["projectID","directoryID","runID","requestIdentity","outcomeSHA256",
         "modelManifestSHA256","preparationManifestSHA256","checkpointSHA256"]
    }
    static func accepts(_ node:WorkflowNode)->Bool {
        guard node.address.domain == .control else {return false}
        switch node.payload {case .frozenModel,.train:return true;default:return false}
    }
    var hasValidPins:Bool {
        UUID(uuidString:projectID) != nil && UUID(uuidString:directoryID) != nil && UUID(uuidString:runID) != nil
            && [requestIdentity,outcomeSHA256,modelManifestSHA256,preparationManifestSHA256,checkpointSHA256]
                .allSatisfy(WorkflowArchiveCodec.isHash)
    }
}
typealias WorkflowPrivateFrozenModelBindings = WorkflowPrivateTrainBindings<WorkflowFrozenModelReference>
