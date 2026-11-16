import Foundation
import CryptoKit
import KwipMLCore

/// Captured ordinary-model input. Authoring remains solely owned by ModelAuthoringStore.
/// A request may outlive edits, but only a matching owner lease may publish its live UI result.
struct ModelExecutionRequest:Equatable {
    let lease:ModelEditorLease
    let snapshot:ModelExecutionSnapshot
    let settings:ExecutionSettings
    let executionSignature:Data
    let exportContentIdentity:String
    var graph:GraphDocument {snapshot.graph}
    var projectID:String {lease.projectID}

    private init(lease:ModelEditorLease,snapshot:ModelExecutionSnapshot,executionSignature:Data,exportContentIdentity:String) {
        self.lease=lease;self.snapshot=snapshot;settings=snapshot.graph.execution ?? ExecutionSettings()
        self.executionSignature=executionSignature;self.exportContentIdentity=exportContentIdentity
    }
    @MainActor static func capture(from owner:ModelAuthoringStore,registry:ComponentRegistry)throws->ModelExecutionRequest {
        // No suspension: lease and source values are captured in the same main-actor operation.
        let lease=owner.lease()
        let snapshot=try ModelExecutionSnapshotBuilder.build(owner.document,registry:registry)
        return try ModelExecutionRequest(lease:lease,snapshot:snapshot,
            executionSignature:ModelExecutionIdentity.fullSignature(snapshot.graph,registry:registry),
            exportContentIdentity:ModelExecutionIdentity.exportContentIdentity(snapshot.graph))
    }
    @MainActor func isCurrent(in owner:ModelAuthoringStore?)->Bool {
        guard let owner else {return false}
        return owner.lease()==lease
    }
}

/// Pure identities: full execution invalidation is distinct from architecture-only checkpoint identity.
enum ModelExecutionIdentity {
    static func fullSignature(_ graph:GraphDocument,registry:ComponentRegistry)throws->Data {
        // Preserve WorkspaceStore.semanticSignature's exact keys, optional settings and encoder options.
        struct Signature:Encodable {let architecture:Data;let execution:ExecutionSettings?}
        let architecture=try GraphSemantics.canonicalData(graph,registry:registry)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return try encoder.encode(Signature(architecture:architecture,execution:graph.execution))
    }
    static func exportContentIdentity(_ graph:GraphDocument)throws->String {
        // Conservatively retain every captured field. Readable aliases depend on labels;
        // harmless extra regeneration is preferable to an uncontracted normalization.
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return SHA256.hash(data:try encoder.encode(graph)).map{String(format:"%02x",$0)}.joined()
    }
}
