import Foundation
import KwipMLCore

/// Captured private evaluation authority; no worker is launched by this service.
/// The stage override is internal and exists for deterministic lifecycle tests.
struct WorkflowEvaluationRequest {
    typealias Stage=(WorkflowFrozenModelReference,WorkflowDocument,WorkflowEvaluationPlan,URL)throws->WorkflowFrozenModelSnapshot
    let historyPolicy:RunHistoryPolicy
    let lease:WorkflowEditorLease
    let document:WorkflowDocument
    let plan:WorkflowEvaluationPlan
    let reference:WorkflowFrozenModelReference
    let bindingRevision:UInt64
    let runtime:RuntimeLocation
    let runtimeEvidence:WorkflowRuntimeEvidence
    let modelRegistrySHA256:String
    let dataRegistrySHA256:String
    let snapshot:WorkflowFrozenModelSnapshot
    let trainScientificIdentity:String?
    let requestIdentity:String
    let executionBudget: ResolvedExecutionBudget?
    let budgetWindow: ExecutionBudgetWindow?
    let evaluationID: UUID
    let storageReservation: WorkflowEvaluationStorageReservation?
    private let bindingIdentity:ObjectIdentifier

    @MainActor static func capture(from owner:WorkflowAuthoringStore,bindings:WorkflowPrivateFrozenModelBindings,
                                   evaluate:WorkflowNodeAddress,runtime:RuntimeLocation,root:URL,
                                   accountingRoot: URL? = nil, stage: Stage? = nil, executionBudget: ResolvedExecutionBudget? = nil, storageLimitBytes: Int64? = nil)async throws->Self {
        try Task.checkCancellation()
        guard storageLimitBytes == nil || stage == nil else { throw failure("Storage admission requires the managed frozen snapshot stage.") }
        let evaluationID = UUID(), stageID = UUID()
        let budgetWindow = try executionBudget.map { try ExecutionBudgetWindow(policy: $0, startedAtUptime: ProcessInfo.processInfo.systemUptime) }
        let lease=owner.lease(),historyPolicy=owner.runHistoryPolicy
        guard bindings.owner === owner,owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits else {
            throw failure("Capture evaluation from the active idle workflow with its own frozen-model bindings and resolved inspector edits.")
        }
        let document=owner.document,modelRegistry=owner.modelRegistry,dataRegistry=owner.dataRegistry
        let plan=try WorkflowEvaluationCompiler.compile(document,evaluate:evaluate,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        let reference=try await WorkflowTrainedArtifactAuthority.resolve(source:plan.modelSource,owner:owner,bindings:bindings,root:root)
        guard owner.isCurrent(lease),owner.document==document else {throw failure("Workflow changed during completed-model verification.")}
        let trainIdentity=try plan.modelSource.isTrain ? WorkflowTrainedArtifactAuthority.currentIdentity(train:plan.modelSource.address,owner:owner):nil
        let revision=bindings.revision,bindingIdentity=ObjectIdentifier(bindings)
        let modelHash=try WorkflowArchiveCodec.registryHash(modelRegistry),dataHash=try WorkflowArchiveCodec.registryHash(dataRegistry)
        let worker=Task.detached(priority:.userInitiated) {
          try await WorkflowHistoryVerificationWork.run(discard: {value in value.0.remove();value.2?.close()}) {()->(WorkflowFrozenModelSnapshot,WorkflowRuntimeEvidence,WorkflowEvaluationStorageReservation?) in
            var staged:WorkflowFrozenModelSnapshot?,retained=false
            let storage = try storageLimitBytes.map { try WorkflowEvaluationStorageReservation(workspace: root, accountingRoot: accountingRoot, projectID: document.projectID, evaluationID: evaluationID, stageID: stageID, limitBytes: $0) }
            defer {if !retained {staged?.remove(); storage?.close()}}
            try Task.checkCancellation()
            let result: WorkflowFrozenModelSnapshot
            if let stage { result = try stage(reference,document,plan,root) }
            else { result = try WorkflowFrozenModelSnapshot.stage(reference,document:document,plan:plan,root:root,stageID:stageID) }
            staged=result
            try Task.checkCancellation()
            guard result.reference==reference else {throw failure("Staged frozen model differs from the captured selection.")}
            try result.verify()
            let evidence=try WorkflowRuntimeEvidence.capture(runtime)
            let modelParts=runtime.kind == .bundled ? ["components.json"]:["Sources","KwipMLCore","Resources","components.json"]
            let modelBytes=try DiagnosticFileIO.read(root:runtime.workingDirectoryURL,parts:modelParts,limit:2_000_000)
            let dataBytes=try DiagnosticFileIO.read(root:runtime.workingDirectoryURL,parts:["runner","dataflow-operators.json"],limit:2_000_000)
            guard WorkflowArchiveCodec.hash(modelBytes)==evidence.workerFiles["components.json"],
                  WorkflowArchiveCodec.hash(dataBytes)==evidence.workerFiles["runner/dataflow-operators.json"],
                  try WorkflowArchiveCodec.registryHash(JSONDecoder().decode(ComponentRegistry.self,from:modelBytes))==modelHash,
                  try WorkflowArchiveCodec.registryHash(JSONDecoder().decode(DataFlowOperatorRegistry.self,from:dataBytes))==dataHash else {
                throw failure("Evaluation editor registries differ from the captured worker runtime.")
            }
            try Task.checkCancellation()
            retained=true
            return(result,evidence,storage)
          }
        }
        let (snapshot,evidence,storageReservation)=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        var accepted=false
        defer {if !accepted {snapshot.remove(); storageReservation?.close()}}
        try Task.checkCancellation()
        guard bindings.owner === owner,ObjectIdentifier(bindings)==bindingIdentity,owner.isCurrent(lease),
              owner.document==document,!owner.isBusy,!owner.hasPendingInspectorEdits,
              bindings.revision==revision,bindings.reference(for:plan.frozenModel)==reference else {
            throw failure("Workflow or frozen-model selection changed during verification. Capture evaluation again.")
        }
        _ = try budgetWindow?.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        struct Identity:Encodable {
            let historyPolicy:RunHistoryPolicy
            let schemaVersion:Int
            let document:WorkflowDocument
            let evaluate:WorkflowNodeAddress
            let reference:WorkflowFrozenModelReference
            let snapshotReceiptSHA256:String
            let modelRegistrySHA256:String
            let dataRegistrySHA256:String
            let runtimeEvidence:WorkflowRuntimeEvidence
            let executable:String
            let arguments:[String]
            let registry:String
            let workingDirectory:String
            let executionBudget: ResolvedExecutionBudget?
        }
        let identity=try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(Identity(historyPolicy:historyPolicy,schemaVersion:1,document:document,evaluate:evaluate,
            reference:reference,snapshotReceiptSHA256:snapshot.receiptSHA256,modelRegistrySHA256:modelHash,dataRegistrySHA256:dataHash,
            runtimeEvidence:evidence,executable:runtime.executableURL.path,arguments:runtime.argumentPrefix,registry:runtime.registryURL.path,workingDirectory:runtime.workingDirectoryURL.path,executionBudget:executionBudget)))
        let result=Self(historyPolicy:historyPolicy,lease:lease,document:document,plan:plan,reference:reference,bindingRevision:revision,runtime:runtime,
            runtimeEvidence:evidence,modelRegistrySHA256:modelHash,dataRegistrySHA256:dataHash,snapshot:snapshot,trainScientificIdentity:trainIdentity,requestIdentity:identity,executionBudget:executionBudget,budgetWindow:budgetWindow,evaluationID:evaluationID,storageReservation:storageReservation,bindingIdentity:bindingIdentity)
        accepted=true
        return result
    }
    func removeSnapshot() {
        snapshot.remove()
        storageReservation?.close()
    }
    @MainActor func isCurrent(in owner:WorkflowAuthoringStore,bindings:WorkflowPrivateFrozenModelBindings)->Bool {
        !owner.isBusy && matchesExecutionScope(in:owner,bindings:bindings)
    }
    @MainActor func matchesExecutionScope(in owner:WorkflowAuthoringStore,bindings:WorkflowPrivateFrozenModelBindings)->Bool {
        // The exact captured document, lease and binding revision below already pin
        // the plan; do not recompile it in this frequently-polled readiness check.
        let identityMatches = !plan.modelSource.isTrain || trainScientificIdentity != nil
        return identityMatches && bindings.owner === owner && ObjectIdentifier(bindings)==bindingIdentity && owner.isCurrent(lease)
            && owner.document==document && !owner.hasPendingInspectorEdits
            && bindings.revision==bindingRevision && bindings.reference(for:plan.frozenModel)==reference
    }
    /// Call off the main actor immediately before launch. Runtime evidence is source
    /// inventory evidence, not a capability probe or a guarantee of worker success.
    func verifyCapturedInputs()throws {
        try Task.checkCancellation()
        guard snapshot.reference==reference else {throw Self.failure("Captured frozen selection changed.")}
        try snapshot.verify()
        struct Selection:Decodable {let schemaVersion:Int;let testEndpoint:DataFlowEndpoint;let partitionRole:String?}
        let bytes=try DiagnosticFileIO.read(root:snapshot.directory,parts:["snapshot.json"],limit:256_000)
        let selection=try JSONDecoder().decode(Selection.self,from:bytes)
        guard WorkflowArchiveCodec.hash(bytes)==snapshot.receiptSHA256,selection.testEndpoint==plan.testEndpoint,
              (selection.schemaVersion == 1 ? "test":selection.partitionRole)==plan.settings.partitionRole else {
            throw Self.failure("Captured test endpoint differs from the frozen snapshot.")
        }
        guard try runtimeEvidence.matchesCurrent(runtime) else {throw Self.failure("Worker runtime changed after evaluation capture.")}
        try Task.checkCancellation()
    }
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
