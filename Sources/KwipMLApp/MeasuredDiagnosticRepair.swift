import Foundation
import KwipMLCore

/// An explicit single-variable experiment. Construction is read-only; Apply is
/// one canonical undoable configuration command, never an automatic retry.
@MainActor final class MeasuredDiagnosticRepair {
    let id=UUID()
    let originalLearningRate:Double
    let proposedLearningRate:Double
    let target:WorkflowNodeAddress
    let findingID:String
    private let ownerSessionID:UUID
    private let originalDocument:WorkflowDocument
    private let originalConfiguration:[String:JSONValue]
    private let proposedConfiguration:[String:JSONValue]
    private let input:MeasuredDiagnosticInput
    private let reportSHA256:String
    private let journalRoot:URL
    private var acceptedDocumentSHA256:String?
    private var attempted=false
    private var reverted=false
    private init(owner:WorkflowAuthoringStore,target:WorkflowNodeAddress,configuration:[String:JSONValue],input:MeasuredDiagnosticInput,findingID:String,reportSHA256:String,journalRoot:URL,rate:Double,proposed:[String:JSONValue]) {
        originalLearningRate=rate;proposedLearningRate=rate/2;self.target=target;self.findingID=findingID
        ownerSessionID=owner.sessionID;originalDocument=owner.document;originalConfiguration=configuration
        proposedConfiguration=proposed
        self.input=input;self.reportSHA256=reportSHA256;self.journalRoot=journalRoot
    }
    static func propose(input:MeasuredDiagnosticInput,findingID:String,train:WorkflowNodeAddress,owner:WorkflowAuthoringStore,journalRoot:URL)throws->MeasuredDiagnosticRepair {
        let report=try MeasuredDiagnostics.evaluate(input)
        guard let finding=report.findings.first(where:{$0.id==findingID}),["gradient.growth.v1","gradient.nonfinite.v1"].contains(finding.ruleID),
              input.capturedTrain==train,owner.isCurrent(owner.lease()),!owner.isBusy,!owner.hasPendingInspectorEdits,owner.document.drafts.isEmpty,
              input.workflowSHA256 == (try digest(owner.document)) else {throw DiagnosticContractError("This repair requires the unchanged captured workflow and a supported measured finding. Apply or revert other edits first.")}
        guard let trainNode=owner.document.nodes.first(where:{$0.address==train}),case .train(let trainConfig)=trainNode.payload else {
            throw DiagnosticContractError("Choose the captured Train before proposing a learning-rate change.")
        }
        guard trainConfig["optimizer"] == nil,
              !owner.document.edges.contains(where:{$0.to.address==train && $0.to.port=="optimizer"}) else {
            throw DiagnosticContractError("Train’s optimizer owns the active learning rate. Review its optimizer settings; changing the fallback rate would not change training.")
        }
        let settings=try WorkflowTrainRunSetup.resolve(in:owner.document,train:train)
        guard let rate=settings.configuration["learningRate"]?.doubleValue,rate.isFinite,rate>0,rate<=1,rate/2>0 else {
            throw DiagnosticContractError("The captured Train node has no safely editable learning-rate setting.")
        }
        var proposedRun=settings.configuration;proposedRun["learningRate"] = .number(rate/2)
        let config:[String:JSONValue],proposed:[String:JSONValue]
        if settings.address == train {
            config=trainConfig
            proposed=try WorkflowTrainRunSetup.setting(proposedRun,in:trainConfig,schemaVersion:owner.document.schemaVersion)
        } else {config=settings.configuration;proposed=proposedRun}
        return .init(owner:owner,target:settings.address,configuration:config,input:input,findingID:findingID,reportSHA256:report.inputSHA256,journalRoot:journalRoot,rate:rate,proposed:proposed)
    }
    func apply(to owner:WorkflowAuthoringStore)throws {
        guard !attempted,!reverted,owner.sessionID==ownerSessionID,owner.isCurrent(owner.lease()),!owner.isBusy,!owner.hasPendingInspectorEdits,
              owner.document==originalDocument else {throw DiagnosticContractError("The proposal is stale or already attempted. Re-evaluate current evidence before changing the model.")}
        // A failed intent write prevents the model mutation entirely.
        try journal("intent",owner:owner);attempted=true
        guard owner.apply(.configure(address:target,config:proposedConfiguration),lease:owner.lease()) != nil else {
            try? journal("refused",owner:owner)
            throw DiagnosticContractError(owner.error ?? "The canonical owner refused the proposed change.")
        }
        acceptedDocumentSHA256=try Self.digest(owner.document)
        do {try journal("applied",owner:owner)}
        catch {throw DiagnosticContractError("Learning-rate change applied and remains undoable, but its completion receipt could not be saved: \(error.localizedDescription)")}
    }
    func revert(in owner:WorkflowAuthoringStore)throws {
        guard let acceptedDocumentSHA256,!reverted,owner.sessionID==ownerSessionID,owner.isCurrent(owner.lease()),!owner.isBusy,!owner.hasPendingInspectorEdits,
              try Self.digest(owner.document)==acceptedDocumentSHA256,owner.canUndo else {throw DiagnosticContractError("Later changes prevent reverting this proposal. Use the normal edit history to review them first.")}
        guard owner.undo(lease:owner.lease()) else {throw DiagnosticContractError(owner.error ?? "The canonical owner refused undo.")}
        reverted=true
        do {try journal("reverted",owner:owner)}
        catch {throw DiagnosticContractError("Learning-rate change reverted, but its audit receipt could not be saved: \(error.localizedDescription)")}
    }
    var hasAppliedChange:Bool {acceptedDocumentSHA256 != nil && !reverted}
    private static func digest(_ document:WorkflowDocument)throws->String {WorkflowArchiveCodec.hash(try WorkflowCodec.encode(document))}
    private func journal(_ action:String,owner:WorkflowAuthoringStore)throws {
        struct Receipt:Encodable {
            let schemaVersion:Int
            let proposalID:UUID
            let projectID:String
            let action:String
            let recordedAt:Date
            let runID:String
            let requestSHA256:String
            let evidenceSHA256:String
            let capturedWorkflowSHA256:String?
            let currentWorkflowSHA256:String
            let findingID:String
            let target:WorkflowNodeAddress
            let before:[String:JSONValue]
            let proposed:[String:JSONValue]
        }
        let receipt=Receipt(schemaVersion:1,proposalID:id,projectID:owner.projectID,action:action,recordedAt:Date(),runID:input.runID,requestSHA256:input.requestSHA256,evidenceSHA256:reportSHA256,capturedWorkflowSHA256:input.workflowSHA256,currentWorkflowSHA256:try Self.digest(owner.document),findingID:findingID,target:target,before:originalConfiguration,proposed:proposedConfiguration)
        try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(receipt),root:journalRoot,parts:["diagnostic-repairs",owner.projectID,id.uuidString+"-"+action+".json"],limit:256_000)
    }
}
