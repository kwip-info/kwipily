import Foundation
import KwipMLCore

enum WorkflowExperimentVariantEvidence {
    case verified(VerifiedWorkflowOutcome)
    case recorded(WorkflowExperimentVariantStatus)
    case unavailable(String)
}
struct WorkflowExperimentHistory {
    struct Variant {
        let ordinal:Int
        let evidence:WorkflowExperimentVariantEvidence
    }
    let record:WorkflowExperimentGroupRecord
    let variants:[Variant]
    var isVerifiedComplete:Bool {
        !variants.isEmpty && variants.allSatisfy {
            if case .verified(let run)=$0.evidence {return run.outcome.status == .completed};return false
        }
    }
}

/// A ledger's status is a claim. Reopen independent run evidence and join it to
/// exact declared variant documents and shared input hashes before verifying it.
enum WorkflowExperimentHistoryReader {
    struct Listing {
        let entries:[WorkflowExperimentGroupRecord]
        let issues:[String]
        let truncated:Bool
    }
    static func list(root:URL,projectID:String,limit:Int=1000)throws->Listing {
        let scan=try WorkflowHistoryReader.scanDirectories(root:root,projectID:projectID,limit:limit,namespace:"workflow-experiments")
        var entries:[WorkflowExperimentGroupRecord]=[],issues=scan.issues
        for id in scan.directories.sorted(by:{$0.uuidString<$1.uuidString}) {
            try Task.checkCancellation()
            do {
                let directory=root.appendingPathComponent("workflow-experiments/\(projectID)/\(id.uuidString)")
                // Read beneath the workspace first, rejecting symbolic links in the path.
                _=try DiagnosticFileIO.read(root:root,parts:["workflow-experiments",projectID,id.uuidString,"group.json"],limit:256_000)
                let record=try WorkflowExperimentGroupStore.readRecord(directory:directory)
                guard record.projectID == projectID else {throw DataFlowContractError("Group belongs to another project.")}
                entries.append(record)
            } catch {issues.append("Group \(id.uuidString): \(error.localizedDescription)")}
        }
        // Missing historical dates stay unknown and sort after dated records.
        entries.sort {
            let a=$0.createdAtUnixSeconds ?? -1,b=$1.createdAtUnixSeconds ?? -1
            return a == b ? $0.groupID < $1.groupID:a > b
        }
        return .init(entries:entries,issues:issues,truncated:scan.truncated)
    }
    struct CapturedContext {
        let record:WorkflowExperimentGroupRecord
        fileprivate let inputs:WorkflowExperimentInputs
        fileprivate let expectedDocuments:[WorkflowDocument]
        fileprivate init(record:WorkflowExperimentGroupRecord,inputs:WorkflowExperimentInputs,expectedDocuments:[WorkflowDocument]) {
            self.record=record;self.inputs=inputs;self.expectedDocuments=expectedDocuments
        }
    }
    enum VerificationPolicy {case historyPinned,recoveryRunning}
    struct VerifiedVariant {
        let run:VerifiedWorkflowOutcome
        let outcomeSHA256:String
    }
    static func captureContext(directory:URL)throws->CapturedContext {
        try Task.checkCancellation()
        let record=try WorkflowExperimentGroupStore.verifiedRecord(directory:directory)
        let source=try WorkflowCodec.decode(DiagnosticFileIO.read(root:directory,parts:["authoring.json"],limit:2_000_000))
        let inputs=try WorkflowExperimentInputs.open(directory:directory,expectedManifestSHA256:record.inputsManifestSHA256)
        let authority=try WorkflowTrainRunSetup.resolve(in:source,train:record.train)
        let config=authority.configuration
        guard let index=source.nodes.firstIndex(where:{$0.address==authority.address}),let value=config["experiment"] else {
            throw DataFlowContractError("The experiment source is missing its captured specification.")
        }
        let spec=try WorkflowExperimentSpecification.decode(value)
        var combinations:[[String:JSONValue]]=[[:]]
        for axis in spec.axes {
            combinations=combinations.flatMap {existing in axis.values.map {value in
                var next=existing;next[axis.field]=value;return next
            }}
        }
        guard combinations.count == record.variants.count else {throw DataFlowContractError("Experiment membership differs from its declared combinations.")}
        let documents=try record.variants.map {entry->WorkflowDocument in
            try Task.checkCancellation()
            var expected=source
            let updated=config.merging(combinations[entry.ordinal-1]) {_,new in new}
            if source.schemaVersion>=16 {
                guard case .train(let trainConfig)=source.nodes[index].payload else {throw DataFlowContractError("The captured Train settings authority changed.")}
                expected.nodes[index].payload = .train(config:try WorkflowTrainRunSetup.setting(updated,in:trainConfig,schemaVersion:source.schemaVersion))
            } else {expected.nodes[index].payload = .runSetup(config:updated)}
            let binding=WorkflowExperimentRequestIdentity(groupID:record.groupID,inputsManifestSHA256:record.inputsManifestSHA256,
                sourceRequestIdentity:record.sourceRequestIdentity,ordinal:entry.ordinal)
            guard try WorkflowExecutionRequest.experimentIdentity(binding:binding,document:expected) == entry.requestIdentity else {
                throw DataFlowContractError("Variant identity does not match the declared experiment.")
            }
            return expected
        }
        return CapturedContext(record:record,inputs:inputs,expectedDocuments:documents)
    }
    static func verifiedVariant(context:CapturedContext,ordinal:Int,workspaceDirectory:URL,policy:VerificationPolicy)throws->VerifiedVariant {
        try Task.checkCancellation()
        let record=context.record
        guard (1...32).contains(ordinal),record.variants.indices.contains(ordinal-1),context.expectedDocuments.indices.contains(ordinal-1) else {throw DataFlowContractError("Unknown experiment ordinal.")}
        let entry=record.variants[ordinal-1],expected=context.expectedDocuments[ordinal-1]
        switch policy {
        case .historyPinned:
            guard entry.outcomeSHA256 != nil else {throw DataFlowContractError("Recorded variant has no pinned outcome.")}
        case .recoveryRunning:
            guard entry.status == .running,entry.outcomeSHA256 == nil else {throw DataFlowContractError("Recovery verification requires an unpinned running variant.")}
        }
        guard let directoryID=entry.directoryID.flatMap(UUID.init(uuidString:)),let runID=entry.runID else {throw DataFlowContractError("Recorded outcome has no run identity.")}
        let parts=["workflow-runs",record.projectID,directoryID.uuidString]
        let bytes=try DiagnosticFileIO.read(root:workspaceDirectory,parts:parts+["outcome.json"],limit:256_000)
        let hash=WorkflowPhaseArtifacts.hash(bytes)
        if policy == .historyPinned,hash != entry.outcomeSHA256 {throw DataFlowContractError("The recorded variant outcome changed.")}
        let reference=WorkflowHistoryReference(projectID:record.projectID,directoryID:directoryID)
        guard case .verified(let run)=WorkflowHistoryReader.load(reference,root:workspaceDirectory) else {throw DataFlowContractError("The variant archive could not be independently verified.")}
        guard try WorkflowArchiveCodec.decodeOutcome(bytes) == run.outcome,run.outcome.runID == runID,run.outcome.requestIdentity == entry.requestIdentity,run.capture.train == record.train else {
            throw DataFlowContractError("The archive belongs to another experiment variant.")
        }
        if policy == .historyPinned {
            guard run.outcome.status.rawValue == entry.status.rawValue else {throw DataFlowContractError("The archive terminal state differs from its recorded variant.")}
        } else {
            guard [.completed,.failed,.cancelled].contains(run.outcome.status) else {throw DataFlowContractError("Recovery requires a verified terminal outcome.")}
        }
        let archived=try WorkflowCodec.decode(DiagnosticFileIO.read(root:workspaceDirectory,
            parts:parts+run.capture.workflow.path.split(separator:"/").map(String.init),limit:2_000_000))
        guard archived == expected,Set(run.capture.sources.map(\.address)) == Set(context.inputs.sources.keys),
              run.capture.sources.allSatisfy({source in
                  guard let input = context.inputs.sources[source.address], input.sha256 == source.sha256, input.byteCount == source.bytes,
                        input.isFileCollection == (source.collectionMembers != nil) else { return false }
                  let captured = input.collectionMembers.map { "\($0.capturePath):\($0.byteCount):\($0.sha256)" }.sorted()
                  let archived = (source.collectionMembers ?? []).map { file in
                      "\(file.path.split(separator: "/").suffix(2).joined(separator: "/")):\(file.bytes):\(file.sha256)"
                  }.sorted()
                  return captured == archived
              }) else {throw DataFlowContractError("The variant source or input bytes differ from its captured group.")}
        // Re-read after independent verification, so the returned pin describes the
        // same immutable outcome inspected above rather than an earlier path read.
        guard try DiagnosticFileIO.read(root:workspaceDirectory,parts:parts+["outcome.json"],limit:256_000) == bytes else {throw DataFlowContractError("Variant outcome changed during verification.")}
        try Task.checkCancellation()
        return .init(run:run,outcomeSHA256:hash)
    }
    static func load(directory:URL,workspaceDirectory:URL)throws->WorkflowExperimentHistory {
        let context=try captureContext(directory:directory)
        let variants=context.record.variants.map {entry->WorkflowExperimentHistory.Variant in
            guard entry.outcomeSHA256 != nil else {return .init(ordinal:entry.ordinal,evidence:.recorded(entry.status))}
            do {
                let verified=try verifiedVariant(context:context,ordinal:entry.ordinal,workspaceDirectory:workspaceDirectory,policy:.historyPinned)
                return .init(ordinal:entry.ordinal,evidence:.verified(verified.run))
            } catch {return .init(ordinal:entry.ordinal,evidence:.unavailable(error.localizedDescription))}
        }
        try Task.checkCancellation()
        return .init(record:context.record,variants:variants)
    }
}
