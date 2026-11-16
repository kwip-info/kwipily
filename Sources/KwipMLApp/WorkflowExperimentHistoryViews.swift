import SwiftUI
import KwipMLCore

struct WorkflowExperimentHistoryList:View {
    let root:URL
    let projectID:String
    let selectedGroup:String?
    let refreshKey:String
    let select:(String)->Void
    @State private var listing:WorkflowExperimentHistoryReader.Listing?
    @State private var message=""
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            SectionLabel(title:"Experiment groups")
            if let listing {
                ForEach(listing.entries,id:\.groupID) {entry in
                    Button {select(entry.groupID)} label: {
                        VStack(alignment:.leading,spacing:4) {
                            Text("Group · " + String(entry.groupID.prefix(8))).font(.caption.bold())
                            if let time=entry.createdAtUnixSeconds {
                                Text(Date(timeIntervalSince1970:time),format:.dateTime.month(.abbreviated).day().hour().minute().second())
                                    .font(.caption2).foregroundStyle(MLTheme.muted)
                            } else {Text("Creation time unavailable").font(.caption2).foregroundStyle(MLTheme.muted)}
                            Text("\(entry.variants.count) variants · \(entry.variants.filter {$0.status == .completed}.count) recorded completed")
                                .font(.caption2).foregroundStyle(MLTheme.muted)
                            Text(entry.stopRequested ? "Stopped · select to verify":"Select to verify captured evidence")
                                .font(.caption2).foregroundStyle(MLTheme.muted)
                        }.frame(maxWidth:.infinity,alignment:.leading).padding(8)
                            .background(selectedGroup == entry.groupID ? MLTheme.panel:Color.clear)
                    }.buttonStyle(.plain).accessibilityLabel("Verify experiment group \(entry.groupID)")
                        .accessibilityValue("\(entry.createdAtUnixSeconds.map { Date(timeIntervalSince1970:$0).formatted(date:.abbreviated,time:.standard) } ?? "Creation time unavailable"). \(entry.variants.count) variants, \(entry.variants.filter {$0.status == .completed}.count) recorded completed. \(entry.stopRequested ? "Stopped":"Select to verify captured evidence")")
                        .accessibilityAddTraits(selectedGroup == entry.groupID ? .isSelected:[])
                }
                if listing.entries.isEmpty {Text("No experiment groups recorded.").font(.caption2).foregroundStyle(MLTheme.muted)}
                if listing.truncated {Text("Group scan limit reached; listing is incomplete and may omit newer groups.").font(.caption2).foregroundStyle(MLTheme.rose)}
                ForEach(Array(listing.issues.enumerated()),id:\.offset) {_,issue in Text(issue).font(.caption2).foregroundStyle(MLTheme.rose)}
            } else {Text(message.isEmpty ? "Reading groups…":message).font(.caption2).foregroundStyle(MLTheme.muted)}
        }.task(id:refreshKey) {
            let worker=Task.detached(priority:.utility) {try WorkflowExperimentHistoryReader.list(root:root,projectID:projectID)}
            do {
                let result=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                guard !Task.isCancelled else {return};listing=result;message=""
            } catch {guard !Task.isCancelled else {return};listing=nil;message=error.localizedDescription}
        }
    }
}

struct WorkflowExperimentGroupDetail:View {
    let root:URL
    let projectID:String
    let groupID:String
    let sessionID:UUID
    let isCurrent:()->Bool
    let onRecoveryRefresh:()->Void
    let selectRun:(WorkflowHistoryReference)->Void
    @State private var result:WorkflowExperimentHistory?
    @State private var message=""
    @State private var loadedScope:WorkflowExperimentRecoveryStore.Scope?
    @State private var refresh=0
    private var scope:WorkflowExperimentRecoveryStore.Scope {.init(workspace:root,projectID:projectID,sessionID:sessionID,groupID:groupID)}
    private struct LoadID:Hashable {let scope:WorkflowExperimentRecoveryStore.Scope;let refresh:Int}
    var body:some View {
        ThemedScrollView {
            VStack(alignment:.leading,spacing:12) {
                HStack {
                    Text("Experiment group").font(.headline)
                    Spacer()
                    Button("Verify group again") {refresh+=1}.buttonStyle(FlatButtonStyle())
                }
                Text(groupID).font(.caption2.monospaced()).textSelection(.enabled)
                WorkflowExperimentRecoveryControls(scope:scope,isCurrent:isCurrent) {
                    refresh+=1;onRecoveryRefresh()
                }.id(scope)
                if loadedScope == scope,let result {
                    if let receipt=result.record.recoveryReceipt {WorkflowExperimentRecoveryReceiptView(receipt:receipt)}
                    Text(result.isVerifiedComplete ? "All variants verified completed":"Review each variant's recorded state and evidence.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    ForEach(result.variants,id:\.ordinal) {variant in
                        VStack(alignment:.leading,spacing:6) {
                            Text("Variant \(variant.ordinal)").font(.caption.bold())
                            switch variant.evidence {
                            case .verified(let run):
                                Text("Verified · \(run.outcome.status.rawValue)").font(.caption)
                                Text("Seed \(run.plan.settings.seed) · Batch \(run.plan.settings.batchSize) · Epochs \(run.plan.settings.epochs) · \(run.plan.settings.device)")
                                    .font(.caption2).foregroundStyle(MLTheme.muted)
                                if let error=run.outcome.error {Text(error).font(.caption2).foregroundStyle(MLTheme.rose)}
                                Button("Inspect variant \(variant.ordinal)") {selectRun(run.reference)}.buttonStyle(FlatButtonStyle())
                            case .recorded(let status):
                                Text("Recorded · \(status.rawValue)").font(.caption)
                                Text("No verified terminal outcome. A pending or running record may belong to active or interrupted work; opening History does not restart it.")
                                    .font(.caption2).foregroundStyle(MLTheme.muted)
                                if let entry=result.record.variants.first(where:{$0.ordinal == variant.ordinal}),
                                   let directoryID=entry.directoryID.flatMap(UUID.init(uuidString:)) {
                                    Button("Inspect recorded run") {selectRun(.init(projectID:projectID,directoryID:directoryID))}
                                        .buttonStyle(FlatButtonStyle()).accessibilityLabel("Inspect recorded run for variant \(variant.ordinal)")
                                }
                            case .unavailable(let reason):
                                Text("Evidence unavailable").font(.caption).foregroundStyle(MLTheme.rose)
                                Text(reason).font(.caption2).textSelection(.enabled)
                            }
                        }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background(MLTheme.panel)
                    }
                    Text("Inspect a completed variant to pin it as a baseline, then inspect another to compare. Verification uses saved evidence; current external files and runtime are not revalidated.")
                        .font(.caption2).foregroundStyle(MLTheme.muted)
                } else {Text(message.isEmpty ? "Verifying group and run archives…":message).font(.caption).foregroundStyle(MLTheme.muted)}
            }.padding(12)
        }.task(id:LoadID(scope:scope,refresh:refresh)) {
            let requestedScope=scope
            result=nil;loadedScope=nil;message=""
            let directory=root.appendingPathComponent("workflow-experiments/\(projectID)/\(groupID)")
            let worker=Task.detached(priority:.userInitiated) {
                try await WorkflowHistoryVerificationWork.run {try WorkflowExperimentHistoryReader.load(directory:directory,workspaceDirectory:root)}
            }
            do {
                let value=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                guard !Task.isCancelled,isCurrent(),scope == requestedScope else {return};result=value;loadedScope=requestedScope
            } catch {guard !Task.isCancelled,isCurrent(),scope == requestedScope else {return};message=error.localizedDescription}
        }
    }
}
