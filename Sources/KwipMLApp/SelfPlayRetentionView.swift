import SwiftUI
import KwipMLCore

struct SelfPlayRetentionView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    let address:WorkflowNodeAddress
    @State private var boundaries:[SelfPlayCampaignGrant.Boundary]=[]
    @State private var selected=""
    @State private var message=""
    var body:some View {
        VStack(alignment:.leading,spacing:10){
            Text("Retained campaign boundaries").font(.headline)
            if boundaries.isEmpty{Text("No campaign boundary available yet.").font(.caption).foregroundStyle(MLTheme.muted)}else{
                FlatChoiceControl(title:"Retained boundary",selection:selected,options:boundaries.map{.init(id:$0.id,title:$0.id+($0.pinned ? " · pinned":" · recent"))}){selected=$0}
                Button("Pin selected boundary"){change(true)}.buttonStyle(FlatButtonStyle()).disabled(host.hasActiveOperation)
                Button("Unpin selected boundary"){change(false)}.buttonStyle(FlatButtonStyle()).disabled(host.hasActiveOperation || !boundaries.contains(where:{$0.id==selected && $0.pinned}))
                Text("Pinned boundaries keep their required replay. Unpinning does not delete immediately; the next admitted commit may prune an old boundary under the configured recent-window policy. History result segments and portable inference exports remain separate.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            if !message.isEmpty{Text(message).font(.caption).foregroundStyle(MLTheme.muted)}
        }.task{await reload()}
    }
    private func reload()async {
        guard let record=host.selfPlayRecords.first(where:{$0.address==address}),let root=host.selfPlayWorkspaceRoot else{return}
        do{boundaries=try await WorkflowHistoryVerificationWork.run{try SelfPlayCampaignGrant.boundaries(record,root:root)};if !boundaries.contains(where:{$0.id==selected}){selected=boundaries.first?.id ?? ""}}catch{message=error.localizedDescription}
    }
    private func change(_ value:Bool){
        guard !host.hasActiveOperation,let record=host.selfPlayRecords.first(where:{$0.address==address}),let root=host.selfPlayWorkspaceRoot else{return}
        Task {do{try await WorkflowHistoryVerificationWork.run{try SelfPlayCampaignGrant.pin(record,root:root,boundary:selected,value:value)};message=value ? "Boundary pinned.":"Pin removed; no files deleted now.";await reload()}catch{message=error.localizedDescription}}
    }
}
