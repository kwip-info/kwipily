import SwiftUI
import KwipMLCore

struct SelfPlayOutputChoices:View {
    @Binding var port:String
    var ports=SelfPlayToolContract.tablePorts
    var body:some View {FlatChoiceControl(title:"Captured output",selection:port,options:ports.map{.init(id:$0,title:$0.capitalized)},compact:true){port=$0}}
}
struct SelfPlayResultsView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    let address:WorkflowNodeAddress
    @State private var port="iterations"
    var body:some View {
        VStack(alignment:.leading,spacing:10){SelfPlayOutputChoices(port:$port,ports:SelfPlayToolContract.tablePorts(document:host.owner.document,address:address))
            if let session=host.selfPlay,session.address==address{SelfPlayStatusView(session:session)}
            if let query=host.selfPlayTableViewSession(for:.init(address:address,port:port)){TableViewOutputView(session:query,executionBusy:host.owner.isBusy).id((host.selfPlay?.id.uuidString ?? "")+port)}
            else{Text("Run Self-play to publish complete bounded tables. Training charts are in Metrics; evaluation is independent of collection.").font(.caption).foregroundStyle(MLTheme.muted)}
        }
    }
}
struct SelfPlayMetricsView:View {
    let rows:[[String:JSONValue]]
    var running=false
    var recorded=false
    @State private var component="loss"
    @State private var projected:[TrainingProgressPoint]=[]
    private var points:[TrainingProgressPoint] {
        var result:[TrainingProgressPoint]=[]
        for (index,row) in rows.enumerated() {
            guard let loss=row[component]?.doubleValue,loss.isFinite else{continue}
            let iteration=Int(row["iteration"]?.integerValue ?? 1)
            let update=Int(row["update"]?.integerValue ?? 1)
            let updates=Int(row["updates"]?.integerValue ?? 1)
            let total=row["totalUpdates"]?.integerValue ?? (index+1)
            let moe:MoETrainingObjective? = component=="loss" ? row["balancePenalty"]?.doubleValue.flatMap{b in row["routerZPenalty"]?.doubleValue.flatMap{z in row["trainingObjective"]?.doubleValue.map{.init(balancePenalty:b,routerZPenalty:z,trainingObjective:$0)}}}:nil
            let point=TrainingProgressPoint(event:"selfPlayProgress",schemaVersion:1,runID:"self-play",sequence:index+1,epoch:iteration,epochBatch:update,epochBatches:updates,batches:total,updates:total,elapsedSeconds:row["elapsedSeconds"]?.doubleValue ?? 0,kind:"training",trainingLoss:loss,moe:moe,numerator:nil,denominator:nil,validationLoss:nil,accuracy:nil)
            result.append(point)
        }
        return result
    }
    var body:some View {VStack(alignment:.leading,spacing:10){FlatChoiceControl(title:"Loss component",selection:component,options:[.init(id:"loss",title:"Joint loss"),.init(id:"policyLoss",title:"Policy loss"),.init(id:"valueLoss",title:"Value loss")]){component=$0};TrainingMetricsChart(metrics:[],progress:projected,observed:rows.count,recorded:recorded,running:running,selfPlay:true);Text("Required captured updates · evaluation attempts and paired uncertainty remain in standard tables. A lower training loss is not proof of game strength.").font(.caption).foregroundStyle(MLTheme.muted)}.onAppear{projected=points}.onChange(of:rows){_,_ in projected=points}.onChange(of:component){_,_ in projected=points}}
}
struct SelfPlayHistoryView:View {
    let record:SelfPlayRecord
    @ObservedObject var workspace:WorkspaceStore
    let isActive:()->Bool
    @State private var port="iterations"
    @State private var tab="outputs"
    @State private var message="Verifying captured Self-play…"
    @State private var artifacts:[String:TableViewQuerySession.Artifact]=[:]
    @State private var measurements:[[String:JSONValue]]=[]
    @State private var query:TableViewQuerySession?
    var body:some View {VStack(alignment:.leading,spacing:10){HStack{Button("Outputs"){tab="outputs"}.buttonStyle(FlatButtonStyle(prominent:tab=="outputs"));Button("Metrics"){tab="metrics"}.buttonStyle(FlatButtonStyle(prominent:tab=="metrics"))}
        if tab=="metrics"{SelfPlayMetricsView(rows:measurements,recorded:true)}else{SelfPlayOutputChoices(port:$port,ports:SelfPlayToolContract.tablePorts(document:record.document,address:record.address));if let query{TableViewOutputView(session:query,executionBusy:workspace.isBusy)}}
        if !message.isEmpty{Text(message).font(.caption).foregroundStyle(MLTheme.muted)}
    }.task(id:record.id){do{let root=workspace.workspaceDirectory;let loaded=try await WorkflowHistoryVerificationWork.run{()->([String:TableViewQuerySession.Artifact],[[String:JSONValue]]) in
        let artifacts=try record.verify(root:root),directory=try record.directory(root)
        let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["artifacts","self-play.json"],limit:8_000_000))
        let measurements=result["measurements"].flatMap{try? JSONDecoder().decode([[String:JSONValue]].self,from:JSONEncoder().encode($0))} ?? [];return (artifacts,measurements)
    };guard !Task.isCancelled,isActive()else{return};artifacts=loaded.0;measurements=loaded.1;message="Verified recorded rules, paired weights and required replay · no live model substituted.";select()}catch{message=error.localizedDescription}}
    .onChange(of:port){_,_ in select()}.onDisappear{query?.cancelIfUnowned()}}
    private func select(){query?.cancelIfUnowned();guard let a=artifacts[port]else{query=nil;return};query=workspace.makeTableViewQuerySession(artifact:a,projectID:record.document.projectID,accepts:isActive)}
}
