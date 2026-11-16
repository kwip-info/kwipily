import SwiftUI
import KwipMLCore

struct ScheduledJobHistoryView:View {
    @ObservedObject var controller:ScheduledRunController
    let projectID:String?
    @State private var query=""
    @State private var status="all"
    @State private var project="all"
    @State private var page=0
    @State private var selected:String?
    @State private var limits=false
    @State private var days=30
    @State private var maximum=1000
    @State private var issue:String?
    private var filtered:[ScheduledJob] {controller.state.jobs.filter {j in
        (projectID==nil || j.rule.projectID==projectID) && (project=="all" || j.rule.projectID==project) && (status=="all" || j.state.rawValue==status) && (query.isEmpty || (j.rule.name+" "+j.rule.projectName+" "+j.message).localizedCaseInsensitiveContains(query))
    }}
    private var visible:[ScheduledJob] {Array(filtered.dropFirst(page*50).prefix(50))}
    private var chosen:ScheduledJob? {visible.first{$0.id==selected}}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            TextField("Find job or project",text:$query).flatField().onChange(of:query){_,_ in page=0;selected=nil}
            HStack {
                if projectID==nil {FlatChoiceControl(title:"Job project filter",selection:project,options:[.init(id:"all",title:"All projects")]+controller.projects.map {.init(id:$0.id,title:$0.name)}) {project=$0;page=0;selected=nil}}
                FlatChoiceControl(title:"Job status filter",selection:status,options:[.init(id:"all",title:"All statuses")]+ScheduledJob.State.allCases.map {.init(id:$0.rawValue,title:$0.title)}) {status=$0;page=0;selected=nil}
                Button("History limits…") {days=controller.state.retentionDays;maximum=controller.state.maximumJobs;limits=true}.buttonStyle(FlatButtonStyle())
            }
            CapturedNativeTable(columns:WorkflowResultTableData.columns(["Started","Schedule","Project","Result","Duration"]),rows:visible.enumerated().map {i,j in WorkflowResultTableData.row(i,[(j.startedAt ?? j.dueAt).formatted(date:.abbreviated,time:.shortened),j.rule.name,j.rule.projectName,j.state.title,j.duration.map {String(format:"%.1fs",$0)} ?? "—"])},selectedRow:Binding(get:{visible.firstIndex{$0.id==selected}},set:{i in if let i,visible.indices.contains(i) {selected=visible[i].id}}),rowHeaderTitle:"#",fillsWidth:true).frame(height:230)
            HStack {Text("\(filtered.count) jobs · Local to this Mac").font(.caption).foregroundStyle(MLTheme.muted);Spacer();Button("Previous") {page=max(0,page-1);selected=nil}.disabled(page==0);Text("\(page+1)").font(.caption);Button("Next") {page+=1;selected=nil}.disabled((page+1)*50>=filtered.count)}.buttonStyle(FlatButtonStyle())
            if let chosen {ScheduledJobDetailView(controller:controller,job:chosen)}else {Text("Select a job to inspect its captured outcome. Opening History never runs a workflow.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let issue {Text(issue).foregroundStyle(MLTheme.rose).font(.caption)}
        }.sheet(isPresented:$limits) {
            VStack(alignment:.leading,spacing:14) {
                Text("Job history limits").font(.headline)
                HStack {Text("Days");TextField("Days",value:$days,format:.number).flatField()}
                HStack {Text("Maximum jobs");TextField("Maximum jobs",value:$maximum,format:.number).flatField()}
                Text("Trims completed job metadata. Active and unresolved jobs are protected. Existing project/model/run retention settings remain separate.").font(.caption).foregroundStyle(MLTheme.muted)
                HStack {Button("Cancel") {limits=false};Spacer();Button("Apply") {Task {do {try await controller.limits(days:days,maximum:maximum);limits=false;issue=nil}catch{issue=error.localizedDescription;limits=false}}}}.buttonStyle(FlatButtonStyle())
            }.padding(20).frame(width:420).background(MLTheme.background)
        }
    }
}
struct ScheduledJobDetailView:View {
    @ObservedObject var controller:ScheduledRunController
    let job:ScheduledJob
    @State private var tab="Overview"
    @State private var issue:String?
    @State private var confirm=false
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {Text(job.rule.name).font(.headline);Spacer();Text(job.state.title).foregroundStyle(job.state == .needsReview ? Color.orange:job.state == .completed ? ComponentVisualFamily.data.primary:MLTheme.muted)}
            Text(job.rule.projectName+" · Saved canvas v\(job.rule.versionNumber)").font(.caption).foregroundStyle(MLTheme.muted)
            HStack {ForEach(["Overview","Activity","Diagnostics","Outputs"],id:\.self) {label in Button(label){tab=label}.buttonStyle(FlatButtonStyle(prominent:tab==label))}}
            if job.state == .needsReview {Text(job.reviewed ? "Outcome reviewed. The schedule remains paused until you explicitly enable it.":"Output delivery needs review. This schedule is paused; check the destination and its receipts before a new write.").font(.caption).foregroundStyle(.orange).fixedSize(horizontal:false,vertical:true)}
            if tab=="Overview" {
                CapturedNativeTable(columns:WorkflowResultTableData.columns(["Field","Captured value"]),rows:[("Trigger",job.trigger),("Workflow",job.rule.projectName),("Version","Saved canvas v\(job.rule.versionNumber)"),("Started",job.startedAt?.formatted(date:.abbreviated,time:.standard) ?? "Not started"),("Finished",job.finishedAt?.formatted(date:.abbreviated,time:.standard) ?? "—"),("Capture",job.captureDescription.isEmpty ? "No execution capture":job.captureDescription),("Result",job.message)].enumerated().map {WorkflowResultTableData.row($0.offset,[$0.element.0,$0.element.1])},selectedRow:.constant(nil),rowHeaderTitle:"#",fillsWidth:true).frame(height:220)
                if !job.effectivePolicy.isEmpty {InspectorSection("Effective resource policy",symbol:"slider.horizontal.3",identifier:"job.policy",initiallyExpanded:false) {Text(String(data:(try? WorkflowArchiveCodec.encode(job.effectivePolicy)) ?? Data(),encoding:.utf8) ?? "Policy unavailable").font(.system(.caption,design:.monospaced)).textSelection(.enabled)}}
            }else if tab=="Activity" {
                if job.activity.isEmpty {Text("No optional activity retained for this job.").font(.caption).foregroundStyle(MLTheme.muted)}
                else {CapturedNativeTable(columns:WorkflowResultTableData.columns(["Captured event"]),rows:job.activity.enumerated().map {WorkflowResultTableData.row($0.offset,[$0.element])},selectedRow:.constant(nil),rowHeaderTitle:"#",fillsWidth:true).frame(height:240)}
            }else if tab=="Diagnostics" {
                Text(job.message.isEmpty ? "No job-level diagnostic recorded.":job.message).font(.caption).textSelection(.enabled).fixedSize(horizontal:false,vertical:true)
            }else {ScheduledJobOutputsView(controller:controller,job:job)}
            if controller.activeJobID==job.id {Text(controller.phase).font(.caption).foregroundStyle(MLTheme.muted);Button("Cancel job") {controller.cancel(job.id)}.buttonStyle(FlatButtonStyle())}
            HStack {
                Button("Open saved canvas as copy") {Task {do {try await controller.openCanvasCopy(job);issue=nil}catch{issue=error.localizedDescription}}}.buttonStyle(FlatButtonStyle())
                if job.runID != nil {Button("Open project History") {controller.openProjectHistory(job)}.buttonStyle(FlatButtonStyle())}
                if job.state == .needsReview && !job.reviewed {Button("Review outcome…") {confirm=true}.buttonStyle(FlatButtonStyle()).disabled(controller.activeJobID != nil)}
            }
            Text("Opening captured output never resends a request. A new run is a new operation.").font(.caption2).foregroundStyle(MLTheme.muted)
            if let issue {Text(issue).foregroundStyle(MLTheme.rose).font(.caption)}
        }.padding(12).overlay(Rectangle().stroke(MLTheme.rule)).alert("Have you checked the output destinations?",isPresented:$confirm) {
            Button("Cancel",role:.cancel){}
            Button("I checked · acknowledge outcome") {Task {do {try await controller.review(job);issue=nil}catch{issue=error.localizedDescription}}}
        }message:{Text("This acknowledges the recorded uncertainty; it does not prove delivery or retry any request. The schedule remains paused until you explicitly enable it.")}
    }
}
private struct ScheduledJobOutputsView:View {
    @ObservedObject var controller:ScheduledRunController
    let job:ScheduledJob
    @State private var records:[DatabaseOutputLedger.Record]=[]
    @State private var preparation:(WorkflowDataPreparationHistory,WorkflowDataPreparationHistory.Entry)?
    @State private var issue:String?
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            ForEach(records) {r in if r.configuration["kind"] == .string("api") {APIWriteReceiptView(record:r)}else {DatabaseOutputReceiptView(record:r)}}
            if let preparation {WorkflowDataPreparationHistoryView(history:preparation.0,entry:preparation.1,isActive:{true}).frame(minHeight:220)}
            else if records.isEmpty {Text(job.state.active ? "Captured outputs appear when available.":"No preparation/write output available here. Retained model results remain in the project's recorded History.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let issue {Text(issue).foregroundStyle(MLTheme.rose).font(.caption)}
        }.task(id:job.id+"/"+(job.runID ?? "")+job.state.rawValue) {
            do {records=try await controller.receipts(for:job);preparation=await controller.preparation(for:job);issue=nil}catch{issue=error.localizedDescription}
        }
    }
}
