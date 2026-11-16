import SwiftUI
import KwipMLCore

struct WebhookHistoryView:View {
    @ObservedObject var controller:WebhookController
    let projectID:String?
    @State private var search=""
    @State private var status="all"
    @State private var selected:String?
    @State private var review=false
    @State private var issue:String?
    @State private var showJob=false
    private var deliveries:[WebhookDelivery]{controller.state.deliveries.filter{(projectID==nil || $0.endpoint.rule.projectID==projectID) && (status=="all" || $0.state.rawValue==status) && (search.isEmpty || ($0.endpoint.name+" "+$0.deliveryID+" "+$0.endpoint.rule.projectName).localizedCaseInsensitiveContains(search))}}
    private var detail:WebhookDelivery?{deliveries.first{$0.id==selected}}
    var body:some View {
        VStack(alignment:.leading,spacing:12){
            HStack{TextField("Find delivery or project",text:$search).flatField();FlatChoiceControl(title:"Delivery status",selection:status,options:[.init(id:"all",title:"All statuses")]+["queued","running","completed","failed","rejected","needsReview","cancelled"].map{.init(id:$0,title:$0=="needsReview" ? "Needs review":$0.capitalized)}){status=$0}.frame(maxWidth:190)}
            CapturedNativeTable(columns:WorkflowResultTableData.columns(["Received","Endpoint","Status","Duplicates","Job"]),rows:deliveries.prefix(100).enumerated().map{i,d in WorkflowResultTableData.row(i,[d.receivedAt.formatted(date:.abbreviated,time:.standard),d.endpoint.name,d.title,String(d.duplicates),d.jobID.map{String($0.prefix(8))} ?? "—"])},selectedRow:Binding(get:{deliveries.firstIndex{$0.id==selected}},set:{i in if let i,deliveries.indices.contains(i){selected=deliveries[i].id;showJob=false}}),rowHeaderTitle:"#",fillsWidth:true).frame(height:210)
            Text("Latest 100 matching deliveries · \(controller.state.deliveries.count) retained. Resolved records expire after \(controller.state.retentionDays) days / \(controller.state.maximumDeliveries) records; pending and unresolved records are protected.").font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
            if let d=detail {
                InspectorSection("Delivery "+d.deliveryID,symbol:"tray",identifier:"webhook.delivery",initiallyExpanded:true){
                    CapturedNativeTable(columns:WorkflowResultTableData.columns(["Field","Captured value"]),rows:[("Receipt",d.id),("Project",d.endpoint.rule.projectName),("Saved canvas","Version \(d.endpoint.rule.versionNumber)"),("Source",d.origin),("Acceptance",String(d.httpStatus)+" · Accepted means queued, not completed"),("Authentication",d.state == .rejected ? "See rejection":"Verified · "+d.endpoint.authentication.rawValue),("Input",d.rows.map{String($0)+" mapped rows"} ?? "Trigger only"),("Job",d.title),("Result",d.message)].enumerated().map{WorkflowResultTableData.row($0.offset,[$0.element.0,$0.element.1])},selectedRow:.constant(nil),rowHeaderTitle:"#",fillsWidth:true).frame(height:230)
                    if d.state == .needsReview {Text(d.reviewed ? "Outcome acknowledged. Enable the endpoint explicitly after reviewing access.":"Delivery needs review. Endpoint paused; check the job and output destination before another attempt.").font(.caption).foregroundStyle(.orange).fixedSize(horizontal:false,vertical:true)}
                    HStack{
                        if let job=controller.scheduler.state.jobs.first(where:{$0.id==d.jobID}) {Button(showJob ? "Hide job":"Open job"){showJob.toggle()}.buttonStyle(FlatButtonStyle());if job.runID != nil{Button("Open project History"){controller.scheduler.openProjectHistory(job)}.buttonStyle(FlatButtonStyle())}}
                        if [.queued,.running].contains(d.state){Button("Cancel delivery"){Task{await controller.cancel(d)}}.buttonStyle(FlatButtonStyle())}
                        if d.state == .needsReview && !d.reviewed {Button("Review outcome…"){review=true}.buttonStyle(FlatButtonStyle())}
                    }
                    InspectorSection("Payload preview",symbol:"tablecells",identifier:"webhook.preview",initiallyExpanded:false){if let bytes=d.preview{Text(String(decoding:bytes,as:UTF8.self)).font(.system(.caption,design:.monospaced)).textSelection(.enabled).fixedSize(horizontal:false,vertical:true)}else{Text(d.state == .queued || d.state == .running ? "Required input is retained privately until this job resolves. Optional preview is off.":"No optional payload preview retained. Required run inputs follow project History/replay rules.").font(.caption).foregroundStyle(MLTheme.muted)}}
                    Text("Inspecting this receipt never repeats a request. Exact retries are deduplicated within retained delivery history.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                if showJob,let job=controller.scheduler.state.jobs.first(where:{$0.id==d.jobID}){ScheduledJobDetailView(controller:controller.scheduler,job:job)}
            }
            if let issue{Text(issue).font(.caption).foregroundStyle(MLTheme.rose)}
        }.alert("Have you reviewed this delivery's outcome?",isPresented:$review){Button("Cancel",role:.cancel){};Button("Acknowledge outcome"){guard let d=detail else{return};Task{do{try await controller.review(d);issue=nil}catch{issue=error.localizedDescription}}}}message:{Text("This neither retries the request nor proves whether a destination write committed. The endpoint stays paused until you explicitly enable it.")}
    }
}
