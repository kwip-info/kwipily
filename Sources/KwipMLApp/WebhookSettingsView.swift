import SwiftUI
import AppKit
import KwipMLCore

struct WebhookSettingsView:View {
    let workspace:WorkspaceStore
    let projectID:String?
    @ObservedObject var controller:WebhookController
    @State private var page="Endpoints"
    @State private var search=""
    @State private var selected:String?
    @State private var draft:WebhookEndpoint?
    @State private var base:WebhookEndpoint?
    @State private var versions:[ScheduleVersionChoice]=[]
    @State private var details:ScheduleVersionDetails?
    @State private var loading=false
    @State private var notice:String?
    @State private var issue:String?
    @State private var settings:WebhookState?
    @State private var showingLimits=false
    @State private var credential:String?
    @State private var showCredential=false
    @State private var confirmRotate=false
    @State private var confirmTest=false
    @State private var sample="{\"items\":[{\"order_id\":1042,\"amount\":29.5}]}"
    @State private var samplePreview:[[String:String]]=[]
    @State private var generation=UUID()
    init(workspace:WorkspaceStore,projectID:String?=nil){self.workspace=workspace;self.projectID=projectID;controller = .shared(workspace:workspace)}
    private var endpoints:[WebhookEndpoint]{controller.state.endpoints.filter{(projectID==nil || $0.rule.projectID==projectID) && (search.isEmpty || ($0.name+" "+$0.rule.projectName).localizedCaseInsensitiveContains(search))}}
    private var pending:Bool {draft != nil && draft != base}
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            HStack {
                ForEach(["Endpoints","Delivery history"],id:\.self){title in Button(title){page=title}.buttonStyle(FlatButtonStyle(prominent:page==title)).disabled(pending)}
                Spacer()
                Button("Pause listener"){Task{await controller.pause()}}.buttonStyle(FlatButtonStyle()).disabled(!controller.state.enabled)
            }
            HStack {
                Toggle(projectID==nil ? "Enable webhook listener":"Enable workspace webhook listener",isOn:Binding(get:{controller.state.enabled},set:{v in Task{var s=controller.state;s.enabled=v;do{try await controller.configure(s)}catch{issue=error.localizedDescription}}})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))
                Spacer();Text(controller.status).foregroundStyle(controller.listening ? ComponentVisualFamily.data.primary:MLTheme.muted)
            }.font(.caption)
            Text("This Mac only · \(ProductBrand.name) must be open and this Mac awake. Not reachable from the Internet. Pausing preserves queued deliveries; an active job continues until cancelled.").font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
            if let notice{Text(notice).font(.caption).foregroundStyle(ComponentVisualFamily.data.primary).fixedSize(horizontal:false,vertical:true)}
            if let problem=issue ?? controller.error {Text(problem).foregroundStyle(MLTheme.rose).font(.caption).fixedSize(horizontal:false,vertical:true)}
            if page=="Delivery history" {WebhookHistoryView(controller:controller,projectID:projectID)}else{
                if projectID==nil {
                    HStack{Text("127.0.0.1 · Port \(controller.state.port)").font(.system(.caption,design:.monospaced));Spacer();Button("Listener & history limits…"){settings=controller.state;showingLimits=true}.buttonStyle(FlatButtonStyle())}
                }
                HStack{TextField("Find endpoint",text:$search).flatField();Button("New endpoint"){Task{await newEndpoint()}}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(pending || loading)}
                CapturedNativeTable(columns:WorkflowResultTableData.columns(["Enabled","Endpoint","Project","Status"]),rows:endpoints.enumerated().map {i,e in WorkflowResultTableData.row(i,[e.enabled ? "On":"Off",e.name,e.rule.projectName,controller.state.deliveries.contains{$0.endpoint.id==e.id && $0.state == .needsReview && !$0.reviewed} ? "Needs review":e.enabled ? "Ready":"Paused"])},selectedRow:Binding(get:{endpoints.firstIndex{$0.id==selected}},set:{i in guard !pending,let i,endpoints.indices.contains(i)else{return};select(endpoints[i])}),rowHeaderTitle:"#",fillsWidth:true).frame(height:170)
                if endpoints.isEmpty {Button("Create local webhook example"){do{let id=try controller.createExample();Task{await controller.scheduler.loadProjects();await newEndpoint(project:id)}}catch{issue=error.localizedDescription}}.buttonStyle(FlatButtonStyle());Text("No endpoints in this scope. Save a workflow canvas version, then add an endpoint.").font(.caption).foregroundStyle(MLTheme.muted)}
                if pending {Text("Apply or Revert before selecting another endpoint.").font(.caption).foregroundStyle(MLTheme.muted)}
                if draft != nil {editor}
            }
        }.task{do{try await controller.ready();await controller.scheduler.loadProjects()}catch{issue=error.localizedDescription}}
        .onChange(of:controller.state.endpoints){_,latest in guard !pending,let selected else{return};guard let e=latest.first(where:{$0.id==selected})else{draft=nil;base=nil;self.selected=nil;return};draft=e;base=e}
        .onChange(of:draft?.payload){_,_ in samplePreview=[]}
        .onChange(of:sample){_,_ in samplePreview=[]}
        .sheet(isPresented:$showingLimits){limitsSheet}
        .sheet(isPresented:$showCredential,onDismiss:{credential=nil}) {
            VStack(alignment:.leading,spacing:14){Text("Endpoint secret").font(.headline);Text("Keep this credential private. It permits requests to this endpoint on this Mac.").font(.caption);Text(credential ?? "Unavailable").font(.system(.body,design:.monospaced)).textSelection(.enabled);HStack{Button("Copy secret"){copy(credential ?? "")};Spacer();Button("Done"){showCredential=false}}}.padding(24).frame(width:520).background(MLTheme.background).buttonStyle(FlatButtonStyle())
        }
        .alert("Send a local test event?",isPresented:$confirmTest){Button("Cancel",role:.cancel){};Button("Run test event"){guard let base else{return};Task{do{let response=try await controller.testEvent(base,body:Data(sample.utf8));issue=nil;notice="Local test · \(response.code) · receipt \(response.delivery ?? "unavailable")"}catch{issue=error.localizedDescription}}}}message:{Text("This sends the sample through the same authentication and delivery authority inside this app and can run the saved workflow and its authorized outputs. It does not test HTTP transport. Use the client example to test an actual local HTTP sender.")}
        .alert("Rotate this endpoint's secret?",isPresented:$confirmRotate){Button("Cancel",role:.cancel){};Button("Rotate"){guard let base else{return};Task{do{try await controller.rotate(base);if let e=controller.state.endpoints.first(where:{$0.id==base.id}){select(e)}}catch{issue=error.localizedDescription}}}}message:{Text("Existing clients must use the new secret. Queued deliveries must finish or be cancelled before rotation.")}
    }
    private var editor:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Workflow",symbol:"doc",identifier:"webhook.workflow",initiallyExpanded:true){
                TextField("Endpoint name",text:binding(\.name)).flatField().accessibilityLabel("Endpoint name")
                FlatChoiceControl(title:"Webhook project",selection:draft?.rule.projectID ?? "",options:controller.scheduler.projects.filter{projectID==nil || $0.id==projectID}.map{.init(id:$0.id,title:$0.name)}){id in Task{await changeProject(id)}}.disabled(projectID != nil || loading)
                FlatChoiceControl(title:"Saved canvas version",selection:draft?.rule.versionID ?? "",options:versions.map{.init(id:$0.id,title:"Version \($0.number) · "+$0.date.formatted(date:.abbreviated,time:.shortened))}){chooseVersion($0)}.disabled(loading)
                Text("Unsaved canvas changes are not used. Each delivery supplies immutable job input.").font(.caption).foregroundStyle(MLTheme.muted)
                if let details,!details.targets.isEmpty {FlatChoiceControl(title:"Run target",selection:details.targets.first{$0.address==draft?.rule.target}?.id ?? "",options:details.targets.map{.init(id:$0.id,title:$0.title)}){id in draft?.rule.target=details.targets.first{$0.id==id}?.address}}
                Toggle("Enable this endpoint",isOn:binding(\.enabled)).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))
            }
            InspectorSection("Endpoint",symbol:"link",identifier:"webhook.endpoint",initiallyExpanded:true){
                if let draft {Text("POST  "+controller.url(for:draft)).font(.system(.caption,design:.monospaced)).textSelection(.enabled).fixedSize(horizontal:false,vertical:true)}
                HStack{Button("Copy URL"){if let draft{copy(controller.url(for:draft))}};Button("Copy client example"){if let draft{copy(controller.clientExample(draft))}};Button("Send test event…"){confirmTest=true}.disabled(!controller.listening || base?.enabled != true)}.buttonStyle(FlatButtonStyle()).disabled(base==nil || pending)
                Text("Clients send application/json and a stable X-Kwip-Delivery-ID. A 202 response means queued, not completed.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Authentication",symbol:"lock",identifier:"webhook.authentication",initiallyExpanded:true){
                FlatChoiceControl(title:"Authentication",selection:draft?.authentication.rawValue ?? "signed",options:[.init(id:"signed",title:"Signed request · HMAC-SHA256"),.init(id:"bearer",title:"Bearer token · this Mac only")]){draft?.authentication=WebhookAuthentication.Method(rawValue:$0) ?? .signed}
                Text("Secret stored securely on this Mac. Timestamped signatures reject requests older than 5 minutes. No unauthenticated mode.").font(.caption).foregroundStyle(MLTheme.muted)
                HStack{Button("Show secret…"){guard let base else{return};do{credential=try controller.secret(for:base);showCredential=true}catch{issue=error.localizedDescription}};Button("Rotate…"){confirmRotate=true}}.buttonStyle(FlatButtonStyle()).disabled(base==nil || pending)
            }
            InspectorSection("Run behavior",symbol:"arrow.triangle.2.circlepath",identifier:"webhook.behavior",initiallyExpanded:true){
                FlatChoiceControl(title:"If the workspace is busy",selection:draft?.busy.rawValue ?? "queue",options:[.init(id:"queue",title:"Queue deliveries"),.init(id:"reject",title:"Reject until idle")]){draft?.busy=WebhookEndpoint.Busy(rawValue:$0) ?? .queue}
                Text("One workflow at a time. An exact retry returns the original receipt; a reused ID with different data is refused.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Payload",symbol:"tablecells",identifier:"webhook.payload",initiallyExpanded:false){payloadEditor}
            InspectorSection("Access & output destinations",symbol:"lock",identifier:"webhook.outputs",initiallyExpanded:false){
                if let details {if details.effects.isEmpty{Text("No output destination writes in this version.").font(.caption)};ForEach(details.effects,id:\.self){Text($0).font(.caption)};if !details.effects.isEmpty {Toggle("Allow unattended writes to these configured destinations",isOn:binding(\.rule.allowEffects)).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}}
                Text("Uses existing project source grants, effective resource/capture limits and private connections. Changed output access blocks execution. Uncertain writes pause this endpoint for review.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ViewThatFits(in:.horizontal){footer;VStack(alignment:.leading,spacing:8){applyButtons;managementButtons}}
        }.font(.caption)
    }
    private var applyButtons:some View {HStack{Button("Apply"){Task{guard let value=draft else{return};do{try await controller.save(value);if let fresh=controller.state.endpoints.first(where:{$0.id==value.id}){draft=fresh;base=fresh;selected=fresh.id};issue=nil}catch{issue=error.localizedDescription}}}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending || loading);Button("Revert"){if let base{select(base)}else{draft=nil};issue=nil}.buttonStyle(FlatButtonStyle()).disabled(!pending)}}
    private var managementButtons:some View {HStack{if base != nil{Button("Remove endpoint"){guard let base else{return};Task{do{try await controller.remove(base);draft=nil;self.base=nil;selected=nil}catch{issue=error.localizedDescription}}}.buttonStyle(FlatButtonStyle()).disabled(pending)}}}
    private var footer:some View {HStack{applyButtons;Spacer();managementButtons}}
    private var tableChoices:[WorkflowNode]{details?.document.nodes.filter{if case .dataOperator(DataFlowSourceOperators.tableInput,let c)=$0.payload{return (try? TableInputConfiguration.format(of:c)) == .json};return false} ?? []}
    private var payloadEditor:some View {
        VStack(alignment:.leading,spacing:10){
            FlatChoiceControl(title:"Payload mode",selection:draft?.payload==nil ? "trigger":"table",options:[.init(id:"trigger",title:"Trigger only"),.init(id:"table",title:"JSON records into a table")]){id in draft?.payload=id=="trigger" ? nil:WebhookPayload(tableNodeID:tableChoices.first?.address.localID ?? "",recordsPath:"items",columns:[.init(field:"order_id",column:"order_id",type:.int64)])}
            if draft?.payload != nil {
                FlatChoiceControl(title:"Table Input",selection:draft?.payload?.tableNodeID ?? "",options:tableChoices.map{.init(id:$0.address.localID,title:$0.name.isEmpty ? "Table Input · "+$0.address.localID:$0.name)}){draft?.payload?.tableNodeID=$0}
                Text("Choose a Table Input configured for JSON document in this saved canvas. The mapped array uses that component's existing JSON options.").font(.caption).foregroundStyle(MLTheme.muted)
                TextField("Records path · leave empty for root array",text:Binding(get:{draft?.payload?.recordsPath ?? ""},set:{draft?.payload?.recordsPath=$0})).flatField().accessibilityLabel("Records path")
                HStack{Text("Include").frame(width:45);Text("JSON field").frame(maxWidth:.infinity,alignment:.leading);Text("Table column").frame(maxWidth:.infinity,alignment:.leading);Text("Type").frame(width:95)}.foregroundStyle(MLTheme.muted)
                ForEach(Array((draft?.payload?.columns ?? []).indices),id:\.self){i in
                    HStack(spacing:6){Toggle("",isOn:columnBinding(i,\.included)).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose)).frame(width:45).accessibilityLabel("Include mapping \(i+1)");TextField("JSON field",text:columnBinding(i,\.field)).flatField();TextField("Table column",text:columnBinding(i,\.column)).flatField();FlatChoiceControl(title:"Column type",selection:draft?.payload?.columns[i].type.rawValue ?? "string",options:WebhookPayload.ValueType.allCases.map{.init(id:$0.rawValue,title:$0.rawValue)}){draft?.payload?.columns[i].type=WebhookPayload.ValueType(rawValue:$0) ?? .string}.frame(width:95);Button{draft?.payload?.columns.remove(at:i)}label:{Image(systemName:"minus.circle")}.buttonStyle(.plain).accessibilityLabel("Remove mapping \(i+1)")}
                }
                Button("Add column"){draft?.payload?.columns.append(.init(field:"field\((draft?.payload?.columns.count ?? 0)+1)",column:"column\((draft?.payload?.columns.count ?? 0)+1)",type:.string))}.buttonStyle(FlatButtonStyle())
                TextEditor(text:$sample).font(.system(.caption,design:.monospaced)).frame(height:80).overlay(Rectangle().stroke(MLTheme.rule)).accessibilityLabel("Sample JSON payload")
                Button("Preview payload"){do{guard let mapping=draft?.payload else{return};let value=try mapping.decode(Data(sample.utf8));if case .array(let rows)=try JSONDecoder().decode(JSONValue.self,from:value.data){samplePreview=rows.prefix(8).compactMap{if case .object(let v)=$0{return v.mapValues{String(data:(try? JSONEncoder().encode($0)) ?? Data(),encoding:.utf8) ?? ""}};return nil}};issue=nil}catch{samplePreview=[];issue=error.localizedDescription}}.buttonStyle(FlatButtonStyle())
                if !samplePreview.isEmpty{let names=(draft?.payload?.columns ?? []).filter(\.included).map(\.column);CapturedNativeTable(columns:WorkflowResultTableData.columns(names),rows:samplePreview.enumerated().map{i,row in WorkflowResultTableData.row(i,names.map{row[$0] ?? ""})},selectedRow:.constant(nil),rowHeaderTitle:"#",fillsWidth:true).frame(height:120)}
                Text("Preview only. Each accepted delivery stores its own input; missing fields and wrong types are rejected. Pending inputs are kept until resolved; optional previews follow project History capture.").font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private var limitsSheet:some View {
        VStack(alignment:.leading,spacing:12){Text("Listener & delivery limits").font(.headline);Text("Bind address · 127.0.0.1 (this Mac only)").font(.caption);number("Port",\.port);number("Maximum request bytes",\.maximumBodyBytes);number("Maximum pending deliveries",\.maximumQueued);number("Requests per minute",\.requestsPerMinute);number("Retention days",\.retentionDays);number("Maximum delivery records",\.maximumDeliveries);Text("Up to 1 MiB per request and 32 pending deliveries. Resolved records expire first; queued and unresolved evidence is protected. An active job continues if the listener is paused.").font(.caption).foregroundStyle(MLTheme.muted);HStack{Button("Cancel"){showingLimits=false};Spacer();Button("Apply"){Task{guard let settings else{return};do{try await controller.configure(settings);showingLimits=false;issue=nil}catch{issue=error.localizedDescription;showingLimits=false}}}}.buttonStyle(FlatButtonStyle())}.padding(22).frame(width:460).background(MLTheme.background)
    }
    private func number(_ title:String,_ path:WritableKeyPath<WebhookState,Int>)->some View{HStack{Text(title);Spacer();TextField(title,value:Binding(get:{settings?[keyPath:path] ?? 0},set:{settings?[keyPath:path]=$0}),format:.number).flatField().frame(width:145)}}
    private func binding<T>(_ path:WritableKeyPath<WebhookEndpoint,T>)->Binding<T>{let fallback=draft![keyPath:path];return Binding(get:{draft?[keyPath:path] ?? fallback},set:{draft?[keyPath:path]=$0})}
    private func columnBinding<T>(_ index:Int,_ path:WritableKeyPath<WebhookPayload.Column,T>)->Binding<T>{let fallback=draft!.payload!.columns[index][keyPath:path];return Binding(get:{guard let columns=draft?.payload?.columns,columns.indices.contains(index)else{return fallback};return columns[index][keyPath:path]},set:{guard draft?.payload?.columns.indices.contains(index)==true else{return};draft?.payload?.columns[index][keyPath:path]=$0})}
    private func copy(_ text:String){NSPasteboard.general.clearContents();NSPasteboard.general.setString(text,forType:.string)}
    private func select(_ e:WebhookEndpoint){selected=e.id;draft=e;base=e;issue=nil;samplePreview=[];Task{await loadVersions(e.rule.projectID)}}
    private func loadVersions(_ project:String)async{loading=true;defer{loading=false};let token=UUID();generation=token;do{let v=try await controller.scheduler.versions(projectID:project);guard generation==token,draft?.rule.projectID==project else{return};versions=v;chooseVersion(draft?.rule.versionID ?? "")}catch{issue=error.localizedDescription}}
    private func chooseVersion(_ id:String){guard let version=versions.first(where:{$0.id==id})else{return};if draft?.rule.versionID != id{draft?.rule.allowEffects=false;draft?.payload=nil};draft?.rule.versionID=id;draft?.rule.versionNumber=version.number;draft?.rule.versionHash=version.hash;details=nil;let project=draft?.rule.projectID ?? "",token=UUID();generation=token;Task{do{let v=try await controller.scheduler.details(projectID:project,versionID:id);guard generation==token,draft?.rule.versionID==id else{return};details=v;if !v.targets.contains(where:{$0.address==draft?.rule.target}){draft?.rule.target=v.targets.count==1 ? v.targets[0].address:nil}}catch{issue=error.localizedDescription}}}
    private func changeProject(_ id:String)async{guard let project=controller.scheduler.projects.first(where:{$0.id==id})else{return};draft?.rule.projectID=id;draft?.rule.projectName=project.name;draft?.rule.allowEffects=false;draft?.payload=nil;await loadVersions(id);if let first=versions.first{chooseVersion(first.id)}else{issue="Explicitly Save this canvas before creating an endpoint."}}
    private func newEndpoint(project preferred:String?=nil)async{await controller.scheduler.loadProjects();guard let project=controller.scheduler.projects.first(where:{if let requested=projectID ?? preferred{return $0.id==requested};return true})else{issue="Open and Save a managed workflow project first.";return};let id=UUID().uuidString;draft=WebhookEndpoint(id:id,rule:ScheduleRule(id:id,projectID:project.id,projectName:project.name,versionID:"",versionNumber:1,versionHash:"",name:"Webhook",timing:.init()));base=nil;selected=nil;await changeProject(project.id)}
}
