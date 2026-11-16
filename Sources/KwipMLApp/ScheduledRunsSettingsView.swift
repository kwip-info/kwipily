import SwiftUI
import KwipMLCore

/// One shared controller, two scopes. Async selections populate cached view models;
/// body/layout never reads files, reconstructs versions or dispatches work.
struct ScheduledRunsSettingsView:View {
    let workspace:WorkspaceStore
    let projectID:String?
    @ObservedObject var controller:ScheduledRunController
    @State private var page="schedules"
    @State private var search=""
    @State private var selected:String?
    @State private var draft=ScheduleRule(projectID:"",projectName:"",versionID:"",versionNumber:1,versionHash:"",name:"New schedule",timing:.init())
    @State private var base:ScheduleRule?
    @State private var editing=false
    @State private var versions:[ScheduleVersionChoice]=[]
    @State private var details:ScheduleVersionDetails?
    @State private var loading=false
    @State private var issue:String?
    @State private var generation=UUID()
    init(workspace:WorkspaceStore,projectID:String?=nil) {self.workspace=workspace;self.projectID=projectID;controller = .shared(workspace:workspace)}
    private var rules:[ScheduleRule] {controller.state.rules.filter {(projectID==nil || $0.projectID==projectID) && (search.isEmpty || ($0.name+" "+$0.projectName).localizedCaseInsensitiveContains(search))}}
    private var pending:Bool {editing && base != draft}
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            HStack {
                Button("Schedules") {page="schedules"}.buttonStyle(FlatButtonStyle(prominent:page=="schedules"))
                Button("Job history") {page="history"}.buttonStyle(FlatButtonStyle(prominent:page=="history"))
                Spacer()
                if projectID==nil {Button("Pause all") {Task {await controller.setEnabled(false)}}.buttonStyle(FlatButtonStyle()).disabled(!controller.state.enabled)}
            }
            HStack {
                if projectID==nil {Toggle("Enable scheduled runs",isOn:Binding(get:{controller.state.enabled},set:{v in Task {await controller.setEnabled(v)}})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}
                else {Text(controller.state.enabled ? "Workspace scheduling enabled":"Workspace scheduling paused");if !controller.state.enabled {Button("Enable workspace scheduling") {Task {await controller.setEnabled(true)}}.buttonStyle(FlatButtonStyle())}}
                Spacer();Text(controller.activeJobID==nil ? "One run at a time":"Job active").foregroundStyle(MLTheme.muted)
            }.font(.caption)
            Text("Runs on this Mac while \(ProductBrand.name) is open and the Mac awake. Pausing stops queued jobs; an active job continues until cancelled.").font(.caption).foregroundStyle(MLTheme.muted)
            if let error=issue ?? controller.error {Text(error).font(.caption).foregroundStyle(MLTheme.rose).fixedSize(horizontal:false,vertical:true)}
            if page=="history" {ScheduledJobHistoryView(controller:controller,projectID:projectID)} else {schedules}
        }.task {do {try await controller.ready();await controller.loadProjects()}catch {issue=error.localizedDescription}}
        .onChange(of:controller.state.rules) {_,latest in
            guard editing,!pending,let selected else{return}
            guard let fresh=latest.first(where:{$0.id==selected}) else{editing=false;self.selected=nil;return}
            let versionChanged=draft.versionID != fresh.versionID
            draft=fresh;base=fresh
            if versionChanged {Task {await loadVersions(project:fresh.projectID)}}
        }
    }
    private var schedules:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {TextField("Find schedule",text:$search).flatField();Button("New schedule") {Task {await newSchedule()}}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(loading || pending)}
            CapturedNativeTable(columns:WorkflowResultTableData.columns(["Enabled","Schedule","Project","Next run","Status"]),rows:rules.enumerated().map {i,r in
                let review=controller.state.jobs.contains{$0.rule.id==r.id && $0.state == .needsReview && !$0.reviewed}
                return WorkflowResultTableData.row(i,[r.enabled ? "On":"Off",r.name,r.projectName,r.nextDue?.formatted(date:.abbreviated,time:.shortened) ?? "—",review ? "Needs review":r.enabled ? "Ready":"Paused"])
            },selectedRow:Binding(get:{rules.firstIndex{$0.id==selected}},set:{i in guard !pending,let i,rules.indices.contains(i) else{return};select(rules[i])}),rowHeaderTitle:"#",fillsWidth:true).frame(height:190)
            if rules.isEmpty {Text("No schedules in this scope. Save a workflow canvas version, then add a schedule.").font(.caption).foregroundStyle(MLTheme.muted)}
            if pending {Text("Apply or Revert this draft before selecting another schedule.").font(.caption).foregroundStyle(MLTheme.muted)}
            if editing {editor}
        }
    }
    private var editor:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Workflow",symbol:"doc",identifier:"schedule.workflow",initiallyExpanded:true) {
                TextField("Schedule name",text:binding(\.name)).flatField().accessibilityLabel("Schedule name")
                FlatChoiceControl(title:"Scheduled project",selection:draft.projectID,options:controller.projects.filter {projectID==nil || $0.id==projectID}.map {.init(id:$0.id,title:$0.name)}) {id in Task {await changeProject(id)}}.disabled(projectID != nil || loading)
                FlatChoiceControl(title:"Saved canvas version",selection:draft.versionID,options:versions.map {.init(id:$0.id,title:"Version \($0.number) · "+$0.date.formatted(date:.abbreviated,time:.shortened)+($0.label.isEmpty ? "":" · "+$0.label))}) {id in chooseVersion(id)}.disabled(loading)
                Text("Unsaved canvas changes are not scheduled. Private sources are captured fresh for each attempt.").font(.caption).foregroundStyle(MLTheme.muted)
                if let details,!details.targets.isEmpty {FlatChoiceControl(title:"Run target",selection:details.targets.first(where:{$0.address==draft.target})?.id ?? "",options:details.targets.map {.init(id:$0.id,title:$0.title)}) {id in draft.target=details.targets.first{$0.id==id}?.address}}
                Toggle("Enable this schedule",isOn:binding(\.enabled)).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))
            }
            InspectorSection("Timing",symbol:"calendar",identifier:"schedule.timing",initiallyExpanded:true) {
                choice("Frequency",path:\.timing.kind,options:ScheduleTiming.Kind.allCases.map {($0,$0 == .interval ? "Every interval":$0.rawValue.capitalized)})
                if [.once,.interval].contains(draft.timing.kind) {DatePicker(draft.timing.kind == .once ? "Run at":"First run",selection:binding(\.timing.anchor),displayedComponents:[.date,.hourAndMinute]).datePickerStyle(.field).environment(\.timeZone,TimeZone(identifier:draft.timing.timeZone) ?? .current)}
                if draft.timing.kind == .interval {HStack {Text("Minutes between runs");TextField("15",value:binding(\.timing.intervalMinutes),format:.number).flatField()}}
                if [.daily,.weekly].contains(draft.timing.kind) {
                    if draft.timing.kind == .weekly {choice("Day",path:\.timing.weekday,options:Array(Calendar.current.weekdaySymbols.enumerated()).map {($0.offset+1,$0.element)})}
                    HStack {Text("At");TextField("Hour",value:binding(\.timing.hour),format:.number).flatField();Text(":");TextField("Minute",value:binding(\.timing.minute),format:.number).flatField()}
                }
                VStack(alignment:.leading,spacing:4) {Text("Time zone");TextField("America/Chicago",text:binding(\.timing.timeZone)).flatField()}
                if let next=try? draft.timing.next(after:Date()) {Text("Next occurrence · "+formatted(next)).font(.caption).foregroundStyle(MLTheme.muted)}
            }
            InspectorSection("Run behavior",symbol:"arrow.triangle.2.circlepath",identifier:"schedule.behavior",initiallyExpanded:true) {
                choice("If the workspace is busy",path:\.busy,options:[(.skip,"Skip occurrence"),(.latest,"Queue latest only")])
                choice("After sleep or app closure",path:\.missed,options:[(.skip,"Skip missed runs"),(.latest,"Run latest once on return")])
            }
            InspectorSection("Execution & capture",symbol:"slider.horizontal.3",identifier:"schedule.execution",initiallyExpanded:false) {
                Text("Uses this project's effective resource and capture settings and the workspace managed-storage budget. Each job records its effective policy. Optional capture off does not erase required model or write evidence.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Access & output destinations",symbol:"lock",identifier:"schedule.access",initiallyExpanded:false) {
                if let details {
                    if details.effects.isEmpty {Text("No output destination writes in this saved canvas.").font(.caption)}
                    ForEach(details.effects,id:\.self) {Text($0).font(.caption)}
                    if !details.effects.isEmpty {Toggle("Allow unattended use of these configured output destinations",isOn:binding(\.allowEffects)).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}
                }
                Text("Uses existing local source grants and private connections. Missing or changed destination access blocks the job; review and Apply again after changing grants or connections. Uncertain writes pause this rule for review; no automatic write retry.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            HStack {
                Button("Apply") {Task {do {try await controller.save(draft);base=controller.state.rules.first{$0.id==draft.id};if let base {draft=base};selected=draft.id;issue=nil}catch{issue=error.localizedDescription}}}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending || loading)
                Button("Revert") {if let fresh=controller.state.rules.first(where:{$0.id==draft.id}) {draft=fresh;base=fresh;chooseVersion(fresh.versionID)}else{editing=false};issue=nil}.buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer()
                if base != nil {
                    Button("Run now") {Task {await controller.runNow(draft.id)}}.buttonStyle(FlatButtonStyle()).disabled(pending || controller.activeJobID != nil).help("Run this saved version once without enabling recurring scheduling")
                    Button("Duplicate") {draft.id=UUID().uuidString;draft.revision=0;draft.name += " copy";draft.enabled=false;base=nil;selected=nil}.buttonStyle(FlatButtonStyle()).disabled(pending)
                    Button("Remove rule") {let id=draft.id;Task {await controller.remove(id);if !controller.state.rules.contains(where:{$0.id==id}) {editing=false;selected=nil}}}.buttonStyle(FlatButtonStyle()).disabled(pending || controller.activeJobID != nil)
                }
            }
        }.font(.caption)
    }
    private func formatted(_ date:Date)->String {var style=Date.FormatStyle(date:.abbreviated,time:.shortened);style.timeZone=TimeZone(identifier:draft.timing.timeZone) ?? .current;return date.formatted(style)+" · "+draft.timing.timeZone}
    private func binding<T>(_ path:WritableKeyPath<ScheduleRule,T>)->Binding<T> {Binding(get:{draft[keyPath:path]},set:{draft[keyPath:path]=$0})}
    private func choice<T:Equatable>(_ title:String,path:WritableKeyPath<ScheduleRule,T>,options:[(T,String)])->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:options.firstIndex{$0.0==draft[keyPath:path]}.map(String.init) ?? "",options:options.enumerated().map {.init(id:String($0.offset),title:$0.element.1)}) {id in if let i=Int(id),options.indices.contains(i) {draft[keyPath:path]=options[i].0}}}
    }
    private func select(_ rule:ScheduleRule) {selected=rule.id;draft=rule;base=rule;editing=true;issue=nil;Task {await loadVersions(project:rule.projectID)}}
    private func loadVersions(project:String)async {loading=true;defer{loading=false};let token=UUID();generation=token;do {let choices=try await controller.versions(projectID:project);guard generation==token,draft.projectID==project else{return};versions=choices;chooseVersion(draft.versionID)}catch{issue=error.localizedDescription}}
    private func chooseVersion(_ id:String) {
        guard let version=versions.first(where:{$0.id==id}) else{return}
        if draft.versionID != id {draft.allowEffects=false}
        draft.versionID=id;draft.versionNumber=version.number;draft.versionHash=version.hash;details=nil
        let project=draft.projectID,token=UUID();generation=token
        Task {do {let loaded=try await controller.details(projectID:project,versionID:id);guard generation==token,draft.versionID==id,draft.projectID==project else{return};details=loaded;draft.hasEffects = !loaded.effects.isEmpty;if !loaded.targets.contains(where:{$0.address==draft.target}) {draft.target=loaded.targets.count==1 ? loaded.targets[0].address:nil}}catch{issue=error.localizedDescription}}
    }
    private func newSchedule()async {
        if controller.projects.isEmpty {await controller.loadProjects()}
        guard let project=controller.projects.first(where:{projectID==nil || $0.id==projectID}) else {issue="Open and explicitly Save a managed workflow project first. External project files remain available for manual runs.";return}
        draft=ScheduleRule(projectID:project.id,projectName:project.name,versionID:"",versionNumber:1,versionHash:"",name:"New schedule",timing:.init());base=nil;selected=nil;editing=true
        await changeProject(project.id)
    }
    private func changeProject(_ id:String)async {
        guard let project=controller.projects.first(where:{$0.id==id}) else{return};draft.projectID=id;draft.projectName=project.name;draft.allowEffects=false
        loading=true;defer{loading=false}
        do {versions=try await controller.versions(projectID:id);guard let version=versions.first else{issue="Explicitly Save this project's canvas before scheduling it.";return};chooseVersion(version.id)}catch{issue=error.localizedDescription}
    }
}
