import SwiftUI
import AppKit
import KwipMLCore

/// The project data grants are independent of action grants and of each other.
enum MCPProjectDataPermission:String,CaseIterable,Identifiable {
    case capturedValues,history
    var id:String {rawValue}
    var title:String {self == .history ? "Read retained History":"Read captured values"}
    var detail:String {self == .history ? "List retained run receipts. Inspecting captured evidence also requires Read captured values.":"Read captured output schemas and values for this project."}
    func isAllowed(project:String,in settings:MCPSettings)->Bool {
        guard let grant=settings.projects[project] else{return false}
        return self == .history ? grant.readHistory:grant.readCapturedValues
    }
    func setAllowed(_ allowed:Bool,project:String,in settings:inout MCPSettings) {
        guard settings.projects[project] != nil else{return}
        if self == .history {settings.projects[project]?.readHistory=allowed}
        else {settings.projects[project]?.readCapturedValues=allowed}
    }
}

struct MCPSettingsView:View {
    @ObservedObject var controller:MCPController
    @State private var draft=MCPSettings.factory
    @State private var baseline=MCPSettings.factory
    @State private var port=""
    @State private var fixed=false
    @State private var clientName=""
    @State private var projectSearch=""
    @State private var setupClient:String?
    @State private var pendingSetup:String?
    @State private var setup=""
    @State private var failure:String?
    @Environment(\.appearanceTheme) private var theme
    private var dirty:Bool {draft != baseline || fixed && Int(port) != draft.fixedPort}
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack {
                Label(controller.status,systemImage:controller.endpoint == nil ? "circle":"circle.fill").foregroundStyle(controller.endpoint == nil ? MLTheme.muted:theme.accentColor)
                Spacer()
                Button("Stop MCP"){controller.stop();reset()}.disabled(!controller.settings.enabled)
                    .accessibilityIdentifier("settings.mcp.stop")
            }
            Toggle("Enable local MCP",isOn:$draft.enabled).toggleStyle(FileInputCheckboxStyle(tint:theme.accentColor))
                .accessibilityIdentifier("settings.mcp.enabled")
            if let endpoint=controller.endpoint {Text(endpoint).font(.system(.callout,design:.monospaced)).textSelection(.enabled)}
            DisclosureGroup("Listener") {
                VStack(alignment:.leading,spacing:12) {
                    Text("127.0.0.1 · this Mac only").font(.callout)
                    Toggle("Use a fixed port",isOn:$fixed).toggleStyle(FileInputCheckboxStyle(tint:theme.accentColor))
                        .onChange(of:fixed){_,value in draft.fixedPort=value ? Int(port):nil}
                    if fixed {TextField("Port · 1024–65535",text:$port).flatField().frame(maxWidth:280)
                        .onChange(of:port){_,value in draft.fixedPort=Int(value)}.accessibilityIdentifier("settings.mcp.port")}
                }.frame(maxWidth:.infinity,alignment:.leading).padding(.top,10)
            }
            HStack {
                Button("Apply"){apply()}.disabled(!dirty).accessibilityIdentifier("settings.mcp.apply")
                Button("Revert"){reset()}.disabled(!dirty)
            }
            if let error=failure ?? controller.error {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
            Divider()
            SectionLabel(title:"Clients")
            Text("Application settings actions affect all windows and have separate client grants. Project settings grants do not grant application access or captured-value reads.").font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                TextField("Client name",text:$clientName).flatField()
                Button("Add client") {
                    do {let id=try controller.addClient(name:clientName);clientName="";reset();if controller.settings.enabled{pendingSetup=id;if controller.endpoint != nil{showSetup(id);pendingSetup=nil}}}catch{failure=error.localizedDescription}
                }.disabled(dirty || clientName.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
            }
            ForEach(draft.clients) {client in
                VStack(alignment:.leading,spacing:10) {
                    HStack {
                        Text(client.name).font(.system(size:14,weight:.semibold))
                        Spacer()
                        Button("Connect a client"){showSetup(client.id)}.disabled(dirty || controller.endpoint==nil)
                        Button("Revoke") {do{try controller.revokeClient(client.id);reset()}catch{failure=error.localizedDescription}}.disabled(dirty)
                    }
                    DisclosureGroup("Allowed actions") {
                        actionGrid {action in Binding(get:{draft.clients.first(where:{$0.id==client.id})?.actions.contains(action) == true},set:{allowed in guard let index=draft.clients.firstIndex(where:{$0.id==client.id})else{return};if allowed{draft.clients[index].actions.insert(action)}else{draft.clients[index].actions.remove(action)}})}
                    }
                }.padding(14).frame(maxWidth:.infinity,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))
            }
            Divider()
            Button("Cancel MCP runs"){controller.gateway.cancelOwnedRuns()}
            projectSection
            DisclosureGroup("Blocked actions") {
                actionGrid {action in Binding(get:{draft.disabledActions.contains(action)},set:{blocked in if blocked{draft.disabledActions.insert(action)}else{draft.disabledActions.remove(action)}})}
            }
            DisclosureGroup("Blocked canvas components") {
                LazyVGrid(columns:permissionColumns,alignment:.leading,spacing:10) {
                    ForEach(controller.componentChoices,id:\.id) {choice in
                        Toggle(choice.title,isOn:Binding(get:{draft.disabledComponents.contains(choice.id)},set:{blocked in if blocked{draft.disabledComponents.insert(choice.id)}else{draft.disabledComponents.remove(choice.id)}}))
                            .toggleStyle(FileInputCheckboxStyle(tint:theme.accentColor))
                    }
                }.frame(maxWidth:.infinity,alignment:.leading).padding(.top,10)
            }
            DisclosureGroup("Activity") {
                VStack(alignment:.leading,spacing:8) {
                    HStack {
                        Text("Retain receipts")
                        Stepper("\(draft.auditRetention)",value:$draft.auditRetention,in:0...1000,step:50)
                        Spacer()
                        Button("Clear receipts"){controller.clearAudit()}
                    }
                    ForEach(controller.audit.suffix(20).reversed()) {entry in
                        HStack {Text(entry.at,style:.time).frame(width:80,alignment:.leading);Text(entry.action.replacingOccurrences(of:"_",with:" "));Spacer();Text(entry.outcome).foregroundStyle(MLTheme.muted)}.font(.caption)
                    }
                }.padding(.top,10)
            }
            HStack {
                Button("Apply changes"){apply()}.disabled(!dirty)
                Button("Revert changes"){reset()}.disabled(!dirty)
            }
            DisclosureGroup("Information") {
                VStack(alignment:.leading,spacing:10) {
                    Text("Connect a client shows instructions. Add the URL and bearer header in your AI client’s MCP settings; \(ProductBrand.name) does not configure that client for you.")
                    Text("Automatic binding chooses an unused localhost port. It can change after restarting; a fixed port keeps the URL stable and reports a conflict instead of changing ports.")
                    Text("Access requires both the client’s action and the project’s action. Projects are unavailable until allowed. New clients start with read-only actions; editing and running must be enabled explicitly. Creating a project grants access to that new project, never to existing projects or files.")
                    Text("Captured values can include private text. Your AI client may send returned values to its model provider. Source access, execution budgets and capture settings still apply; MCP does not grant filesystem access or run arbitrary code.")
                    Text("Stop or revoke prevents new requests. Runs already admitted continue under their normal limits; cancel them in the canvas or through their owning MCP client. Receipts omit arguments, source paths, tokens and captured values.")
                }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top,10)
            }
        }.frame(maxWidth:760,alignment:.leading).onAppear{reset()}
        .onChange(of:controller.settings){old,_ in if draft==old && (!fixed || Int(port)==old.fixedPort){reset()}}
        .onChange(of:controller.endpoint){_,endpoint in if endpoint != nil,let id=pendingSetup{showSetup(id);pendingSetup=nil}}
        .sheet(item:Binding(get:{setupClient.map{SetupIdentity(id:$0)}},set:{setupClient=$0?.id}),onDismiss:{setup=""}) {identity in
            MCPClientSetupView(clientName:controller.settings.clients.first(where:{$0.id==identity.id})?.name ?? "\(ProductBrand.name)",
                configuration:setup,onDone:{setupClient=nil;setup=""})
        }
    }
    private var permissionColumns:[GridItem] {
        [GridItem(.flexible(minimum:0),alignment:.leading),GridItem(.flexible(minimum:0),alignment:.leading)]
    }
    private var projectSection:some View {
        let projects=controller.availableProjects
        let search=projectSearch.trimmingCharacters(in:.whitespacesAndNewlines)
        let visible=projects.filter{search.isEmpty || $0.name.localizedStandardContains(search)}
        return VStack(alignment:.leading,spacing:12) {
            HStack {
                SectionLabel(title:"Projects")
                Spacer()
                Text("\(projects.filter{draft.projects[$0.id] != nil}.count) of \(projects.count) allowed")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            if !projects.isEmpty {
                TextField("Search projects",text:$projectSearch).flatField()
                    .accessibilityIdentifier("settings.mcp.projects.search")
                VStack(alignment:.leading,spacing:0) {
                    ForEach(visible) {record in
                        VStack(alignment:.leading,spacing:10) {
                            HStack {
                                Toggle(isOn:Binding(get:{draft.projects[record.id] != nil},set:{allowed in draft.projects[record.id]=allowed ? .init(actions:Set(MCPAction.allCases.filter{$0.requiresProject})):nil})) {
                                    VStack(alignment:.leading,spacing:4) {
                                        Text(record.name).font(.system(size:13,weight:.semibold)).lineLimit(2)
                                        Text(record.updatedAt.formatted(date:.abbreviated,time:.shortened))
                                            .font(.caption).foregroundStyle(MLTheme.muted)
                                    }
                                }.toggleStyle(FileInputCheckboxStyle(tint:theme.accentColor))
                                Spacer(minLength:8)
                                Text(draft.projects[record.id] == nil ? "Private":"Allowed")
                                    .font(.caption).foregroundStyle(draft.projects[record.id] == nil ? MLTheme.muted:theme.accentColor)
                            }
                            if draft.projects[record.id] != nil {
                                DisclosureGroup("Permissions") {
                                    VStack(alignment:.leading,spacing:12) {
                                        ForEach(MCPProjectDataPermission.allCases) {permission in
                                            Toggle(permission.title,isOn:Binding(get:{permission.isAllowed(project:record.id,in:draft)},set:{permission.setAllowed($0,project:record.id,in:&draft)}))
                                                .toggleStyle(FileInputCheckboxStyle(tint:theme.accentColor))
                                                .accessibilityIdentifier("settings.mcp.project.\(record.id).\(permission.rawValue)")
                                                .help(permission.detail)
                                        }
                                        Text("Settings actions control future-run choices; they do not grant captured-value access.")
                                            .font(.caption).foregroundStyle(MLTheme.muted)
                                        actionGrid(projectOnly:true) {action in Binding(get:{draft.projects[record.id]?.actions.contains(action)==true},set:{allowed in if allowed{draft.projects[record.id]?.actions.insert(action)}else{draft.projects[record.id]?.actions.remove(action)}})}
                                    }.padding(.top,8)
                                }.padding(.leading,20)
                            }
                        }.padding(12).frame(maxWidth:.infinity,alignment:.leading)
                            .background(draft.projects[record.id] == nil ? Color.clear:theme.accentColor.opacity(0.035))
                        if record.id != visible.last?.id {Divider()}
                    }
                    if visible.isEmpty {Text("No matching projects.").font(.caption).foregroundStyle(MLTheme.muted).padding(12)}
                }.overlay(Rectangle().stroke(MLTheme.rule))
            }else{Text("No managed projects yet.").font(.caption).foregroundStyle(MLTheme.muted)}
        }
    }
    private struct SetupIdentity:Identifiable {let id:String}
    private func actionGrid(projectOnly:Bool=false,binding:@escaping(MCPAction)->Binding<Bool>)->some View {
        LazyVGrid(columns:permissionColumns,alignment:.leading,spacing:10) {
            ForEach(MCPAction.allCases.filter{!projectOnly || $0.requiresProject},id:\.self){action in Toggle(action.title,isOn:binding(action)).toggleStyle(FileInputCheckboxStyle(tint:theme.accentColor))}
        }.padding(.top,10)
    }
    private func reset(){draft=controller.settings;baseline=controller.settings;fixed=draft.fixedPort != nil;port=draft.fixedPort.map(String.init) ?? "8781";failure=nil}
    private func apply(){do{guard baseline==controller.settings else{failure="MCP access changed while this draft was open. Revert to reload the current policy.";return};if fixed{guard let number=Int(port),(1024...65535).contains(number)else{throw MCPSettings.Refusal(code:"invalid_settings")};draft.fixedPort=number}else{draft.fixedPort=nil};try controller.save(draft);reset()}catch{failure=error.localizedDescription}}
    private func showSetup(_ id:String){do{setup=try controller.connectionConfiguration(clientID:id);setupClient=id;failure=nil}catch{failure=error.localizedDescription}}
}
