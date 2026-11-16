import SwiftUI
import KwipMLCore

struct GameSessionInspectorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @State private var editor:ExpandedConfigurationWindowController?
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Frozen-policy episodes").font(.headline)
            Text("Bind the authored game to model policies, then run a finite batch. Weights remain unchanged.").font(.caption).foregroundStyle(MLTheme.muted)
            if let config=try? GameSessionConfiguration(context.configuration) {
                Text("\(config.values["episodes"]?.integerValue ?? 8) episodes · \(config.values["seats"]?.stringValue == "alternate" ? "Alternating seats":"Fixed seats")").font(.caption)
                ForEach(["A","B"],id:\.self) {role in Text("Policy \(role) · \(sourceTitle(config.source(role)))").font(.caption).foregroundStyle(MLTheme.muted)}
            }
            Button("Configure game session…"){
                if let editor,!editor.didClose {editor.present();return}
                let next=ExpandedConfigurationWindowController(title:"Game session",identifier:"info.kwip.KwipML.game-session-editor"){controller in
                    AnyView(GameSessionEditorView(context:context,host:host,requestClose:{[weak controller] in controller?.close()},registerDiscard:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty}).windowActivatingControls().appAppearance(host.configurationAppearance))
                };editor=next;next.present()
            }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
            if let session=host.gameEpisodes,session.address==context.authority.address {GameSessionStatusView(session:session)}
            Text("Run Flow executes the connected session. Validation checks only rules and model bindings; it does not run episodes or surrounding writes.").font(.caption).foregroundStyle(MLTheme.muted)
        }.onDisappear{editor?.close();editor=nil}
    }
    private func sourceTitle(_ value:String)->String {switch value {case "canvas":return "Canvas definition";case "saved":return "Saved weights";case "sameA":return "Same snapshot as A";default:return "Random legal"}}
}
struct GameSessionStatusView:View {
    @ObservedObject var session:GameEpisodeSession
    var body:some View {Text(session.message).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
}

struct GameSessionEditorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    let requestClose:()->Void
    let registerDiscard:(@escaping ()->Void,@escaping ()->Bool)->Void
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var section="Game"
    @State private var numericPending=Set<String>()
    @State private var editIdentity=UUID()
    @State private var message=""
    @State private var closing=false
    @State private var validation:GameEpisodeSession?
    @State private var tested:[String:JSONValue]?
    let sections=["Game","Players","Run limits","Capture"]
    var body:some View {
        VStack(spacing:0) {
            HStack {VStack(alignment:.leading,spacing:4){Text("Game session").font(.title2.weight(.semibold));Text("Generic rule/model binding · frozen-policy episodes").font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close"){close()}.buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider()
            HStack(spacing:0) {
                VStack(alignment:.leading,spacing:5) {ForEach(sections,id:\.self){name in Button {section=name} label:{Text(name).frame(maxWidth:.infinity,alignment:.leading).padding(9).background(section==name ? MLTheme.teal.opacity(0.13):Color.clear).contentShape(Rectangle())}.buttonStyle(.plain)};Spacer()}.padding(12).frame(width:210)
                Divider()
                ScrollView {VStack(alignment:.leading,spacing:18){Text(section).font(.title3.weight(.semibold));content}.padding(24).frame(maxWidth:.infinity,alignment:.leading)}
            }
            Divider()
            VStack(alignment:.leading,spacing:10) {
                if let validation {GameSessionStatusView(session:validation)}
                if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.muted)}
                HStack {Button("Revert"){guard context.isCurrent else{return};draft=(try? GameSessionConfiguration(original).values) ?? original;numericPending=[];editIdentity=UUID();tested=nil;validation?.cancel();validation=nil;message=""}.buttonStyle(FlatButtonStyle());Spacer();Button("Validate bindings") {tested=draft;validation=host.startGameSession(context.authority.address,override:draft,validateOnly:true)}.buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || !numericPending.isEmpty || validation?.isBusy==true || host.hasActiveOperation);Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || !numericPending.isEmpty || validation?.isBusy==true || tested != draft || validation?.message != "Bindings validated · no episodes run.")}
            }.padding(16)
        }.frame(minWidth:790,idealWidth:900,minHeight:600,idealHeight:700).background(MLTheme.background)
        .onAppear {original=context.configuration;draft=(try? GameSessionConfiguration(original).values) ?? GameSessionConfiguration.defaults;registerDiscard({discard()}, {!closing && (!numericPending.isEmpty || draft != ((try? GameSessionConfiguration(original).values) ?? original))})}
        .onChange(of:draft){_,_ in reportPending()}
        .onDisappear {discard()}
    }
    @ViewBuilder private var content:some View {
        switch section {
        case "Game":
            group("Authored definition") {Text("Game definition → Definition · authored tests must pass").font(.callout);Text("Current player, legal moves and outcomes follow the authored rules.").font(.caption).foregroundStyle(MLTheme.muted)}
            group("Episode batch") {number("Episodes",key:"episodes");choice("Seats",key:"seats",items:[("alternate","Alternate seats"),("fixed","Fixed seats")]);choice("Policy A starts as",key:"aSeat",items:[("0","Player 0"),("1","Player 1")],integer:true);number("Session seed",key:"seed")}
        case "Players":ForEach(["A","B"],id:\.self){role in group("Policy "+role){policy(role)}}
        case "Run limits":
            group("Bounded execution") {number("Moves per episode",key:"maxMoves");number("Total moves",key:"maximumTotalMoves");number("Seconds per episode",key:"episodeSeconds",integer:false);number("Seconds per session",key:"sessionSeconds",integer:false);choice("Device",key:"device",items:[("cpu","CPU reference"),("mps","Apple GPU (MPS)")]);Text("Move/time limits are truncations, never draws or terminal rewards. The workspace’s resource limits and cancellation also apply.").font(.caption).foregroundStyle(MLTheme.muted)}
        default:
            group("Decision detail") {Toggle("Capture observations, masks, logits and probabilities",isOn:Binding(get:{draft["captureDecisions"] == .bool(true)},set:{draft["captureDecisions"] = .bool($0)})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.teal));Text("Complete outputs: at most 8 MB, including optional decision detail.").font(.caption).foregroundStyle(MLTheme.muted)}
            group("Recorded execution") {Text("History retains the exact applied rules, model and settings.").font(.callout);Text("Learning is configured in Train → Self-play.").font(.caption).foregroundStyle(MLTheme.muted)}
        }
    }
    private func group<Content:View>(_ title:String,@ViewBuilder content:()->Content)->some View {VStack(alignment:.leading,spacing:12){Text(title).font(.headline);content()}.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))}
    private func close(){requestClose()}
    private func discard(){guard !closing else{return};closing=true;validation?.cancel();context.setNumericPending(origin:context.numericOrigin(key:"gameSession"),key:"gameSession",pending:false)}
    private func reportPending(){context.setNumericPending(origin:context.numericOrigin(key:"gameSession"),key:"gameSession",pending:!closing && (!numericPending.isEmpty || draft != ((try? GameSessionConfiguration(original).values) ?? original)))}
    private func number(_ title:String,key:String,integer:Bool=true)->some View {
        let maximum:Double=key=="episodes" ? 256:key=="maxMoves" ? 4096:key=="maximumTotalMoves" ? 32768:key=="seed" ? 2147483647:3600
        let minimum:Double=key=="seed" ? 0:integer ? 1:0.001
        let origin=editIdentity.uuidString+":"+key
        return VStack(alignment:.leading,spacing:5) {
            Text(title).font(.caption).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:integer ? .wholeDisplay(draft[key]?.doubleValue ?? 0):.decimal(draft[key]?.doubleValue ?? 0),domain:integer ? .integer(minimum:minimum,maximum:maximum):.decimal(minimum:minimum,maximum:maximum),origin:origin,currentOrigin:{editIdentity.uuidString+":"+key},onDraftChange:{_,pending in if pending{numericPending.insert(key)}else{numericPending.remove(key)};reportPending()}) {value in
                switch value {case .integer(let v):draft[key] = .integer(v);case .decimal(let v):draft[key] = .number(v)};return true
            }
        }
    }
    private func choice(_ title:String,key:String,items:[(String,String)],integer:Bool=false)->some View {
        VStack(alignment:.leading,spacing:5) {
            Text(title).font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:title,selection:integer ? String(draft[key]?.integerValue ?? 0):draft[key]?.stringValue ?? "",options:items.map{.init(id:$0.0,title:$0.1)}) {
                draft[key]=integer ? .integer(Int64($0) ?? 0):.string($0)
            }
        }
    }
    private func policyChoice(_ title:String,role:String,key:String,items:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:5) {
            Text(title).font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:title,selection:policyDraft(role)[key]?.stringValue ?? "",options:items.map{.init(id:$0.0,title:$0.1)}) {setPolicy(role,key,.string($0))}
        }
    }
    private func policy(_ role:String)->some View {
        let p=policyDraft(role)
        return VStack(alignment:.leading,spacing:12) {
            policyChoice("Source",role:role,key:"source",items:[("canvas","Canvas definition"),("saved","Saved weights"),("random","Random legal")] + (role=="B" ? [("sameA","Same snapshot as A")]:[]))
            if p["source"] == .string("canvas") {Text("Initialization seed").font(.caption).foregroundStyle(MLTheme.muted);TextField("Initialization seed",text:Binding(get:{p["initializationSeed"]?.stringValue ?? String(p["initializationSeed"]?.integerValue ?? 17)},set:{setPolicy(role,"initializationSeed",Int64($0).map(JSONValue.integer) ?? .string($0))})).flatField()}
            if p["source"] != .string("random") {
                Text("Model input name").font(.caption).foregroundStyle(MLTheme.muted)
                TextField("Named input",text:Binding(get:{p["inputName"]?.stringValue ?? "input"},set:{setPolicy(role,"inputName",.string($0))})).flatField()
                policyChoice("Selection",role:role,key:"selection",items:[("greedy","Greedy"),("sampling","Seeded sampling")])
                if p["selection"] == .string("sampling") {Text("Temperature").font(.caption).foregroundStyle(MLTheme.muted);TextField("Temperature",text:Binding(get:{p["temperature"]?.stringValue ?? String(p["temperature"]?.doubleValue ?? 1)},set:{setPolicy(role,"temperature",Double($0).flatMap{$0.isFinite ? JSONValue.number($0):nil} ?? .string($0))})).flatField()}
            }
            Text("Disconnect incompatible model wires before changing source. Same snapshot shares immutable weights, not action-selection randomness.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func policyDraft(_ role:String)->[String:JSONValue] {
        var p=(try? GameSessionConfiguration([:]).policy(role)) ?? [:]
        if case .object(let all)=draft["policies"],case .object(let v)=all[role] {p.merge(v){_,new in new}};return p
    }
    private func setPolicy(_ role:String,_ key:String,_ value:JSONValue) {var all: [String:JSONValue]=[:];if case .object(let v)=draft["policies"] {all=v};var p:[String:JSONValue]=[:];if case .object(let v)=all[role]{p=v};p[key]=value;all[role] = .object(p);draft["policies"] = .object(all)}
    private func apply(){do {
        _=try GameSessionConfiguration(draft)
        // Clear this editor's pending marker while its revision lease is still current.
        // A successful configure advances the lease before onDisappear can run.
        closing=true
        context.setNumericPending(origin:context.numericOrigin(key:"gameSession"),key:"gameSession",pending:false)
        guard context.authority.configure(draft,expected:original) else {closing=false;message=context.statusMessage;reportPending();return}
        requestClose()
    }catch{message=error.localizedDescription}}
}

struct GameSessionResultsView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    let address:WorkflowNodeAddress
    @State private var port="summary"
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            GameSessionOutputChoices(port:$port)
            if let session=host.gameEpisodes,session.address==address {GameSessionStatusView(session:session)}
            if let query=host.gameTableViewSession(.init(address:address,port:port)) {TableViewOutputView(session:query,executionBusy:host.owner.isBusy).id((host.gameEpisodes?.id.uuidString ?? "")+port)}
            else {Text("Run the session to publish complete captured tables. No game board renderer or training is implied.").foregroundStyle(MLTheme.muted)}
        }
    }
}

/// Shared themed selector for live and recorded table outputs.
struct GameSessionOutputChoices:View {
    @Binding var port:String
    var body:some View {
        FlatChoiceControl(title:"Captured output",selection:port,options:[.init(id:"summary",title:"Summary"),.init(id:"episodes",title:"Episodes"),.init(id:"moves",title:"Moves")],compact:true) {port=$0}
    }
}
