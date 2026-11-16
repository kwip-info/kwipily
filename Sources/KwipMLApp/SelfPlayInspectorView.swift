import SwiftUI
import KwipMLCore

struct SelfPlayInspectorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @State private var editor:SelfPlayEditorWindowController?
    @State private var campaign:[String:JSONValue]?
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Game policy/value · captured compute profile").font(.headline)
            Text("Frozen collection → search → joint learning → committed checkpoint → independent evaluation").font(.caption).foregroundStyle(MLTheme.muted)
            Button("Configure Self-play…"){
                if let editor,!editor.didClose {editor.present()}
                else {
                    let appearance=host.configurationAppearance
                    let controller=SelfPlayEditorWindowController {controller in
                        AnyView(SelfPlayEditorView(context:context,host:host,
                            requestClose:{[weak controller] in controller?.close()},
                            registerDiscard:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty})
                            .windowActivatingControls().appAppearance(appearance))
                    }
                    editor=controller;controller.present()
                }
            }.buttonStyle(FlatButtonStyle())
            if let session=host.selfPlay,session.address==context.authority.address {SelfPlayStatusView(session:session)}
            if let record=host.selfPlayRecords.first(where:{$0.address==context.authority.address}) {
                Text("Latest boundary · "+String(record.id.prefix(8))).font(.caption)
                if let campaign {
                    Text("Campaign \(String((campaign["id"]?.stringValue ?? "").prefix(8))) · phase \(campaign["phase"]?.integerValue ?? 1)\nCommitted \(campaign["iteration"]?.integerValue ?? 0) / \(campaign["target"]?.integerValue ?? 0) · \(campaign["segmentCount"]?.integerValue ?? 0) retained result segments").font(.caption).foregroundStyle(MLTheme.muted)
                    Button("Resume campaign"){_=host.startSelfPlay(context.authority.address,resume:record)}.buttonStyle(FlatButtonStyle()).disabled(host.hasActiveOperation || context.isBusy || campaign["iteration"]==campaign["target"])
                    Button("Extend with applied budget"){_=host.startSelfPlay(context.authority.address,resume:record,campaignAction:"extend")}.buttonStyle(FlatButtonStyle()).disabled(host.hasActiveOperation || context.isBusy)
                    Button("Fork weights · fresh optimizer & replay"){_=host.startSelfPlay(context.authority.address,resume:record,campaignAction:"fork")}.buttonStyle(FlatButtonStyle()).disabled(host.hasActiveOperation || context.isBusy)
                }else{Button("Resume committed boundary"){_=host.startSelfPlay(context.authority.address,resume:record)}.buttonStyle(FlatButtonStyle()).disabled(host.hasActiveOperation || context.isBusy || record.completed)}
            }
            if let session=host.selfPlay,session.isBusy {
                if session.campaignEnabled{Button(session.pauseRequested ? "Pause requested":"Pause at boundary"){session.pauseAtBoundary()}.buttonStyle(FlatButtonStyle()).disabled(session.pauseRequested)}
                Button("Stop now"){host.cancel()}.buttonStyle(FlatButtonStyle())
            }
            Text("Apply saves settings. Run Flow starts training. Resume requires exact rule/model/settings/runtime pins; original records remain immutable.").font(.caption).foregroundStyle(MLTheme.muted)
        }.onDisappear{editor?.close();editor=nil}.onAppear{host.reloadSelfPlayRecords()}.task(id:host.selfPlayRecords.first(where:{$0.address==context.authority.address})?.id){guard let record=host.selfPlayRecords.first(where:{$0.address==context.authority.address}),let root=host.selfPlayWorkspaceRoot else{return};campaign=try? await WorkflowHistoryVerificationWork.run{try SelfPlayCampaignGrant.index(record,root:root)}}
    }
}
struct SelfPlayStatusView:View {
    @ObservedObject var session:SelfPlaySession
    var body:some View {Text(session.message).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
}
struct SelfPlayEditorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    let requestClose:()->Void
    let registerDiscard:(@escaping ()->Void,@escaping ()->Bool)->Void
    @State private var original:[String:JSONValue]=[:]
    @State private var draft=SelfPlayConfiguration.defaults
    @State private var loadedDraft=SelfPlayConfiguration.defaults
    @State private var section="Game & model"
    @State private var message=""
    @State private var closing=false
    @State private var tested:[String:JSONValue]?
    @State private var validation:SelfPlaySession?
    @State private var pending=Set<String>()
    @State private var identity=UUID()
    @State private var openingsText="[[]]"
    @State private var loadedOpeningsText="[[]]"
    let sections=["Game & model","Campaign","Self-play","Search","Learning","Replay & checkpoints","Evaluation","Compute","Resources & History"]
    var body:some View {
        VStack(spacing:0) {
            HStack {VStack(alignment:.leading,spacing:4){Text("Train · Self-play").font(.title2.weight(.semibold));Text("Explicit paired heads · bounded generic rule interpreter").font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close"){close()}.buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider()
            HStack(spacing:0) {
                VStack(alignment:.leading,spacing:5){ForEach(sections,id:\.self){name in Button {section=name}label:{Text(name).frame(maxWidth:.infinity,alignment:.leading).padding(9).background(section==name ? MLTheme.teal.opacity(0.13):Color.clear).contentShape(Rectangle())}.buttonStyle(.plain)};Spacer()}.padding(12).frame(width:220)
                Divider()
                ScrollView {VStack(alignment:.leading,spacing:18){Text(section).font(.title3.weight(.semibold));content}.padding(24).frame(maxWidth:.infinity,alignment:.leading)}
            }
            Divider()
            VStack(alignment:.leading,spacing:10){if let validation{SelfPlayStatusView(session:validation)};if !message.isEmpty{Text(message).font(.caption).foregroundStyle(MLTheme.rose)}
                HStack {Button("Revert"){load();tested=nil;validation?.cancel();validation=nil;identity=UUID()}.buttonStyle(FlatButtonStyle());Spacer();Button("Validate bindings"){tested=draft;validation=host.startSelfPlay(context.authority.address,override:draft,validateOnly:true)}.buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || !pending.isEmpty || host.hasActiveOperation);Button("Apply"){apply()}.buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || !pending.isEmpty || tested != draft || validation?.message != "Bindings validated · no training run.")}
            }.padding(16)
        }.frame(minWidth:800,idealWidth:940,minHeight:610,idealHeight:730).background(MLTheme.background)
        .onAppear{load();registerDiscard({discard()}, {!closing && (!pending.isEmpty || draft != loadedDraft)})}.onChange(of:draft){_,_ in report()}.onDisappear{discard()}
    }
    @ViewBuilder private var content:some View {
        switch section {
        case "Game & model":
            group("Canvas bindings"){Text("Game definition → Definition · paired Model Output → Model");Text("Policy: [batch, actions] logits · Value: [batch, 1] in −1…1").font(.caption).foregroundStyle(MLTheme.muted)}
            group("Qualification"){Text("Two-player, deterministic, fully observable zero-sum rules").font(.callout);number("Initialization seed","initializationSeed",0,2147483647)}
        case "Self-play":group("Collection budget"){number("Total iterations","iterations",1,SelfPlayCampaignGrant.enabled(draft) ? 1000000:99);number("Games per iteration","games",2,256);number("Moves per game","maxMoves",1,4096);number("Run seed","seed",0,2147483647);number("Sampling temperature","temperature",0.01,10,integer:false);number("Temperature through move","temperatureMoves",0,4096);Text("Collect with frozen weights, then update. Truncated games do not enter replay.").font(.caption).foregroundStyle(MLTheme.muted)}
        case "Campaign":group("Finite campaign"){
            FlatChoiceControl(title:"Training scope",selection:SelfPlayCampaignGrant.enabled(draft) ? "campaign":"single",options:[.init(id:"single",title:"Single bounded run"),.init(id:"campaign",title:"Campaign · continue across bounded jobs")]) {value in
                if value=="campaign" {draft["campaign"]=SelfPlayConfiguration.advancedDefaults["campaign"];setNested("campaign","enabled",.bool(true))}else{draft["campaign"]=nil}
            }
            if SelfPlayCampaignGrant.enabled(draft) {
                number("Iterations per job slice","sliceIterations",1,99,scope:"campaign")
                number("Campaign active seconds","maximumSeconds",1,604800,scope:"campaign")
                number("Total owned bytes","maximumBytes",67108864,8589934592,scope:"campaign")
                Text("App open and Mac awake · Run Flow starts a new campaign").font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
        case "Search":group("Bounded PUCT"){
            FlatChoiceControl(title:"Reuse successor tree",selection:nested("search")["reuseTree"] == .bool(true) ? "on":"off",options:[.init(id:"off",title:"Tree reuse · off (reference)"),.init(id:"on",title:"Tree reuse · verified successor subtree")]) {setNested("search","reuseTree",.bool($0=="on"))}
number("Simulations per move","simulations",1,4096,search:true);number("Exploration coefficient","exploration",0,100,integer:false,search:true);number("Maximum depth","maximumDepth",1,512,search:true);number("Maximum nodes","maximumNodes",2,10000,search:true);number("Seconds per search","seconds",0.001,60,integer:false,search:true);number("Root noise alpha","noiseAlpha",0.001,10,integer:false,search:true);number("Root noise mix","noiseMix",0,1,integer:false,search:true);Text("Legal actions only · evaluation disables root noise").font(.caption).foregroundStyle(MLTheme.muted)}
        case "Learning":group("Joint updates"){choice("Optimizer","optimizer",[("adam","Adam"),("adamw","AdamW"),("sgd","SGD")]);number("Learning rate","learningRate",1e-7,1,integer:false);choice("Schedule","scheduler",[("constant","Constant"),("linear","Linear"),("cosine","Cosine")]);if draft["scheduler"] != .string("constant"){number("Final learning rate","finalLearningRate",1e-7,1,integer:false)};if draft["optimizer"] == .string("sgd"){number("Momentum","momentum",0,0.999,integer:false)}else{number("Beta 1","beta1",0,0.9999,integer:false);number("Beta 2","beta2",0,0.999999,integer:false);number("Epsilon","epsilon",1e-12,1,integer:false)};number("Updates per iteration","updates",1,4096);number("Minibatch positions","batch",1,1024);number("Policy loss weight","policyWeight",0,100,integer:false);number("Value loss weight","valueWeight",0,100,integer:false);number("Weight decay","weightDecay",0,1,integer:false);number("Clip gradient norm","gradientClip",0.001,1000,integer:false);Text("Masked policy cross-entropy + value mean squared error. AdamW uses decoupled decay, not an L2-loss claim. At most 4096 completed-update measurements per run.").font(.caption).foregroundStyle(MLTheme.muted)}
        case "Replay & checkpoints":group("Required learning data"){SelfPlayRetentionView(host:host,address:context.authority.address);number("Replay positions","replayPositions",1,100000);number("Replay bytes","replayBytes",65536,SelfPlayCampaignGrant.enabled(draft) ? 2147483648:536870912);choice("When full","replayPolicy",[("stop","Stop before update"),("rolling","Rolling active window")]);number("Recent resume boundaries","checkpoints",1,10);Text("Rolling needs room for the next commit. The previous recovery boundary stays until commit succeeds.").font(.caption).foregroundStyle(MLTheme.muted)}
        case "Evaluation":group("Independent evaluation suite"){
            number("Evaluate every iterations","evaluationEvery",1,99);number("Balanced seat pairs per mode","evaluationPairs",1,128)
            if draft["evaluation"]==nil {Button("Configure opponent & opening suite"){draft["evaluation"]=SelfPlayConfiguration.advancedDefaults["evaluation"]}.buttonStyle(FlatButtonStyle())}
            else {
                number("Independent suite seed","seed",0,2147483647,scope:"evaluation")
                ForEach([("initial","Initial frozen model"),("random","Random legal actions"),("uniform-search","Uniform-prior bounded search")],id:\.0){id,title in opponentChoice(id,title)}
                ForEach(host.selfPlayRecords.filter(\.completed).prefix(5)){record in opponentChoice("checkpoint:"+record.id,"Checkpoint "+String(record.id.prefix(8)))}
                Text("Held-out opening action sequences · JSON").font(.caption)
                GameRuleCodeEditor(text:$openingsText,language:.json,label:"Held-out opening sequences",limit:65536).frame(minHeight:130).overlay(Rectangle().stroke(MLTheme.rule)).onChange(of:openingsText){_,_ in if openingsText != loadedOpeningsText{pending.insert("opening-buffer")}else{pending.remove("opening-buffer")};report()}
                Button("Use opening suite") {do{let value=try JSONDecoder().decode(JSONValue.self,from:Data(openingsText.utf8));guard case .array=value else{throw WorkflowArchiveError("Use an array of action-sequence arrays.")};setNested("evaluation","openings",value);loadedOpeningsText=openingsText;pending.remove("opening-buffer");report();message="Opening draft saved; Validate checks actual legal transitions."}catch{message=error.localizedDescription}}.buttonStyle(FlatButtonStyle())
            }
            Text("Held-out openings · frozen opponents · both seats · no automatic champion selection").font(.caption).foregroundStyle(MLTheme.muted)
        }
        case "Compute":group("Qualified execution"){
            choice("Learning device","device",[("cpu","CPU reference"),("mps","Apple GPU · validate actual graph"),("auto","Automatic · CPU until speed-qualified")])
            if draft["compute"]==nil {Button("Configure batching & caches"){draft["compute"]=SelfPlayConfiguration.advancedDefaults["compute"]}.buttonStyle(FlatButtonStyle())}
            else {
                choice("Model evaluation device","inferenceDevice",[("cpu","CPU"),("mps","Apple GPU · MPS"),("auto","Automatic")],scope:"compute")
                number("Active games","activeGames",1,8,scope:"compute");number("Inference batch size","batchSize",1,256,scope:"compute");number("Maximum batch wait","batchWait",0,0.05,integer:false,scope:"compute");number("Cache bytes","cacheBytes",0,268435456,scope:"compute")
            }
            Text("Rules and search stay on CPU. Automatic uses CPU until speed-qualified; MPS requires actual graph validation.").font(.caption).foregroundStyle(MLTheme.muted)
        }
        default:group("Resources & History"){number("Run wall seconds","seconds",0.01,3600,integer:false);Text("Per job: 32,768 planned moves · 2 million parameters · 256 observation values/actions");Text("Standard tables: 8 MB · required replay retained independently of capture").font(.caption).foregroundStyle(MLTheme.muted)}
        }
    }
    private func group<C:View>(_ title:String,@ViewBuilder content:()->C)->some View {VStack(alignment:.leading,spacing:12){Text(title).font(.headline);content()}.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))}
    private func choice(_ title:String,_ key:String,_ options:[(String,String)],scope:String?=nil)->some View {VStack(alignment:.leading,spacing:5){Text(title).font(.caption).foregroundStyle(MLTheme.muted);FlatChoiceControl(title:title,selection:(scope.map{nested($0)[key]} ?? draft[key])?.stringValue ?? "",options:options.map{.init(id:$0.0,title:$0.1)}){if let scope{setNested(scope,key,.string($0))}else{draft[key] = .string($0)}}}}
    private func number(_ title:String,_ key:String,_ minimum:Double,_ maximum:Double,integer:Bool=true,search:Bool=false,scope:String?=nil)->some View {
        let fields: [String:JSONValue];if let scope {fields=nested(scope)}else if search,case .object(let value)=draft["search"]{fields=value}else{fields=draft}
        let origin=identity.uuidString+":"+(scope.map{$0+"."} ?? (search ? "search.":""))+key
        return VStack(alignment:.leading,spacing:5){Text(title).font(.caption).foregroundStyle(MLTheme.muted);BoundedNumberField(title:title,value:integer ? .wholeDisplay(fields[key]?.doubleValue ?? 0):.decimal(fields[key]?.doubleValue ?? 0),domain:integer ? .integer(minimum:minimum,maximum:maximum):.decimal(minimum:minimum,maximum:maximum),origin:origin,currentOrigin:{identity.uuidString+":"+(scope.map{$0+"."} ?? (search ? "search.":""))+key},onDraftChange:{_,dirty in if dirty{pending.insert(origin)}else{pending.remove(origin)};report()}){value in
            let json:JSONValue;switch value{case .integer(let n):json = .integer(n);case .decimal(let n):json = .number(n)}
            if let scope{setNested(scope,key,json)}else if search{var next=fields;next[key]=json;draft["search"] = .object(next)}else{draft[key]=json};return true
        }}
    }
    private func nested(_ key:String)->[String:JSONValue] {if case .object(let value)=draft[key]{return value};return [:]}
    private func setNested(_ group:String,_ key:String,_ value:JSONValue){var next=nested(group);next[key]=value;draft[group] = .object(next)}
    private func opponentChoice(_ id:String,_ title:String)->some View {
        let values=nested("evaluation")["opponents"] ?? .array([.string("initial")])
        let names: [JSONValue];if case .array(let x)=values{names=x}else{names=[]}
        return FlatChoiceControl(title:title,selection:names.contains(.string(id)) ? "on":"off",options:[.init(id:"off",title:title+" · excluded"),.init(id:"on",title:title+" · included")]){value in var next=names.filter{$0 != .string(id)};if value=="on"{next.append(.string(id))};setNested("evaluation","opponents",.array(next))}
    }
    private func load(){guard context.isCurrent else{return};original=context.configuration;draft=(try? SelfPlayConfiguration(original).values) ?? SelfPlayConfiguration.defaults;loadedDraft=draft;pending=[];message="";openingsText=(try? String(data:JSONEncoder().encode(nested("evaluation")["openings"] ?? .array([.array([])])),encoding:.utf8)) ?? "[[]]";loadedOpeningsText=openingsText}
    private func clearPending(){context.setNumericPending(origin:context.numericOrigin(key:"selfPlay"),key:"selfPlay",pending:false)}
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"selfPlay"),key:"selfPlay",pending:!closing && (!pending.isEmpty || draft != loadedDraft))}
    private func discard(){guard !closing else{return};closing=true;validation?.cancel();clearPending()}
    private func close(){requestClose()}
    private func apply(){do{_=try SelfPlayConfiguration(["selfPlay":.object(draft)]);closing=true;clearPending();guard context.authority.configure(["selfPlay":.object(draft)],expected:original)else{closing=false;message=context.statusMessage;report();return};requestClose()}catch{message=error.localizedDescription}}
}
