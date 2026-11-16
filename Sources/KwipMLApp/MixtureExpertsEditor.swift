import SwiftUI
import Combine
import KwipMLCore

struct MixtureExpertsInspector:View {
    @ObservedObject var context:WorkflowModelNodeContext
    var makeValidation:(()->MixtureValidationSession?)?
    @State private var editor:ExpandedConfigurationWindowController?
    @Environment(\.appearanceTheme) private var theme
    var body:some View {
        VStack(alignment:.leading,spacing:10){
            Text("\(context.configuration["expert_count"]?.integerValue ?? 4) experts · top-\(context.configuration["top_k"]?.integerValue ?? 2) · dropless").font(.caption)
            Button("Expand / collapse container"){_=context.openBlock()}.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
            Button("Configure mixture…"){
                if let editor,!editor.didClose{editor.present();return}
                let next=ExpandedConfigurationWindowController(title:"Mixture of experts",identifier:"info.kwip.KwipML.moe-editor"){controller in
                    AnyView(MixtureExpertsEditor(context:context,validation:makeValidation?(),requestClose:{[weak controller] in controller?.close()},register:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty}).windowActivatingControls().environment(\.appearanceTheme,theme))
                };editor=next;next.present()
            }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
            Text("All experts remain resident. Configure saves settings; Train runs the model.").font(.caption).foregroundStyle(MLTheme.muted)
        }.onDisappear{editor?.close();editor=nil}
    }
}

struct MixtureExpertsEditor:View {
    @ObservedObject var context:WorkflowModelNodeContext
    var validation:MixtureValidationSession?
    var requestClose:()->Void
    var register:(@escaping()->Void,@escaping()->Bool)->Void
    @State private var validationObservation:AnyCancellable?
    @State private var config:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var section="Experts"
    @State private var selected=0
    @State private var message=""
    @State private var showRouterSource=false
    @State private var routerSource=""
    @State private var routerSourceError=""
    @State private var validated:[String:JSONValue]?
    private let sections=["Interface","Experts","Routing","Training & limits","Capture"]
    private var count:Int {config["expert_count"]?.integerValue ?? 4}
    private var dirty:Bool {!numericPending.isEmpty || config != original}
    var body:some View {
        VStack(spacing:0){
            HStack{VStack(alignment:.leading){Text("Mixture of experts").font(.headline);Text(context.authoredName).font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close",action:requestClose).buttonStyle(FlatButtonStyle())}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider().overlay(MLTheme.rule)
                HStack(spacing:0){
                    VStack(alignment:.leading,spacing:0){ForEach(sections,id:\.self){name in Button{nameChanged(name)}label:{Text(name).frame(maxWidth:.infinity,alignment:.leading).padding(13).background(section==name ? MLTheme.rose.opacity(0.22):.clear)}.buttonStyle(.plain)};Spacer()}.frame(width:190)
                    Divider().overlay(MLTheme.rule)
                    ThemedScrollView{VStack(alignment:.leading,spacing:16){Text(section).font(.title2.weight(.semibold));content}.padding(22).frame(maxWidth:.infinity,alignment:.leading)}
                }
            Divider().overlay(MLTheme.rule)
            VStack(alignment:.leading,spacing:10){Text(message.isEmpty ? "Draft · Validate values and gradients before applying":message).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled);HStack{Button("Revert"){load()}.buttonStyle(FlatButtonStyle());Spacer();Text("Apply saves settings. Run Flow trains.").font(.caption).foregroundStyle(MLTheme.muted);Button(validation?.isBusy == true ? "Cancel validation":"Validate"){if validation?.isBusy == true{validation?.cancel()}else{validate()}}.buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || !numericPending.isEmpty);Button("Apply"){context.setNumericPending(origin:context.numericOrigin(key:"moeDraft"),key:"moeDraft",pending:false);if context.authority.configure(config,expected:original){original=config;requestClose()}else{message=context.statusMessage;report()}}.buttonStyle(FlatButtonStyle()).disabled(validated != config || !numericPending.isEmpty || context.isBusy)}}.padding(16)
        }.background(MLTheme.background).foregroundStyle(Color.primary).onAppear{load();validationObservation=validation?.objectWillChange.sink{_ in DispatchQueue.main.async{message=validation?.message ?? "";validated=validation?.validated}};register({validation?.cancel()}, {dirty})}.onChange(of:config){_,_ in validation?.cancel();validated=nil;report()}.onChange(of:context.contextIdentity){_,_ in requestClose()}.onDisappear{validation?.cancel();validationObservation=nil;context.setNumericPending(origin:context.numericOrigin(key:"moeDraft"),key:"moeDraft",pending:false)}
    }
    @ViewBuilder private var content:some View {
        switch section {
        case "Interface":
            numeric("Input features","input_features",1...512);numeric("Output features","output_features",1...512)
            toggle("Use valid-unit mask","mask")
            Text("Float32 [N,D] or [N,T,D]. Mask uses int64 zero/one [N,1] or [N,T]. Masked output is zero; no masked units enter routing statistics.").font(.caption).foregroundStyle(MLTheme.muted)
        case "Experts":
            Text("Expand the MoE container on the main canvas to edit each named expert’s layers and their real input/output ports. Close this settings draft before editing layers.").font(.caption).foregroundStyle(MLTheme.muted)
            numeric("Expert count","expert_count",2...4)
            ForEach(Array(0..<count),id:\.self){i in expertRow(i)}
            VStack(alignment:.leading,spacing:10){Text("Selected expert").font(.headline);TextField("Expert label",text:Binding(get:{expert(selected)["name"]?.stringValue ?? ""},set:{var e=expert(selected);e["name"] = .string($0);setExpert(e,at:selected)})).flatField();Toggle("Trainable",isOn:Binding(get:{expert(selected)["trainable"]==true},set:{var e=expert(selected);e["trainable"] = .bool($0);setExpert(e,at:selected)})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose));Text("Stable identity · \(expert(selected)["id"]?.stringValue ?? "")").font(.caption).foregroundStyle(MLTheme.muted);Button("Duplicate into next unused slot"){duplicate()}.buttonStyle(FlatButtonStyle()).disabled(count==4);Button("Remove selected expert"){remove()}.buttonStyle(FlatButtonStyle()).disabled(count<=2)}.padding(14).overlay(Rectangle().stroke(MLTheme.rule))
            Text("Reducing the count removes the last active slot from this model; previous History is preserved. Expert order is the saved tie order. Labels and layout do not rename parameter identities.").font(.caption).foregroundStyle(MLTheme.muted)
        case "Routing":
            Text("Routing belongs to this container. Edit its bounded router architecture here; the main canvas shows expert layers only.").font(.caption).foregroundStyle(MLTheme.muted)
            numeric("Experts per unit","top_k",1...count)
            choice("Routing weights","weighting",[("normalized","Normalize selected experts"),("probability","Retain full-softmax probabilities")]);number("Temperature","temperature",0.0001...100)
            if config["top_k"]==1 && config["weighting"] == .string("normalized"){Text("Normalized top-1 has no task-loss gradient to the gate. Auxiliary loss may train the gate.").font(.caption).foregroundStyle(.orange)}
            Text("Each observation or sequence position routes independently. Score ties use the saved expert order.").font(.caption).foregroundStyle(MLTheme.muted)
            Divider().overlay(MLTheme.rule)
            MixtureRouterControls(source:Binding(get:{config["router"]?.stringValue ?? ""},set:{config["router"] = .string($0);routerSource=MixtureRouterPresentation.formatted($0)}),registry:context.authority.owner.modelRegistry,origin:context.numericOrigin(key:"router"),pending:{key,pending in setPending(key,pending)}).disabled(numericPending.contains("router-source"))
            Button {showRouterSource.toggle()} label:{HStack{Image(systemName:showRouterSource ? "chevron.down":"chevron.right");Text("Advanced router JSON");Spacer()}}.buttonStyle(FlatButtonStyle())
            if showRouterSource {
                Text("For topology or advanced edits. Use JSON draft stages the source here; Validate and Apply below commit the mixture settings.").font(.caption).foregroundStyle(MLTheme.muted)
                GameRuleCodeEditor(text:Binding(get:{routerSource},set:{routerSource=$0;setPending("router-source",$0 != MixtureRouterPresentation.formatted(config["router"]?.stringValue ?? ""))}),language:.json,label:"Advanced router JSON",limit:16384).frame(height:340).overlay(Rectangle().stroke(MLTheme.rule))
                HStack {
                    Button("Revert JSON draft"){routerSource=MixtureRouterPresentation.formatted(config["router"]?.stringValue ?? "");routerSourceError="";setPending("router-source",false)}.buttonStyle(FlatButtonStyle())
                    Spacer()
                    Button("Use JSON draft"){do{let source=try MixtureRouterPresentation.compact(routerSource);config["router"] = .string(source);routerSource=MixtureRouterPresentation.formatted(source);routerSourceError="";setPending("router-source",false)}catch{routerSourceError=error.localizedDescription}}.buttonStyle(FlatButtonStyle())
                }
                if !routerSourceError.isEmpty {Text(routerSourceError).font(.caption).foregroundStyle(MLTheme.rose)}
            }

        case "Training & limits":
            number("Load balance coefficient","balance_coefficient",0...10);number("Router z-loss coefficient","z_coefficient",0...10)
            Text("Shared Train optimizer, schedule and model selection remain in Train. Regularizers contribute only to its training objective; validation selection uses the task metric.").font(.caption).foregroundStyle(MLTheme.muted)
            Text("Process all selected assignments · no silent drops. All experts/router remain resident; concentrated dispatch and existing resource/storage limits apply.").font(.caption)
        default:
            choice("Routing capture","capture",[("off","Off"),("aggregate","Aggregate routing"),("sample","Aggregate + bounded unit sample")]);numeric("Maximum sampled unit visits","sample_limit",0...1024)
            Text("Routing summary is aggregate evidence, not raw data. Samples are bounded, exclude raw source content, and never determine the learning population. Off means Not captured.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func expertRow(_ i:Int)->some View {
        let record=expert(i)
        return HStack {
            Button {selected=i} label: {Text(record["name"]?.stringValue ?? "Expert").frame(maxWidth:.infinity,alignment:.leading)}.buttonStyle(.plain)
            Text(record["trainable"] == .bool(true) ? "Trainable":"Frozen").font(.caption)

        }.padding(12).background(selected==i ? MLTheme.rose.opacity(0.15):MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func nameChanged(_ value:String){section=value}
    private func load(){guard context.isCurrent else{return};config=context.configuration;original=config;selected=0;validated=nil;message="";numericPending=[];showRouterSource=false;routerSource=MixtureRouterPresentation.formatted(config["router"]?.stringValue ?? "");routerSourceError="";report()}
    private func setPending(_ key:String,_ pending:Bool){if pending{numericPending.insert(key)}else{numericPending.remove(key)};report()}
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"moeDraft"),key:"moeDraft",pending:dirty)}
    private func validate(){do{_=try MixtureExpertsConfiguration(values:config,registry:context.authority.owner.modelRegistry);guard let validation else{message="A current workspace runtime is required for finite gradient validation.";return};validation.run(config);message=validation.message}catch{validated=nil;message=error.localizedDescription}}
    private func expert(_ i:Int)->[String:JSONValue]{guard let text=config["expert_\(i+1)"]?.stringValue,let data=text.data(using:.utf8),case .object(let row)=try? JSONDecoder().decode(JSONValue.self,from:data) else{return [:]};return row}
    private func encode(_ value:JSONValue)->String {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];return (try? String(data:encoder.encode(value),encoding:.utf8)) ?? ""}
    private func setExpert(_ row:[String:JSONValue],at index:Int){config["expert_\(index+1)"] = .string(encode(.object(row)))}
    private func duplicate(){guard count<4 else{return};var row=expert(selected);row["id"] = .string("expert_"+UUID().uuidString.replacingOccurrences(of:"-",with:"").prefix(20));row["name"] = .string((row["name"]?.stringValue ?? "Expert")+" copy");setExpert(row,at:count);config["expert_count"] = .integer(Int64(count+1));message="Update the router output width to match the new expert count before applying."}
    private func remove(){guard count>2 else{return};let retired=expert(selected);if selected<count-1{for i in selected..<(count-1){setExpert(expert(i+1),at:i)}};setExpert(retired,at:count-1);config["expert_count"] = .integer(Int64(count-1));if let k=config["top_k"]?.integerValue,k>=count{config["top_k"] = .integer(Int64(count))};selected=min(selected,count-1);message="Expert removed from the active model. Update router output width; structural changes require a new compatible checkpoint."}
    private func numeric(_ label:String,_ key:String,_ range:ClosedRange<Int>)->some View {number(label,key,Double(range.lowerBound)...Double(range.upperBound),integer:true)}
    @State private var numericPending:Set<String>=[]
    private func number(_ label:String,_ key:String,_ range:ClosedRange<Double>,integer:Bool=false)->some View {VStack(alignment:.leading,spacing:5){Text(label).font(.caption);BoundedNumberField(title:label,value:integer ? .wholeDisplay(config[key]?.doubleValue ?? range.lowerBound):.decimal(config[key]?.doubleValue ?? range.lowerBound),domain:integer ? .integer(minimum:range.lowerBound,maximum:range.upperBound):.decimal(minimum:range.lowerBound,maximum:range.upperBound),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{_,pending in if pending{numericPending.insert(key)}else{numericPending.remove(key)};report()}){candidate in switch candidate{case .integer(let n):config[key] = .integer(n);if key=="expert_count"{selected=min(selected,Int(n)-1)};case .decimal(let n):config[key] = .number(n)};return true}}}
    private func toggle(_ label:String,_ key:String)->some View {Toggle(label,isOn:Binding(get:{config[key]==true},set:{config[key] = .bool($0)})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}
    private func choice(_ label:String,_ key:String,_ options:[(String,String)])->some View {VStack(alignment:.leading,spacing:5){Text(label).font(.caption);FlatChoiceControl(title:label,selection:config[key]?.stringValue ?? "",options:options.map{.init(id:$0.0,title:$0.1)}){config[key] = .string($0)}}}
}
