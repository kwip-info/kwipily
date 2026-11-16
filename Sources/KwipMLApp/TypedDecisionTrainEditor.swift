import SwiftUI
import KwipMLCore

struct TypedDecisionTrainEditor:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var store:ModelLoaderStore
    let requestClose:()->Void
    let registerDiscard:(@escaping ()->Void,@escaping ()->Bool)->Void
    @State private var original:[String:JSONValue]=[:]
    @State private var config:[String:JSONValue]=[:]
    @State private var checked:[String:JSONValue]?
    @State private var section="Data & targets"
    @State private var error=""
    @State private var closing=false
    @State private var generation=UUID()
    @State private var mappingColumns:[DataColumnSchema]=[]
    private var address:WorkflowNodeAddress {context.authority.address}
    private var fine:[String:JSONValue] {object(config["fineTuning"])}
    private var run:[String:JSONValue] {object(config["runSetup"])}
    private var questions:[String:JSONValue] {object(fine["questionSet"])}
    private var questionRows:[[String:JSONValue]] {if case .array(let a)=questions["questions"]{return a.map(object)};return []}
    private var states:[String] {if case .array(let a)=fine["stateColumns"]{return a.compactMap(\.stringValue)};return []}
    private var validated:Bool {checked==config && store.fineTuningInspectedConfiguration==config && store.fineTuningPreview?["kind"] == .string("typedDecisionPreparation") && !store.isBusy}
    var body:some View {
        VStack(spacing:0){
            HStack{VStack(alignment:.leading,spacing:4){Text("Train · Typed decisions").font(.title2.weight(.semibold));Text(context.authoredName).font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close",action:close).buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider()
            HStack(spacing:0){
                VStack(alignment:.leading,spacing:5){ForEach(["Data & targets","Questions","Optimization","Calibration","Inspection"],id:\.self){name in Button{section=name}label:{Text(name).frame(maxWidth:.infinity,alignment:.leading).padding(10).background(section==name ? MLTheme.rose.opacity(0.16):Color.clear).contentShape(Rectangle())}.buttonStyle(.plain)};Spacer()}.padding(12).frame(width:220)
                Divider()
                ScrollView{VStack(alignment:.leading,spacing:18){Text(section).font(.title3.weight(.semibold));content}.padding(24).frame(maxWidth:.infinity,alignment:.leading)}
            }
            Divider()
            VStack(alignment:.leading,spacing:8){Text(error.isEmpty ? (store.isBusy ? store.status:validated ? "Mappings and exact token budgets inspected · no training performed":"Draft · validate mappings before Apply"):error).font(.caption).foregroundStyle(error.isEmpty ? MLTheme.muted:MLTheme.rose).textSelection(.enabled);HStack{Button("Revert",action:load).buttonStyle(FlatButtonStyle()).disabled(store.isBusy);Spacer();Button("Validate & inspect",action:validate).buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || host.hasActiveOperation);Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!context.isCurrent || !validated)}}.padding(16)
        }.frame(minWidth:800,idealWidth:1000,minHeight:610,idealHeight:760).background(MLTheme.background)
        .onAppear{load();registerDiscard({discard()}, {!closing && (config != original)})}.onChange(of:config){_,_ in report()}.onDisappear{discard()}
    }
    @ViewBuilder private var content:some View {
        switch section {
        case "Questions":
            TypedDecisionGroup("Reuse an immutable question revision") {
                ForEach(Array(DecisionQuestionResource.choices(host.owner.document).enumerated()),id:\.offset){_,choice in Button("Use "+choice.0){set("questionSet",.object(choice.1));synchronizeTargets()}.buttonStyle(FlatButtonStyle())}
                Text("Reusing copies the selected revision. Edits create a new revision here; other tools and History retain their original content. The trained package carries this exact question set.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            DecisionQuestionEditor(value:Binding(get:{questions},set:{set("questionSet",.object($0));synchronizeTargets()}))
        case "Optimization":
            TypedDecisionGroup("Parameters and objective") {
                choice("Trainable parameters","mode",[("head","Decision head · encoder frozen"),("full","Encoder + decision head")])
                Text("Soft-target cross-entropy over each question's declared options. The unused action/escalate head remains frozen; its probabilities are not a review gate.").font(.caption).foregroundStyle(MLTheme.muted)
                number("Decision-head learning rate","headLearningRate",integer:false)
                if fine["mode"] == .string("full"){number("Encoder learning rate","encoderLearningRate",integer:false)}
                Button("Use qualified AdamW recipe") {if let optimizer=try? OptimizerConfiguration(componentID:"optimizer_adamw",values:[:]) {config["optimizer"]=optimizer.recipeValue;config.removeValue(forKey:"scheduler");checked=nil}}.buttonStyle(FlatButtonStyle())
                Text("AdamW · independent encoder/head rates · gradient norm capped at1. RLCD, LoRA and other optimizers/schedules are unavailable until separately qualified.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            TypedDecisionGroup("Training budget") {number("Total epochs","epochs",scope:"run",integer:true);number("Micro-batch decision items ·1–8","batchSize",scope:"run",integer:true);number("Gradient accumulation steps","accumulation",integer:true);number("Seed","seed",scope:"run",integer:true);flag("Recompute activations during backward","activationCheckpointing")}
            TypedDecisionGroup("Inference selection") {choice("Inference weights","modelSelection",[("finalEpoch","Final epoch"),("bestValidation","Best validation")],scope:"control");Text("Validation selects inference weights. It never fits preprocessing, model weights or calibration. The protected Test population is for independent Evaluate.").font(.caption).foregroundStyle(MLTheme.muted)}
        case "Calibration":
            TypedDecisionGroup("Reserve before fitting") {
                entry("Source group column","groupColumn")
                number("Training groups reserved for calibration · fraction0–0.5","calibrationFraction",integer:false)
                number("Target source-group error · 0–0.5","targetError",integer:false)
                Text("Zero: no calibration. Reserve whole Training groups; Validation and Test stay separate.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            TypedDecisionGroup("Evidence and review") {Text("Reserved groups split into selection and independent audit. Unsupported audit evidence keeps decisions in review.").font(.caption)}
        case "Inspection":
            TypedDecisionGroup("Execution") {Text("CPU · float32 · weights reuse with a fresh optimizer").font(.caption);Text("Memory admission · at most 16 GB, or the lower project limit").font(.caption).foregroundStyle(MLTheme.muted)}
            if let preview=store.fineTuningPreview,preview["kind"] == .string("typedDecisionPreparation") {TypedDecisionGroup("Last explicit input inspection") {Text("Training items · \(preview["trainingItems"]?.integerValue ?? 0)\nValidation items · \(preview["validationItems"]?.integerValue ?? 0)\nHeld-out calibration groups · \(preview["calibrationGroups"]?.integerValue ?? 0)\nEncoder limit · \(preview["encoderTokens"]?.integerValue ?? 0) tokens\nDecision-head limit · \(preview["decisionHeadTokens"]?.integerValue ?? 0) tokens").font(.caption.monospaced());Text("These package limits are not free state capacity. Inspection accounts for every option and question; no text is silently dropped.").font(.caption).foregroundStyle(MLTheme.muted)}}
        default:
            TypedDecisionGroup("State columns") {
                Text("Training and Validation use the same named state fields. Target columns are answers for learning; do not include them among the input fields.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(Array(states.indices),id:\.self){index in HStack{ConfigurationColumnField(title:"State column",value:Binding(get:{states.indices.contains(index) ? states[index]:""},set:{name in var a=states;guard a.indices.contains(index)else{return};a[index]=name;set("stateColumns",.array(a.map(JSONValue.string)))}),columns:mappingColumns);Button("Remove"){var a=states;a.remove(at:index);set("stateColumns",.array(a.map(JSONValue.string)))}.buttonStyle(FlatButtonStyle()).disabled(states.count<=1)}}
                Button("Add state field"){set("stateColumns",.array((states+["field\(states.count+1)"]).map(JSONValue.string)))}.buttonStyle(FlatButtonStyle()).disabled(states.count>=16)
                entry("Source group column","groupColumn")
            }
            TypedDecisionGroup("Targets by question") {
                ForEach(Array(questionRows.enumerated()),id:\.offset){_,q in if let id=q["id"]?.stringValue {
                    VStack(alignment:.leading,spacing:6){Text(id).font(.headline);ConfigurationColumnField(title:"Target column",value:Binding(get:{object(fine["targetColumns"])[id]?.stringValue ?? ""},set:{var a=object(fine["targetColumns"]);a[id] = .string($0);set("targetColumns",.object(a))}),columns:mappingColumns,allowed:["string","int64","float32","float64","bool"]);FlatChoiceControl(title:"Target format",selection:object(fine["targetModes"])[id]?.stringValue ?? "label",options:[.init(id:"label",title:"Declared option key"),.init(id:"distribution",title:"JSON probabilities · every option")]){var a=object(fine["targetModes"]);a[id] = .string($0);set("targetModes",.object(a))}}.padding(10).overlay(Rectangle().stroke(MLTheme.rule))
                }}
                Text("Label targets use stable option keys, including false/true for yes/no. Distribution targets are explicit finite nonnegative JSON maps whose probabilities sum to1; no token shifting or next-word loss.").font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func object(_ value:JSONValue?)->[String:JSONValue]{if case .object(let x)=value{return x};return [:]}
    private func values(_ scope:String)->[String:JSONValue]{scope=="run" ? run:scope=="control" ? object(run["trainingControl"]):fine}
    private func set(_ key:String,_ value:JSONValue,scope:String="fine") {var target=values(scope);target[key]=value;if scope=="control"{var r=run;r["trainingControl"] = .object(target);config["runSetup"] = .object(r)}else{config[scope=="run" ? "runSetup":"fineTuning"] = .object(target)}}
    private func entry(_ title:String,_ key:String)->some View{ConfigurationColumnField(title:title,value:Binding(get:{fine[key]?.stringValue ?? ""},set:{set(key,.string($0))}),columns:mappingColumns)}
    private func number(_ title:String,_ key:String,scope:String="fine",integer:Bool)->some View{VStack(alignment:.leading,spacing:5){Text(title).font(.caption).foregroundStyle(MLTheme.muted);FileInputDraftEntry(title:title,value:Binding(get:{values(scope)[key] ?? 1},set:{set(key,$0,scope:scope)}),parse:{raw in if integer{return Int64(raw).map(JSONValue.integer) ?? .string(raw)};guard let n=Double(raw),n.isFinite else{return .string(raw)};return .number(n)}).id(generation.uuidString+scope+key)}}
    private func choice(_ title:String,_ key:String,_ options:[(String,String)],scope:String="fine")->some View{VStack(alignment:.leading,spacing:5){Text(title).font(.caption).foregroundStyle(MLTheme.muted);FlatChoiceControl(title:title,selection:values(scope)[key]?.stringValue ?? options[0].0,options:options.map{.init(id:$0.0,title:$0.1)}){set(key,.string($0),scope:scope)}}}
    private func flag(_ title:String,_ key:String)->some View{Toggle(title,isOn:Binding(get:{fine[key] == .bool(true)},set:{set(key,.bool($0))})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}
    private func synchronizeTargets(){var columns=object(fine["targetColumns"]),modes=object(fine["targetModes"]);let ids=Set(questionRows.compactMap{$0["id"]?.stringValue});columns=columns.filter{ids.contains($0.key)};modes=modes.filter{ids.contains($0.key)};for id in ids where columns[id]==nil{columns[id] = .string(id)};set("targetColumns",.object(columns));set("targetModes",.object(modes))}
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"fineTuningDraft"),key:"fineTuningDraft",pending:!closing && config != original)}
    private func load(){guard context.isCurrent else{return};host.cancelFineTuningInspection(address);original=context.configuration;config=original;mappingColumns=host.sourceProvider.recordedSchema(for:address,in:host.owner.document,inputPort:"training");var r=run;r["device"] = .string("cpu");if r["trainingControl"]==nil,let control=try? JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(TrainingControlPolicy.factory)){r["trainingControl"]=control};config["runSetup"] = .object(r);checked=nil;error="";generation=UUID();report()}
    private func validate(){guard context.isCurrent else{return};do{var next=fine;next["questionSet"] = .object(DecisionQuestionResource.advanced(questions,from:object(object(original["fineTuning"])["questionSet"])));next=try TypedDecisionConfiguration.validateTraining(next);config["fineTuning"] = .object(next);var d=host.owner.document;guard let i=d.nodes.firstIndex(where:{$0.address==address})else{return};d.nodes[i].payload = .train(config:config);try WorkflowCodec.validate(d);context.setNumericPending(origin:context.numericOrigin(key:"fineTuningDraft"),key:"fineTuningDraft",pending:false);checked=config;store.clearFineTuningInspection();if !host.inspectFineTuning(address,draft:config){checked=nil;error=host.statusMessage;report()}else{error=""}}catch{self.error=error.localizedDescription;report()}}
    private func apply(){guard validated else{return};context.setNumericPending(origin:context.numericOrigin(key:"fineTuningDraft"),key:"fineTuningDraft",pending:false);guard context.authority.configure(config,expected:original)else{error=context.statusMessage;report();return};closing=true;requestClose()}
    private func discard(){guard !closing else{return};closing=true;host.cancelFineTuningInspection(address);context.setNumericPending(origin:context.numericOrigin(key:"fineTuningDraft"),key:"fineTuningDraft",pending:false)}
    private func close(){requestClose()}
}
