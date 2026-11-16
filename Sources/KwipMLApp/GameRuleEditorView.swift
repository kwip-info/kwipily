import SwiftUI
import AppKit
import KwipMLCore

struct GameRuleConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @State private var showEditor=false
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Definition",symbol:"list.bullet.rectangle",identifier:"game.definition",initiallyExpanded:true,accent:ComponentVisualFamily.data.primary) {
                if case .object(let d)?=node.config["definition"] {
                    Text(d["name"]?.stringValue ?? "Game rules").font(.headline)
                    Text("Version 1 · two players · deterministic rules").foregroundStyle(MLTheme.muted)
                }
                Button("Edit rules…"){showEditor=true}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(context.isBusy).accessibilityIdentifier("game.rules.edit")
            }
            Text("Apply saves one definition revision. Run Flow executes only authored tests here; Summary, Tests, Trace and Definition are standard table outputs. No model training or game visualization.").font(.caption).foregroundStyle(MLTheme.muted)
        }.sheet(isPresented:$showEditor){GameRuleEditorView(context:context,node:node)}
    }
}

private struct GameRuleEditorView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.dismiss) private var dismiss
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var section="General"
    @State private var text=""
    @State private var sourcePending=false
    @State private var fieldPending=Set<String>()
    @State private var discardPrompt=false
    @State private var fieldReloadID=UUID()
    @State private var error=""
    @State private var runner:GameRuleValidationSession?
    private var values:[String:JSONValue] {["definition":.object(draft)]}
    var body:some View {
        VStack(spacing:0) {
            HStack {VStack(alignment:.leading){Text("Game definition").font(.title2.weight(.semibold));Text("Author state, actions and rules · no executable scripts").font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close"){close()}.buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            Divider()
            HStack(alignment:.top,spacing:0) {
                navigation
                Divider()
                VStack(alignment:.leading,spacing:0) {
                    HStack(alignment:.top) {
                        VStack(alignment:.leading,spacing:5) {
                            Text(section == "Source" ? "Definition JSON":section).font(.title3.weight(.semibold))
                            Text(sectionDescription).font(.caption).foregroundStyle(MLTheme.muted)
                        }
                        Spacer()
                        Text(sourcePending || !fieldPending.isEmpty ? "Section edits pending":values == original ? "Saved definition":"Draft changes")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }.padding(18)
                    Divider()
                    if section=="Tests" || section=="Source" {
                        expressionEditor.padding(18).disabled(runner?.isBusy == true)
                    } else {
                        ScrollView {VStack(alignment:.leading,spacing:14) {
                            if section=="General" {general}
                            else if section=="State" || section=="Actions" {schemaEditor.id(fieldReloadID)}
                            else {GameRuleFormView(draft:$draft,section:section)}
                        }.padding(18).frame(maxWidth:.infinity,alignment:.leading)}.disabled(runner?.isBusy == true)
                    }
                    if !error.isEmpty {Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding([.horizontal,.bottom],18)}
                }.frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading)
            }
            Divider()
            if let runner {
                GameRuleValidationControls(runner:runner,values:values,original:original,sourcePending:sourcePending || !fieldPending.isEmpty,onApply:apply,onRevert:reload,onImport:importJSON,onExport:exportJSON)
            }
        }.frame(minWidth:850,idealWidth:1060,maxWidth:.infinity,minHeight:620,idealHeight:780,maxHeight:.infinity)
            .background(MLTheme.background).interactiveDismissDisabled(sourcePending || !fieldPending.isEmpty || values != original)
            .alert("Discard unsaved rule changes?",isPresented:$discardPrompt){Button("Keep editing",role:.cancel){};Button("Discard changes",role:.destructive){runner?.cancel();dismiss()}}message:{Text("The saved canvas definition is preserved. Apply before closing to keep your draft.")}
            .onAppear{reload();runner=context.makeGameRuleValidationSession()}
            .onDisappear{runner?.cancel();context.setNumericPending(origin:context.numericOrigin(key:"gameRules"),key:"gameRules",pending:false)}
            .onChange(of:draft){_,_ in publishPending()}.onChange(of:sourcePending){_,_ in publishPending()}.onChange(of:fieldPending){_,_ in publishPending()}
    }
    private var navigation:some View {
        VStack(alignment:.leading,spacing:18) {
            ForEach([("Definition",["General","State","Actions"]),("Behaviour",["Rules","Observations"]),("Verify & share",["Tests","Source"])],id:\.0) {group in
                VStack(alignment:.leading,spacing:3) {
                    Text(group.0.uppercased()).font(.system(size:10,weight:.semibold)).foregroundStyle(MLTheme.muted).padding(.horizontal,12).padding(.bottom,5)
                    ForEach(group.1,id:\.self){item in
                        Button{switchSection(item)}label:{
                            HStack(spacing:9){Image(systemName:sectionIcon(item)).frame(width:16);Text(item == "Source" ? "JSON source":item);Spacer();if section==item{Image(systemName:"chevron.right").font(.caption2)}}
                                .padding(.horizontal,12).padding(.vertical,10).frame(maxWidth:.infinity,alignment:.leading)
                                .background(section==item ? MLTheme.rose.opacity(0.13):Color.clear)
                                .overlay(alignment:.leading){if section==item{Rectangle().fill(MLTheme.rose).frame(width:3)}}
                        }.buttonStyle(.plain).disabled(sourcePending || !fieldPending.isEmpty || runner?.isBusy == true)
                    }
                }
            }
            Spacer()
            Text("One definition.\nNo executable scripts.").font(.caption).foregroundStyle(MLTheme.muted).padding(.horizontal,12)
        }.padding(.vertical,20).padding(.horizontal,8).frame(width:180).frame(maxHeight:.infinity).background(MLTheme.panel.opacity(0.35))
    }
    private func sectionIcon(_ item:String)->String {
        switch item {case "General":return "doc.text";case "State":return "square.stack.3d.up";case "Actions":return "cursorarrow";case "Rules":return "arrow.triangle.branch";case "Observations":return "eye";case "Tests":return "checklist";default:return "curlybraces"}
    }
    private var sectionDescription:String {
        switch section {
        case "General":return "Name your game and choose a starting example."
        case "State":return "The values that describe one position in the game."
        case "Actions":return "The candidate moves a player can request."
        case "Rules":return "Whose turn it is, what is legal, and how a move changes state."
        case "Observations":return "The numeric information available from each player's perspective."
        case "Tests":return "Specify actions and expected outcomes before applying a definition."
        default:return "Advanced editing of the same definition used by the structured controls."
        }
    }
    private var general:some View {
        VStack(alignment:.leading,spacing:14) {
            InspectorSection("Identity & capabilities",symbol:"doc.text",identifier:"game.identity",initiallyExpanded:true,accent:ComponentVisualFamily.data.primary) {
                Text("Name").font(.caption).foregroundStyle(MLTheme.muted)
                TextField("Definition name",text:Binding(get:{draft["name"]?.stringValue ?? ""},set:{draft["name"] = .string($0)})).flatField()
                Text("This evaluator supports two players (0 and 1), deterministic transitions, bounded finite actions and numeric observations. Rules and array dimensions are authored data; game names do not select engine behavior.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Learning examples",symbol:"book",identifier:"game.examples",initiallyExpanded:true,accent:ComponentVisualFamily.data.primary) {
                Text("Load an editable example into this draft. Saved canvas changes only after Validate and Apply.").font(.caption)
                ForEach(GameRuleConfiguration.examples,id:\.self){name in Button("Load "+name){do {draft=try GameRuleConfiguration.example(name);error=""}catch{self.error=error.localizedDescription}}.buttonStyle(FlatButtonStyle()).disabled(runner?.isBusy == true)}
            }
            Text("Validate runs a disposable source-free worker job against this draft. It cannot run other canvas nodes or repeat API/database writes. Run Flow publishes the complete standard test and trace tables for the applied revision.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private var schemaEditor:some View {
        let key=section=="State" ? "state":"actions"
        let fields=object(key)
        return VStack(alignment:.leading,spacing:14) {
            Text(section+" schema").font(.headline)
            Text(section=="State" ? "Every state field needs a type, finite bounds and initial value. Arrays use fixed length and a typed item schema. No state fields are inferred from the game name.":"Actions use finite integer, Boolean or enum fields. Their product cannot exceed 256 candidate moves. Legal rules decide which candidates are allowed.").font(.caption).foregroundStyle(MLTheme.muted)
            ForEach(fields.keys.sorted(),id:\.self){name in
                GameRuleFieldView(name:name,value:fields[name] ?? .object([:]),isState:key=="state",onPending:{pending in if pending {fieldPending.insert(name)}else{fieldPending.remove(name)}},onApply:{value in var f=object(key);f[name]=value;draft[key] = .object(f);fieldPending.remove(name)},onRemove:{var f=object(key);f.removeValue(forKey:name);draft[key] = .object(f);fieldPending.remove(name)})
            }
            Button("Add field") {var f=fields;var i=1;while f["field\(i)"] != nil {i+=1};var v:[String:JSONValue]=["type":.string("integer"),"minimum":0,"maximum":10];if key=="state"{v["initial"]=0};f["field\(i)"] = .object(v);draft[key] = .object(f)}.buttonStyle(FlatButtonStyle())
            Text("Field names and richer nested constraints can be changed in Source. Update references when renaming; validation refuses missing fields.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private var expressionEditor:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                Label("JSON",systemImage:"curlybraces").font(.caption.weight(.semibold))
                Text("\(text.split(separator:"\n",omittingEmptySubsequences:false).count) lines · \(text.utf8.count) bytes").font(.caption).foregroundStyle(MLTheme.muted)
                Spacer()
                Button("Format JSON",action:formatText).buttonStyle(FlatButtonStyle())
            }
            GameRuleCodeEditor(text:Binding(get:{text},set:{if text != $0{text=$0;sourcePending=true}}),language:.json,label:section == "Source" ? "Definition JSON":"Rule tests JSON")
                .frame(minHeight:180,maxHeight:.infinity).overlay(Rectangle().stroke(MLTheme.muted.opacity(0.35)))
            HStack {
                Button("Update draft section",action:updateSection).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!sourcePending)
                Button("Discard section edits"){loadText();sourcePending=false;error=""}.buttonStyle(FlatButtonStyle()).disabled(!sourcePending)
                Spacer()
            }
            Text(instructions).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
        }.frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading)
    }
    private func formatText(){do{let value=try GameRuleConfiguration.decodeValue(Data(text.utf8));let next=formatted(value);if text != next{text=next;sourcePending=true};error=""}catch{self.error=error.localizedDescription}}
    private var instructions:String {
        switch section {
        case "Rules":return "Edit currentPlayer, legal, transitions, terminal, reward and invariants as JSON. Expressions use state.field, action.field, params.field, actor and player. Transition updates may use next.field after its dependency. Only arithmetic, comparisons, conditionals, bounded comprehensions and declared functions are admitted. No Python calls or imports."
        case "Observations":return "Edit the numeric observation expression array. The player variable selects the requested perspective. Each expression produces one finite numeric value; shape is derived from this array. No fixed board/action layout is assumed."
        case "Tests":return "Each test has name, optional complete initial state, an actions array and expect (partial state, terminal, winner, rewards), or expectError. Errors expected by a test are not automatic success: state/outcome assertions must still match. Up to 32 tests and 128 moves per test, with 2,048 trace moves total."
        default:return "Version1 canonical definition JSON. Source and the typed field editors use the same draft. State/action/observation changes require revalidation. Canonical bytes and SHA-256 are available from the Definition output after Run Flow."
        }
    }
    private var keys:[String] {section=="Rules" ? ["currentPlayer","legal","transitions","terminal","reward","invariants"]:section=="Observations" ? ["observations"]:section=="Tests" ? ["tests"]:Array(draft.keys)}
    private func object(_ key:String)->[String:JSONValue] {if case .object(let v)?=draft[key]{return v};return [:]}
    private func formatted(_ value:JSONValue)->String {let e=JSONEncoder();e.outputFormatting=[.prettyPrinted,.sortedKeys,.withoutEscapingSlashes];return (try? String(decoding:e.encode(value),as:UTF8.self)) ?? "{}"}
    private func loadText(){text=formatted(section=="Observations" ? draft["observations"] ?? .array([]):section=="Tests" ? draft["tests"] ?? .array([]):.object(draft.filter{keys.contains($0.key)}))}
    private func switchSection(_ value:String){section=value;loadText();error=""}
    private func updateSection(){do {let value=try GameRuleConfiguration.decodeValue(Data(text.utf8));if section=="Observations" || section=="Tests" {guard case .array=value else{throw DataFlowContractError("Use a JSON array for this section.")};draft[section=="Tests" ? "tests":"observations"]=value} else {guard case .object(let fields)=value else{throw DataFlowContractError("Use a JSON object.")};if section=="Source"{draft=try GameRuleConfiguration.decode(Data(text.utf8))}else{guard Set(fields.keys)==Set(keys) else{throw DataFlowContractError("Keep exactly the displayed rule fields.")};draft.merge(fields){_,new in new}}};sourcePending=false;error=""}catch{self.error=error.localizedDescription}}
    private func reload(){original=context.node?.config ?? node.config;if case .object(let d)?=original["definition"]{draft=d};sourcePending=false;fieldPending=[];fieldReloadID=UUID();error="";loadText();publishPending()}
    private func publishPending(){context.setNumericPending(origin:context.numericOrigin(key:"gameRules"),key:"gameRules",pending:(sourcePending || !fieldPending.isEmpty) || values != original)}
    private func apply(){guard !sourcePending,fieldPending.isEmpty,runner?.validated==values else{return};if context.applyTableConfiguration(values,expected:original){original=values;error="";publishPending()}else{error=context.statusMessage}}
    private func close(){if sourcePending || !fieldPending.isEmpty || values != original {discardPrompt=true}else{runner?.cancel();dismiss()}}
    private func importJSON(){let panel=NSOpenPanel();panel.allowedContentTypes=[.json];panel.allowsMultipleSelection=false;guard panel.runModal() == .OK,let url=panel.url else{return};do {let scoped=url.startAccessingSecurityScopedResource();defer{if scoped{url.stopAccessingSecurityScopedResource()}};draft=try GameRuleConfiguration.decode(DiagnosticFileIO.read(root:url.deletingLastPathComponent(),parts:[url.lastPathComponent],limit:131072));sourcePending=false;fieldPending=[];loadText();error=""}catch{self.error=error.localizedDescription}}
    private func exportJSON(){let panel=NSSavePanel();panel.nameFieldStringValue="game-definition.json";panel.allowedContentTypes=[.json];guard panel.runModal() == .OK,let url=panel.url else{return};do {let scoped=url.startAccessingSecurityScopedResource();defer{if scoped{url.stopAccessingSecurityScopedResource()}};try Data(formatted(.object(draft)).utf8).write(to:url);error=""}catch{self.error=error.localizedDescription}}
}

private struct GameRuleValidationControls:View {
    @ObservedObject var runner:GameRuleValidationSession
    let values:[String:JSONValue];let original:[String:JSONValue];let sourcePending:Bool
    let onApply:()->Void;let onRevert:()->Void;let onImport:()->Void;let onExport:()->Void
    var body:some View {
        VStack(spacing:12) {
            HStack(spacing:8) {
                Image(systemName:runner.isBusy ? "hourglass":sourcePending ? "pencil":"checklist")
                Text(sourcePending ? "Update or discard the section edits to navigate and validate.":runner.lastTested != nil && runner.lastTested != values ? "Draft changed · validate again.":runner.message)
                    .textSelection(.enabled)
                Spacer(minLength:0)
            }.font(.caption).foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,alignment:.leading)
            HStack(spacing:8) {
                Button("Import JSON…",action:onImport).buttonStyle(FlatButtonStyle()).disabled(runner.isBusy || sourcePending)
                Button("Export JSON…",action:onExport).buttonStyle(FlatButtonStyle()).disabled(runner.isBusy || sourcePending)
                Spacer()
                Button("Revert changes",action:onRevert).buttonStyle(FlatButtonStyle()).disabled(runner.isBusy || (!sourcePending && values==original))
                if runner.isBusy {Button("Cancel validation"){runner.cancel()}.buttonStyle(FlatButtonStyle())}
                Button("Validate & run tests"){runner.run(values)}.buttonStyle(FlatButtonStyle()).disabled(sourcePending || runner.isBusy)
                Button("Apply",action:onApply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(sourcePending || runner.validated != values || runner.isBusy || values==original).accessibilityIdentifier("game.rules.apply")
            }
        }.padding(16).background(MLTheme.panel.opacity(0.45))
    }
}

private struct GameRuleFieldView:View {
    let name:String;let value:JSONValue;let isState:Bool
    let onPending:(Bool)->Void;let onApply:(JSONValue)->Void;let onRemove:()->Void
    @State private var source="";@State private var error=""
    @State private var type="integer";@State private var minimum="0";@State private var maximum="10";@State private var initial="0";@State private var vocabulary="[]"
    @State private var dirty=false
    var body:some View {
        InspectorSection(name,symbol:"list.bullet.rectangle",identifier:"game.field.\(name)",initiallyExpanded:true,accent:ComponentVisualFamily.data.primary) {
            Text("Type").font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:"Type",selection:type,options:["integer","number","boolean","string","array","optional"].map{.init(id:$0,title:$0)}){if type != $0 {type=$0;mark()}}
            if type=="integer" || type=="number" {
                entry("Minimum",text:$minimum);entry("Maximum",text:$maximum)
            }
            if type=="string" {entry("Allowed values · JSON text array",text:$vocabulary)}
            if type=="array" || type=="optional" {
                Text("Nested type, bounds & initial value · JSON").font(.caption).foregroundStyle(MLTheme.muted)
                GameRuleCodeEditor(text:Binding(get:{source},set:{if source != $0{source=$0;mark()}}),language:.json,label:"Nested schema JSON").frame(height:160).overlay(Rectangle().stroke(MLTheme.muted.opacity(0.35)))
            } else if isState {entry("Initial value · JSON literal",text:$initial)}
            HStack {Button("Update field",action:update).buttonStyle(FlatButtonStyle()).disabled(!dirty);Button("Discard field edits"){reload()}.buttonStyle(FlatButtonStyle()).disabled(!dirty);Button("Remove field",action:onRemove).buttonStyle(FlatButtonStyle())}
            if !error.isEmpty{Text(error).foregroundStyle(MLTheme.rose).font(.caption)}
        }.onAppear(perform:reload).onChange(of:value){_,_ in reload()}
    }
    private func entry(_ title:String,text:Binding<String>)->some View {VStack(alignment:.leading,spacing:4){Text(title).font(.caption).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{text.wrappedValue},set:{let next=String($0.prefix(131072));if text.wrappedValue != next {text.wrappedValue=next;mark()}})).flatField()}}
    private func mark(){dirty=true;onPending(true)}
    private func parse(_ text:String)throws->JSONValue {try GameRuleConfiguration.decodeValue(Data(text.utf8))}
    private func update(){do {
        let result:JSONValue
        if type=="array" || type=="optional" {let v=try parse(source);guard case .object(let fields)=v,fields["type"] == .string(type) else{throw DataFlowContractError("Nested schema type must match the selected type.")};result=v}
        else {var fields:[String:JSONValue]=["type":.string(type)];if type=="integer" || type=="number" {fields["minimum"]=try parse(minimum);fields["maximum"]=try parse(maximum)};if type=="string"{fields["values"]=try parse(vocabulary)};if isState{fields["initial"]=try parse(initial)};result = .object(fields)}
        onApply(result);dirty=false;onPending(false);error=""
    }catch{self.error=error.localizedDescription}}
    private func display(_ value:JSONValue)->String{let e=JSONEncoder();e.outputFormatting=[.prettyPrinted,.sortedKeys];return (try? String(decoding:e.encode(value),as:UTF8.self)) ?? "null"}
    private func reload(){source=display(value);if case .object(let v)=value {type=v["type"]?.stringValue ?? "integer";minimum=display(v["minimum"] ?? 0);maximum=display(v["maximum"] ?? 10);initial=display(v["initial"] ?? .null);vocabulary=display(v["values"] ?? .array([]))};dirty=false;onPending(false);error=""}
}

/// Structured expression authoring over the same JSON IR as Source, never a second engine.
private struct GameRuleFormView:View {
    @Binding var draft:[String:JSONValue]
    let section:String
    private var accent:Color {ComponentVisualFamily.data.primary}
    var body:some View {
        VStack(alignment:.leading,spacing:14){
            if section=="Rules" {
                InspectorSection("Turn & legality",symbol:"arrow.triangle.branch",identifier:"game.legal",initiallyExpanded:true,accent:accent){
                    entry("Player to move · must be 0 or 1",binding:scalar("currentPlayer"))
                    entry("Legal action · Boolean predicate",binding:scalar("legal"))
                }
                Text("Transitions").font(.headline)
                Text("Exactly one guard must match each legal action. Updates are atomic. next.field references form explicit dependencies; ambiguous guards or cyclic updates are refused.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(0..<array("transitions").count,id:\.self){i in
                    InspectorSection("Transition \(i+1)",symbol:"arrow.right",identifier:"game.transition.\(i)",initiallyExpanded:true,accent:accent){
                        entry("Name",binding:item("transitions",i,"name"));entry("When · Boolean guard",binding:item("transitions",i,"when"))
                        ForEach(updates(i).keys.sorted(),id:\.self){field in
                            HStack {entry("Update "+field,binding:update(i,field));Button("Remove"){var v=updates(i);v.removeValue(forKey:field);setUpdates(i,v)}.buttonStyle(FlatButtonStyle())}
                        }
                        FlatChoiceControl(title:"Add state update",selection:"",options:stateFields.map{.init(id:$0,title:$0)}){field in var v=updates(i);v[field] = .string("state."+field);setUpdates(i,v)}
                        Button("Remove transition"){remove("transitions",i)}.buttonStyle(FlatButtonStyle())
                    }
                }
                Button("Add transition"){append("transitions",.object(["name":.string("New transition"),"when":.string("True"),"updates":.object([:])]))}.buttonStyle(FlatButtonStyle())
                Text("Terminal outcomes").font(.headline)
                Text("At most one condition may match. Use winner null for a draw, or player 0/1. Define no-legal-action handling explicitly in your terminal rules.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(0..<array("terminal").count,id:\.self){i in
                    InspectorSection("Outcome \(i+1)",symbol:"flag",identifier:"game.terminal.\(i)",initiallyExpanded:true,accent:accent){entry("Name",binding:item("terminal",i,"name"));entry("When · Boolean condition",binding:item("terminal",i,"when"));entry("Winner · player expression or None",binding:item("terminal",i,"winner"));Button("Remove outcome"){remove("terminal",i)}.buttonStyle(FlatButtonStyle())}
                }
                Button("Add outcome"){append("terminal",.object(["name":.string("New outcome"),"when":.string("False"),"winner":.string("None")]))}.buttonStyle(FlatButtonStyle())
                InspectorSection("Reward & invariants",symbol:"equal.square",identifier:"game.reward",initiallyExpanded:true,accent:accent){entry("Reward · player perspective · −1…1",binding:scalar("reward"));expressions("invariants",label:"Invariant");Button("Add invariant"){append("invariants",.string("True"))}.buttonStyle(FlatButtonStyle())}
            } else {
                Text("Observation mapping").font(.headline)
                Text("Each expression produces one finite numeric value. player is the requested player (0 or 1); no board dimensions are assumed. These mappings are testable contracts for a future model consumer, not training execution.").font(.caption).foregroundStyle(MLTheme.muted)
                expressions("observations",label:"Value")
                Button("Add observation value"){append("observations",.string("0"))}.buttonStyle(FlatButtonStyle())
            }
            Text("Syntax colors are a reading aid; Validate checks whether a rule is allowed. Expressions: state.field · action.field · params.field · actor · player. Functions: range, len, sum, min, max, all, any, abs, set_at, repeat. Bounded list comprehensions are supported; imports, attributes on values and arbitrary calls are not. Source exposes the same definition for advanced editing.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private var stateFields:[String]{if case .object(let v)?=draft["state"]{return v.keys.sorted()};return []}
    private func array(_ key:String)->[JSONValue]{if case .array(let a)?=draft[key]{return a};return []}
    private func append(_ key:String,_ v:JSONValue){var a=array(key);a.append(v);draft[key] = .array(a)}
    private func remove(_ key:String,_ i:Int){var a=array(key);guard a.indices.contains(i) else{return};a.remove(at:i);draft[key] = .array(a)}
    private func scalar(_ key:String)->Binding<String>{Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string(String($0.prefix(4096)))})}
    private func item(_ key:String,_ i:Int,_ field:String)->Binding<String>{Binding(get:{let a=array(key);guard a.indices.contains(i),case .object(let v)=a[i] else{return ""};return v[field]?.stringValue ?? ""},set:{var a=array(key);guard a.indices.contains(i),case .object(var v)=a[i] else{return};v[field] = .string(String($0.prefix(4096)));a[i] = .object(v);draft[key] = .array(a)})}
    private func updates(_ i:Int)->[String:JSONValue]{let a=array("transitions");guard a.indices.contains(i),case .object(let t)=a[i],case .object(let v)?=t["updates"] else{return [:]};return v}
    private func setUpdates(_ i:Int,_ v:[String:JSONValue]){var a=array("transitions");guard a.indices.contains(i),case .object(var t)=a[i] else{return};t["updates"] = .object(v);a[i] = .object(t);draft["transitions"] = .array(a)}
    private func update(_ i:Int,_ field:String)->Binding<String>{Binding(get:{updates(i)[field]?.stringValue ?? ""},set:{var v=updates(i);v[field] = .string(String($0.prefix(4096)));setUpdates(i,v)})}
    private func entry(_ title:String,binding:Binding<String>)->some View{
        VStack(alignment:.leading,spacing:6){
            Text(title).font(.caption).foregroundStyle(MLTheme.muted)
            if title=="Name" {TextField(title,text:binding).flatField()}
            else {GameRuleCodeEditor(text:binding,language:.expression,label:title,limit:4096).frame(height:expressionHeight(binding.wrappedValue)).overlay(Rectangle().stroke(MLTheme.muted.opacity(0.35)))}
        }
    }
    private func expressionHeight(_ value:String)->CGFloat {min(136,max(44,CGFloat(value.split(separator:"\n",omittingEmptySubsequences:false).count + value.count/90)*16+24))}
    private func expressions(_ key:String,label:String)->some View{ForEach(0..<array(key).count,id:\.self){i in HStack{entry(label+" \(i+1)",binding:Binding(get:{let a=array(key);return a.indices.contains(i) ? a[i].stringValue ?? "":""},set:{var a=array(key);guard a.indices.contains(i) else{return};a[i] = .string(String($0.prefix(4096)));draft[key] = .array(a)}));Button("Remove"){remove(key,i)}.buttonStyle(FlatButtonStyle())}}}
}
