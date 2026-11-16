import SwiftUI
import KwipMLCore

struct DecisionConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @State private var editor:ExpandedConfigurationWindowController?
    var body:some View {
        VStack(alignment:.leading,spacing:12){
            Text("Typed decisions · verified Model resource").font(.caption.weight(.semibold))
            Text("Choice, ordinal score and yes/no over named table columns. Decisions and all option Scores use ordinary table outputs.").font(.caption).foregroundStyle(MLTheme.muted)
            Button("Configure decisions…"){
                if let editor,!editor.didClose {editor.present()}else {
                    let next=ExpandedConfigurationWindowController(title:"Decide",identifier:"info.kwip.KwipML.decision-editor"){controller in AnyView(DecisionEditorView(context:context,node:node,requestClose:{[weak controller] in controller?.close()},registerDiscard:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty}).windowActivatingControls().appAppearance(context.decisionAppearance))};editor=next;next.present()
                }
            }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
            Text("Apply saves setup. Run Flow scores the connected records. Model downloads and training remain explicit operations.").font(.caption).foregroundStyle(MLTheme.muted)
        }.onDisappear{editor?.close();editor=nil}
    }
}

private struct DecisionEditorView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    let requestClose:()->Void
    let registerDiscard:(@escaping ()->Void,@escaping ()->Bool)->Void
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var section="Data"
    @State private var error=""
    @State private var closing=false
    @State private var generation=UUID()
    private var pending:Bool{draft != original}
    private var questions:[String:JSONValue]{if case .object(let q)=draft["questionSet"]{return q};return TypedDecisionConfiguration.defaultQuestionSet}
    private var columns:[String]{if case .array(let a)=draft["stateColumns"]{return a.compactMap(\.stringValue)};return []}
    var body:some View {
        VStack(spacing:0){
            HStack{VStack(alignment:.leading,spacing:4){Text("Decide").font(.title2.weight(.semibold));Text(context.authoredName).font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close",action:close).buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider()
            HStack(spacing:0){
                VStack(alignment:.leading,spacing:5){ForEach(["Data","Questions","Resources"],id:\.self){name in Button{section=name}label:{Text(name).frame(maxWidth:.infinity,alignment:.leading).padding(10).background(section==name ? MLTheme.rose.opacity(0.16):Color.clear).contentShape(Rectangle())}.buttonStyle(.plain)};Spacer()}.padding(12).frame(width:200)
                Divider()
                ScrollView{VStack(alignment:.leading,spacing:18){Text(section).font(.title3.weight(.semibold));content}.padding(24).frame(maxWidth:.infinity,alignment:.leading)}
            }
            Divider()
            VStack(alignment:.leading,spacing:8){if !error.isEmpty{Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)};HStack{Button("Revert",action:load).buttonStyle(FlatButtonStyle()).disabled(!pending);Text("Apply saves configuration. Run Flow produces complete standard tables.").font(.caption).foregroundStyle(MLTheme.muted);Spacer();Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending || context.isBusy)}}.padding(16)
        }.frame(minWidth:800,idealWidth:1000,minHeight:610,idealHeight:760).background(MLTheme.background)
        .onAppear{load();registerDiscard({discard()}, {!closing && (pending)})}.onChange(of:draft){_,_ in report()}.onDisappear{discard()}
    }
    @ViewBuilder private var content:some View {
        switch section {
        case "Data":
            TypedDecisionGroup("Connected state") {
                Text("Table · "+(context.inputSourceName(forInput:"table") ?? "Connect a table")).font(.caption)
                Text("Each selected source column keeps its name in the model's structured state. Missing required values and oversized states are refused; data is never silently truncated.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(Array(columns.indices),id:\.self){index in
                    HStack{columnChoice("State field \(index+1)",value:Binding(get:{columns.indices.contains(index) ? columns[index]:""},set:{name in var names=columns;guard names.indices.contains(index)else{return};names[index]=name;draft["stateColumns"] = .array(names.map(JSONValue.string))}));Button("Remove"){var names=columns;names.remove(at:index);draft["stateColumns"] = .array(names.map(JSONValue.string))}.buttonStyle(FlatButtonStyle()).disabled(columns.count<=1)}
                }
                Button("Add state field"){let choices=context.upstreamSchema.filter{["string","int64","float32","float64","bool"].contains($0.dtype) && !columns.contains($0.name)};draft["stateColumns"] = .array((columns+[choices.first?.name ?? "field\(columns.count+1)"]).map(JSONValue.string))}.buttonStyle(FlatButtonStyle()).disabled(columns.count>=16)
                columnChoice("Unique source key · blank uses captured row identity",value:Binding(get:{draft["identityColumn"]?.stringValue ?? ""},set:{draft["identityColumn"] = .string($0)}),allowBlank:true)
            }
        case "Questions":
            TypedDecisionGroup("Question source") {
                Toggle("Use the connected trained model's question set",isOn:Binding(get:{draft["questionsFromModel"] == .bool(true)},set:{draft["questionsFromModel"] = .bool($0)})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))
                if draft["questionsFromModel"] == .bool(true) {
                    Text(context.connectedDecisionQuestions?["name"]?.stringValue ?? "No question set on the connected model. Choose a project set instead.").font(.caption).foregroundStyle(MLTheme.muted)
                    if let q=context.connectedDecisionQuestions {Button("Copy this revision into the draft"){draft["questionSet"] = .object(q);draft["questionsFromModel"] = false}.buttonStyle(FlatButtonStyle())}
                } else {
                    ForEach(Array(context.decisionQuestionResources.enumerated()),id:\.offset){_,choice in Button("Use "+choice.0){draft["questionSet"] = .object(choice.1)}.buttonStyle(FlatButtonStyle())}
                    Text("Reusing a revision copies its immutable content. Editing this draft creates a new revision; other tools and recorded runs keep their selected version.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            if draft["questionsFromModel"] != .bool(true) {DecisionQuestionEditor(value:Binding(get:{questions},set:{draft["questionSet"] = .object($0)}))}
        default:
            TypedDecisionGroup("Qualified execution") {Text("CPU reference · offline after explicit model acquisition");number("Records per CPU batch ·1–8","batchRows");number("Maximum complete input records ·1–10,000","maximumRows");number("Maximum complete score rows ·1–200,000","maximumScoreRows")}

        }
    }
    private func columnChoice(_ title:String,value:Binding<String>,allowBlank:Bool=false)->some View {VStack(alignment:.leading,spacing:5){Text(title).font(.caption).foregroundStyle(MLTheme.muted);let names=context.upstreamSchema.filter{["string","int64","float32","float64","bool"].contains($0.dtype)}.map(\.name);if names.isEmpty{TextField(title,text:value).flatField()}else{FlatChoiceControl(title:title,selection:value.wrappedValue,options:(allowBlank ? [FlatChoiceOption(id:"",title:"Captured observation identity")]:[])+names.map{.init(id:$0,title:$0)}){value.wrappedValue=$0}}}}
    private func number(_ title:String,_ key:String)->some View{VStack(alignment:.leading,spacing:5){Text(title).font(.caption);FileInputDraftEntry(title:title,value:Binding(get:{draft[key] ?? 1},set:{draft[key]=$0}),parse:{Int64($0).map(JSONValue.integer) ?? .string($0)}).id(generation.uuidString+key)}}
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"typedDecision"),key:"typedDecision",pending:!closing && pending)}
    private func load(){guard context.isCurrent else{return};draft=context.node?.config ?? node.config;original=draft;generation=UUID();error="";report()}
    private func apply(){do{var next=draft;if case .object(let old)=original["questionSet"]{next["questionSet"] = .object(DecisionQuestionResource.advanced(questions,from:old))};try TypedDecisionConfiguration.validate(next);guard context.applyTableConfiguration(next,expected:original)else{error=context.statusMessage;return};closing=true;context.setNumericPending(origin:context.numericOrigin(key:"typedDecision"),key:"typedDecision",pending:false);requestClose()}catch{self.error=error.localizedDescription}}
    private func discard(){guard !closing else{return};closing=true;context.setNumericPending(origin:context.numericOrigin(key:"typedDecision"),key:"typedDecision",pending:false)}
    private func close(){requestClose()}
}
