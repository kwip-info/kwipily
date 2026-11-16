import SwiftUI
import KwipMLCore

/// Unpublished strings stay local until the entire sequence configuration validates.
struct RecurrentInspectorDraft:Equatable {
    var method:RecurrentSequenceConfiguration.Method
    var values:[String:JSONValue]
    var isCell:Bool=false
    var numeric:[String:String]=[:]
    var remembered:[String:JSONValue]=[:]
    func text(_ key:String)->String {
        if let value=numeric[key] {return value}
        if key=="dropout" {return values[key]?.doubleValue.map {String($0)} ?? ""}
        return values[key]?.integerValue.map(String.init) ?? ""
    }
    mutating func setText(_ text:String,key:String) {numeric[key]=text}
    mutating func select(_ next:RecurrentSequenceConfiguration.Method,registry:ComponentRegistry)throws {
        guard next != method else {return}
        let current=try configuration(registry:registry)
        var defaults=try RecurrentAuthoringConfiguration.defaults(for:next,isCell:isCell,registry:registry)
        for key in ["input_size","hidden_size","num_layers","bias","bidirectional","dropout","state_mode"] where defaults[key] != nil {defaults[key]=current[key]}
        var cache=remembered
        for key in ["nonlinearity","proj_size"] {if let value=current[key] {cache[key]=value}}
        for key in ["nonlinearity","proj_size"] where defaults[key] != nil {if let value=cache[key] {defaults[key]=value}}
        self = .init(method:next,values:defaults,isCell:isCell,remembered:cache)
    }
    func configuration(registry:ComponentRegistry)throws->[String:JSONValue] {
        var result=values
        for (key,text) in numeric {
            let cleaned=text.trimmingCharacters(in:.whitespacesAndNewlines)
            if key=="dropout" {
                guard let value=Double(cleaned),value.isFinite else {throw GraphConfigurationError("Enter a finite dropout probability from 0 to 1.")}
                result[key] = .number(value)
            } else {
                guard ["input_size","hidden_size","num_layers","proj_size"].contains(key),let value=Int64(cleaned) else {throw GraphConfigurationError("Enter a whole number for the feature or layer count.")}
                result[key] = .integer(value)
            }
        }
        return try RecurrentAuthoringConfiguration(componentID:RecurrentAuthoringConfiguration.primitiveID(method,isCell:isCell),configurationVersion:2,values:result,registry:registry).values
    }
}

struct RecurrentConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var allowedMethods:[RecurrentSequenceConfiguration.Method]=RecurrentSequenceConfiguration.Method.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(RecurrentSequenceConfiguration.Method,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=RecurrentInspectorDraft(method:.rnn,values:[:])
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baselineType=""
    @State private var rejection:String?
    private var isCell:Bool {node.type.hasSuffix("_cell")}
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.layers.primary:ComponentVisualFamily.layers.secondary}
    private var candidate:[String:JSONValue]? {try? draft.configuration(registry:registry)}
    private var changed:Bool {RecurrentAuthoringConfiguration.primitiveID(draft.method,isCell:isCell) != baselineType || (candidate.map {$0 != baselineValues} ?? !draft.numeric.isEmpty)}
    private var portInputs:[String] {draft.values["state_mode"] == .string("supplied") ? (draft.method == .lstm ? ["in","hidden","cell"]:["in","hidden"]):["in"]}
    private var portOutputs:[String] {draft.method == .lstm ? ["out","hidden","cell"]:["out","hidden"]}
    private var selectorBusy:Bool {isBusy || !boundFields.isEmpty || (!draft.numeric.isEmpty && candidate==nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            choice("Method",value:draft.method.rawValue,options:allowedMethods.map {.init(id:$0.rawValue,title:$0.title+(isCell ? " Cell":""))}) {value in
                guard let next=RecurrentSequenceConfiguration.Method(rawValue:value) else {return}
                do {try draft.select(next,registry:registry);rejection=nil} catch {rejection=error.localizedDescription}
            }.disabled(selectorBusy)
            VStack(alignment:.leading,spacing:3) {
                Text("Inputs · "+portInputs.joined(separator:", "))
                Text("Outputs · "+portOutputs.joined(separator:", "))
            }.foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading).padding(9)
                .background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            number("Input features",key:"input_size")
            number("Hidden features",key:"hidden_size")
            if !isCell {number("Layers",key:"num_layers")}
            if draft.method == .rnn {
                choice("RNN activation",value:draft.values["nonlinearity"]?.stringValue ?? "tanh",options:[.init(id:"tanh",title:"Tanh"),.init(id:"relu",title:"ReLU")]) {draft.values["nonlinearity"] = .string($0);rejection=nil}
                    .disabled(isBusy || boundFields.contains("nonlinearity"))
            }
            choice("Starting state",value:draft.values["state_mode"]?.stringValue ?? "zero",options:[.init(id:"zero",title:"Start from zeros"),.init(id:"supplied",title:"Use state connections")]) {draft.values["state_mode"] = .string($0);rejection=nil}
                .disabled(isBusy || boundFields.contains("state_mode"))
            flag("Use bias",key:"bias")
            if !isCell {
                flag("Bidirectional",key:"bidirectional")
                number("Between-layer dropout",key:"dropout").disabled(Int(draft.text("num_layers"))==1 && Double(draft.text("dropout"))==0)
            }
            if !isCell && draft.method == .lstm {
                number("Projection features",key:"proj_size")
                if (Int(draft.text("proj_size")) ?? 0)>0 {Label("Projection requires CPU",systemImage:"exclamationmark.triangle").foregroundStyle(.orange)}
            }
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("recurrent.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("recurrent.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Recurrent changes")}
            }
            if let rejection {Text(rejection).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            else if changed,candidate==nil {Text(validationMessage).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
        }.font(.caption).tint(accent)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.type != baselineType || next.config != baselineSource else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    private var validationMessage:String {do {_=try draft.configuration(registry:registry);return ""} catch {return error.localizedDescription}}
    private func choice(_ title:String,value:String,options:[FlatChoiceOption],select:@escaping(String)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value,options:options,accent:accent,onSelect:select).accessibilityIdentifier("recurrent."+title.lowercased())}
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil}))
            .toggleStyle(.checkbox).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("recurrent."+key)
    }
    private func number(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);numericCell(title,key:key)}
    }
    private func numericCell(_ title:String,key:String)->some View {
        TextField(title,text:Binding(get:{draft.text(key)},set:{draft.setText($0,key:key);rejection=nil}))
            .flatField().frame(minWidth:35,maxWidth:.infinity).disabled(isBusy || boundFields.contains(key)).accessibilityLabel(title)
    }
    private func reload() {
        baselineType=node.type;baselineSource=node.config
        let method=RecurrentAuthoringConfiguration.method(for:node.type) ?? .rnn
        do {let value=try RecurrentAuthoringConfiguration(componentID:node.type,configurationVersion:configurationVersion,values:node.config,registry:registry);draft = .init(method:value.method,values:value.values,isCell:value.isCell);rejection=nil}
        catch {draft = .init(method:method,values:node.config,isCell:isCell);rejection=error.localizedDescription}
        baselineValues=draft.values;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,allowedMethods.contains(draft.method),let values=candidate else {return}
        guard onApply(draft.method,values,baselineType,baselineSource) else {rejection="Recurrent settings were not applied. Check connected or exposed state ports, named parameter policies and Block bindings.";return}
        baselineType=RecurrentAuthoringConfiguration.primitiveID(draft.method,isCell:isCell);baselineSource=values;baselineValues=values;draft.values=values;draft.numeric=[:];rejection=nil;onPending(false)
    }
}
