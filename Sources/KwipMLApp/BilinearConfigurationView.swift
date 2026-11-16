import SwiftUI
import KwipMLCore

/// Unpublished strings stay local until the entire Bilinear configuration validates.
struct BilinearInspectorDraft:Equatable {
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    func text(_ key:String)->String {numeric[key] ?? values[key]?.integerValue.map(String.init) ?? ""}
    mutating func setText(_ text:String,key:String) {numeric[key]=text}
    func configuration()throws->[String:JSONValue] {
        var result=values
        for (key,text) in numeric {
            guard ["in1_features","in2_features","out_features"].contains(key),let value=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {
                throw GraphConfigurationError("Enter a whole number for the feature width.")
            }
            result[key] = .integer(value)
        }
        return try BilinearConfiguration(values:result).values
    }
}

struct BilinearConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:([String:JSONValue],[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=BilinearInspectorDraft(values:[:])
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.layers.primary:ComponentVisualFamily.layers.secondary}
    private var candidate:[String:JSONValue]? {try? draft.configuration()}
    private var changed:Bool {(candidate.map {$0 != baselineValues} ?? !draft.numeric.isEmpty)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Tensor a · Tensor b → out",systemImage:"circle.hexagongrid").foregroundStyle(accent)
                .frame(maxWidth:.infinity,alignment:.leading).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            number("First input features · a",key:"in1_features")
            number("Second input features · b",key:"in2_features")
            number("Output features",key:"out_features")
            flag("Use bias",key:"bias")
            if let values=candidate,let parsed=try? BilinearConfiguration(values:values) {
                Text("\(parsed.parameterCount.formatted()) parameters").foregroundStyle(accent).accessibilityIdentifier("bilinear.parameters")
            }
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("bilinear.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("bilinear.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Bilinear changes")}
            }
            if let rejection {Text(rejection).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            else if changed,candidate==nil {Text(validationMessage).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
        }.font(.caption).tint(accent)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.config != baselineSource else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    private var validationMessage:String {do {_=try draft.configuration();return ""} catch {return error.localizedDescription}}
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil}))
            .toggleStyle(.checkbox).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("bilinear."+key)
    }
    private func number(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);numericCell(title,key:key)}
    }
    private func numericCell(_ title:String,key:String)->some View {
        TextField(title,text:Binding(get:{draft.text(key)},set:{draft.setText($0,key:key);rejection=nil}))
            .flatField().frame(minWidth:35,maxWidth:.infinity).disabled(isBusy || boundFields.contains(key)).accessibilityLabel(title)
    }
    private func reload() {
        baselineSource=node.config
        do {let value=try BilinearConfiguration(values:node.config);draft = .init(values:value.values);rejection=nil}
        catch {draft = .init(values:node.config);rejection=error.localizedDescription}
        baselineValues=draft.values;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,let values=candidate else {return}
        guard onApply(values,baselineSource) else {rejection="Bilinear settings were not applied. Check named parameter policies and the containing block’s bindings.";return}
        baselineSource=values;baselineValues=values;draft.values=values;draft.numeric=[:];rejection=nil;onPending(false)
    }
}
