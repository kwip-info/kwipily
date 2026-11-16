import SwiftUI
import KwipMLCore

/// Unpublished strings stay local until the entire affine configuration validates.
struct LinearInspectorDraft:Equatable {
    var variant:LinearConfiguration.Variant
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    var explicitWidth:JSONValue? = nil
    func text(_ key:String)->String {
        numeric[key] ?? values[key]?.integerValue.map(String.init) ?? ""
    }
    mutating func setText(_ text:String,key:String) {numeric[key]=text}
    mutating func select(_ next:LinearConfiguration.Variant,registry:ComponentRegistry)throws {
        guard next != variant else {return}
        let current=try configuration(registry:registry)
        var defaults=try LinearConfiguration.defaults(for:next,registry:registry)
        for key in ["out_features","bias"] {defaults[key]=current[key]}
        let width=current["in_features"] ?? explicitWidth
        if next == .explicit,let width {defaults["in_features"]=width}
        self = .init(variant:next,values:defaults,explicitWidth:width)
    }
    func configuration(registry:ComponentRegistry)throws->[String:JSONValue] {
        var result=values
        for (key,text) in numeric {
            guard ["in_features","out_features"].contains(key),let value=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {
                throw GraphConfigurationError("Enter a whole number for the feature width.")
            }
            result[key] = .integer(value)
        }
        return try LinearConfiguration(componentID:variant.rawValue,configurationVersion:2,values:result,registry:registry).values
    }
}

struct LinearConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var allowedVariants:[LinearConfiguration.Variant]=LinearConfiguration.Variant.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(LinearConfiguration.Variant,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=LinearInspectorDraft(variant:.explicit,values:[:])
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baselineType=""
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.layers.primary:ComponentVisualFamily.layers.secondary}
    private var candidate:[String:JSONValue]? {try? draft.configuration(registry:registry)}
    private var changed:Bool {draft.variant.rawValue != baselineType || (candidate.map {$0 != baselineValues} ?? !draft.numeric.isEmpty)}
    private var selectorBusy:Bool {isBusy || !boundFields.isEmpty || (!draft.numeric.isEmpty && candidate==nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Tensor · in → out",systemImage:"circle.hexagongrid").foregroundStyle(accent)
                .frame(maxWidth:.infinity,alignment:.leading).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            choice("Input sizing",value:draft.variant.rawValue,options:allowedVariants.map {.init(id:$0.rawValue,title:$0.title)}) {value in
                guard let next=LinearConfiguration.Variant(rawValue:value) else {return}
                do {try draft.select(next,registry:registry);rejection=nil} catch {rejection=error.localizedDescription}
            }.disabled(selectorBusy)
            if draft.variant == .explicit {number("Input features",key:"in_features")}
            else {Label("Inferred width · Lazy Linear",systemImage:"arrow.left.and.right").foregroundStyle(accent)}
            number("Output features",key:"out_features")
            flag("Use bias",key:"bias")
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("linear.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("linear.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Linear changes")}
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
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value,options:options,accent:accent,onSelect:select).accessibilityIdentifier("linear."+title.lowercased())}
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil}))
            .toggleStyle(.checkbox).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("linear."+key)
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
        let variant=LinearConfiguration.Variant(rawValue:node.type) ?? .explicit
        do {let value=try LinearConfiguration(componentID:node.type,configurationVersion:configurationVersion,values:node.config,registry:registry);draft = .init(variant:value.variant,values:value.values);rejection=nil}
        catch {draft = .init(variant:variant,values:node.config);rejection=error.localizedDescription}
        baselineValues=draft.values;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,allowedVariants.contains(draft.variant),let values=candidate else {return}
        guard onApply(draft.variant,values,baselineType,baselineSource) else {rejection="Linear settings were not applied. Check parameter overrides, connected outputs and the containing block’s bindings.";return}
        baselineType=draft.variant.rawValue;baselineSource=values;baselineValues=values;draft.values=values;draft.numeric=[:];rejection=nil;onPending(false)
    }
}
