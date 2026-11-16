import SwiftUI
import KwipMLCore

struct EmbeddingInspectorDraft:Equatable {
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    var paddingEnabled:Bool
    var normEnabled:Bool
    init(values:[String:JSONValue]) {self.values=values;paddingEnabled=values["padding_idx"] != .null;normEnabled=values["max_norm"] != .null}
    func text(_ key:String)->String {
        if let text=numeric[key] {return text}
        if let n=values[key]?.integerValue {return String(n)}
        if let n=values[key]?.doubleValue {return String(n)}
        return key=="padding_idx" ? "0":key=="max_norm" ? "1":""
    }
    mutating func setText(_ text:String,key:String) {numeric[key]=text}
    func configuration(registry:ComponentRegistry)throws->[String:JSONValue] {
        var result=values
        for key in ["num_embeddings","embedding_dim","padding_idx","max_norm","norm_type"] {
            if key=="padding_idx",!paddingEnabled {result[key] = .null;continue}
            if key=="max_norm",!normEnabled {result[key] = .null;continue}
            let text=text(key).trimmingCharacters(in:.whitespacesAndNewlines)
            if ["num_embeddings","embedding_dim","padding_idx"].contains(key) {
                guard let n=Int64(text) else {throw GraphConfigurationError("Enter a whole number for table dimensions and padding row.")};result[key] = .integer(n)
            } else {
                guard let n=Double(text),n.isFinite,n>0 else {throw GraphConfigurationError("Norm values must be positive finite numbers.")};result[key] = .number(n)
            }
        }
        return try EmbeddingConfiguration.validatedAuthoring(result,registry:registry)
    }
}

struct EmbeddingConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:([String:JSONValue],[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=EmbeddingInspectorDraft(values:[:])
    @State private var baseline:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.embeddings.primary:ComponentVisualFamily.embeddings.secondary}
    private var candidate:[String:JSONValue]? {try? draft.configuration(registry:registry)}
    private var changed:Bool {candidate.map {$0 != baseline} ?? (draft.values != baseline || !draft.numeric.isEmpty || draft.paddingEnabled != (baseline["padding_idx"] != .null) || draft.normEnabled != (baseline["max_norm"] != .null))}
    private var weightOverride:String? {if case .object(let row)=node.parameterSettings?["weight"],let mode=row["trainability"]?.stringValue,mode != "inherit" {return mode};return nil}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            number("Table rows",key:"num_embeddings")
            number("Vector dimensions",key:"embedding_dim")
            if let values=candidate,let config=try? EmbeddingConfiguration(values:values) {
                Label("\(config.parameterCount.formatted()) weights",systemImage:"tablecells").foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading).padding(8).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.25)))
            }
            Toggle("Use padding row",isOn:$draft.paddingEnabled).disabled(isBusy || boundFields.contains("padding_idx")).accessibilityIdentifier("embedding.padding")
            if draft.paddingEnabled {number("Padding row",key:"padding_idx")}
            Toggle("Limit vector norm",isOn:$draft.normEnabled).disabled(isBusy || boundFields.contains("max_norm")).accessibilityIdentifier("embedding.limit-norm")
            if draft.normEnabled {number("Maximum norm",key:"max_norm");number("Norm order",key:"norm_type")}
            Text("Gradients").fontWeight(.semibold).foregroundStyle(accent)
            flag("Scale gradients by frequency",key:"scale_grad_by_freq")
            flag("Sparse gradients",key:"sparse")
            flag("Freeze learned weights",key:"freeze").disabled(weightOverride != nil)
            if let mode=weightOverride {Label("Weights · \(mode.capitalized) override",systemImage:"slider.horizontal.3").foregroundStyle(accent).help("Change Trainability in Weights to Layer default before changing Freeze learned weights.")}
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("embedding.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("embedding.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Embedding changes")}
            }
            if let rejection {Text(rejection).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            else if changed,candidate==nil {Text(validationMessage).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
        }.font(.caption).tint(accent).toggleStyle(.checkbox)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.config != original else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    private var validationMessage:String {do {_=try draft.configuration(registry:registry);return ""} catch {return error.localizedDescription}}
    private func number(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title)
            TextField(title,text:Binding(get:{draft.text(key)},set:{draft.setText($0,key:key);rejection=nil})).flatField().frame(maxWidth:.infinity).disabled(isBusy || boundFields.contains(key)).accessibilityLabel(title)
        }
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil})).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("embedding."+key)
    }
    private func reload() {
        original=node.config
        let values=(try? GraphMigration.effectiveConfiguration(of:node,schemaVersion:configurationVersion,registry:registry)) ?? node.config
        baseline=(try? EmbeddingConfiguration.validatedAuthoring(values,registry:registry)) ?? values;draft = .init(values:values);rejection=nil;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,let values=candidate else {return}
        guard onApply(values,original) else {rejection="Embedding settings were not applied. Check selection, saved settings and containing block bindings.";return}
        baseline=values;original=values;draft = .init(values:values);rejection=nil;onPending(false)
    }
}
