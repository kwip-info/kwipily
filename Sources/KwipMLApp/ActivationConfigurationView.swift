import SwiftUI
import KwipMLCore

/// One draft over existing activation contracts. Apply changes the actual primitive atomically.
struct ActivationConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var allowedFunctions:[ActivationConfiguration.Function]=ActivationConfiguration.Function.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(ActivationConfiguration.Function,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var function:ActivationConfiguration.Function = .relu
    @State private var values:[String:JSONValue]=[:]
    @State private var baselineType=""
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var pendingNumbers=Set<String>()
    @State private var generation=0
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.activations.primary:ComponentVisualFamily.activations.secondary}
    private var fields:[ComponentField] {registry.component(type:function.rawValue)?.fields ?? []}
    private var hasSlopeInitializer:Bool {
        guard case .object(let settings)=node.parameterSettings?["weight"] else {return false}
        return settings["initialization"] != nil
    }
    private var changed:Bool {function.rawValue != baselineType || values != baselineValues}
    private var pending:Bool {changed || !pendingNumbers.isEmpty}
    private var origin:String {node.id+".activation."+String(generation)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Tensor · in → out",systemImage:"circle.hexagongrid")
                .font(.caption).foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading)
                .padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            choice("Function",value:function.rawValue,options:allowedFunctions.map {.init(id:$0.rawValue,title:$0.title)}) {id in
                guard let next=ActivationConfiguration.Function(rawValue:id),next != function else {return}
                do {values=try ActivationConfiguration.defaults(for:next,registry:registry);function=next;customAxis=false;pendingNumbers=[];generation+=1;rejection=nil}
                catch {rejection=error.localizedDescription}
            }.disabled(isBusy || !pendingNumbers.isEmpty || !boundFields.isEmpty)
            if function == .gelu {
                choice("Approximation",value:values["approximate_tanh"]?.boolValue == true ? "tanh":"exact",
                       options:[.init(id:"exact",title:"Exact"),.init(id:"tanh",title:"Tanh approximation")]) {values["approximate_tanh"] = .bool($0=="tanh")}
                    .disabled(isBusy || boundFields.contains("approximate_tanh"))
            } else if function == .softmax || function == .logSoftmax {
                choice("Axis",value:axisChoice,options:[.init(id:"-1",title:"Last axis (−1)"),.init(id:"1",title:"First sample axis (1)"),.init(id:"custom",title:"Custom index")]) {id in
                    if id=="custom" {customAxis=true} else {values["dim"] = .integer(Int64(id) ?? -1);customAxis=false}
                }.disabled(isBusy || boundFields.contains("dim"))
                if customAxis || axisChoice=="custom",let field=fields.first(where:{$0.key=="dim"}) {fieldEditor(field)}
            } else {
                ForEach(fields,id:\.key) {field in fieldEditor(field)}
            }
            if baselineType=="prelu",function.rawValue != baselineType,node.parameterSettings?.isEmpty == false {
                Label("Changing function removes its learned-slope overrides.",systemImage:"exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange).fixedSize(horizontal:false,vertical:true)
            }
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent)
                    .background(accent.opacity(0.10)).disabled(isBusy || !changed || !pendingNumbers.isEmpty)
                    .accessibilityIdentifier("activation.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !pending)
                    .accessibilityIdentifier("activation.revert")
                Spacer(minLength:0)
                if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Activation changes")}
            }
            if let rejection {Text(rejection).font(.caption).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
        }.font(.caption).tint(accent)
            .onAppear {reload()}
            .onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.type != baselineType || next.config != baselineSource else {return}
                if !pending {reload()}
                else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }
            .onChange(of:pending) {_,value in onPending(value)}
            .onDisappear {onPending(false)}
    }
    @State private var customAxis=false
    private var axisChoice:String {
        let index=values["dim"]?.integerValue ?? -1
        return customAxis ? "custom":index == -1 ? "-1":index == 1 ? "1":"custom"
    }
    private func choice(_ title:String,value:String,options:[FlatChoiceOption],select:@escaping(String)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title)
            FlatChoiceControl(title:title,selection:value,options:options,accent:accent,onSelect:select)
                .accessibilityIdentifier("activation."+title.lowercased())
        }
    }
    private func fieldEditor(_ field:ComponentField)->some View {
        let token=origin+"."+field.key
        return RegistryFieldControl(field:field,value:values[field.key] ?? field.defaultValue,origin:token,
            currentOrigin:{isBusy ? nil:origin+"."+field.key},onDraftChange:{id,dirty in
                guard id==origin+"."+field.key else {return}
                if dirty {pendingNumbers.insert(field.key)} else {pendingNumbers.remove(field.key)}
            },componentID:function.rawValue,configuration:values) {value in
                guard !isBusy,!boundFields.contains(field.key) else {return false}
                values[field.key]=value;rejection=nil;return true
            }.disabled(isBusy || boundFields.contains(field.key) || (field.key=="init" && hasSlopeInitializer))
            .help(boundFields.contains(field.key) ? "Controlled by the containing block’s exposed settings.":field.key=="init" && hasSlopeInitializer ? "Controlled by the learned-slope starting-values override.":field.title)
    }
    private func reload() {
        baselineType=node.type;baselineSource=node.config
        function=ActivationConfiguration.Function(rawValue:node.type) ?? .relu
        do {values=try ActivationConfiguration(componentID:node.type,configurationVersion:configurationVersion,values:node.config,registry:registry).values;rejection=nil}
        catch {values=node.config;rejection=error.localizedDescription}
        baselineValues=values;pendingNumbers=[];customAxis=false;generation+=1;onPending(false)
    }
    private func apply() {
        guard !isBusy,pendingNumbers.isEmpty,allowedFunctions.contains(function) else {return}
        do {
            let checked=try ActivationConfiguration(componentID:function.rawValue,configurationVersion:2,values:values,registry:registry)
            guard onApply(function,checked.values,baselineType,baselineSource) else {rejection="Activation settings were not applied. Check the current selection and containing block’s bindings.";return}
            baselineType=function.rawValue;baselineSource=checked.values;baselineValues=checked.values;values=checked.values
            rejection=nil;generation+=1;onPending(false)
        } catch {rejection=error.localizedDescription}
    }
}
