import SwiftUI
import KwipMLCore

/// Unpublished strings stay local until the entire per-axis configuration validates.
struct ConvolutionInspectorDraft:Equatable {
    var variant:ConvolutionConfiguration.Variant
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    func text(_ key:String,_ axis:Int?=nil)->String {
        let token=key+(axis.map {".\($0)"} ?? "")
        if let value=numeric[token] {return value}
        if let axis,case .array(let items)=values[key],items.indices.contains(axis) {return items[axis].integerValue.map(String.init) ?? ""}
        return values[key]?.integerValue.map(String.init) ?? ""
    }
    mutating func setText(_ text:String,key:String,axis:Int?=nil) {numeric[key+(axis.map {".\($0)"} ?? "")]=text}
    mutating func select(_ next:ConvolutionConfiguration.Variant,registry:ComponentRegistry)throws {
        let current=try configuration(registry:registry)
        var values=try ConvolutionConfiguration.defaults(for:next,registry:registry)
        for (key,value) in current where values[key] != nil {
            if ["in_channels","out_channels","groups","bias","padding_policy","padding_mode"].contains(key) || variant.dimensions==next.dimensions {values[key]=value}
        }
        if next.operation == .depthwise {
            let input=current["in_channels"]?.integerValue ?? 1
            let output=current["out_channels"]?.integerValue ?? input
            values["channel_multiplier"] = .integer(Int64(input>0 && output % input == 0 ? max(1,output/input):1))
        } else if variant.operation == .depthwise {
            let input=current["in_channels"]?.integerValue ?? 1
            values["out_channels"] = .integer(Int64(input*(current["channel_multiplier"]?.integerValue ?? 1)))
            values["groups"] = .integer(Int64(input))
        }
        self = .init(variant:next,values:try ConvolutionConfiguration(componentID:next.rawValue,configurationVersion:2,values:values,registry:registry).values)
    }
    mutating func setPaddingPolicy(_ policy:String) {
        values["padding_policy"] = .string(policy)
        if policy != "explicit" {
            numeric=numeric.filter {!$0.key.hasPrefix("padding.") && $0.key != "padding"}
            values["padding"]=variant.dimensions==1 ? .integer(0):.array(Array(repeating:.integer(0),count:variant.dimensions))
        }
    }
    func configuration(registry:ComponentRegistry)throws->[String:JSONValue] {
        var result=values
        for (token,text) in numeric {
            let pieces=token.split(separator:".");let key=String(pieces[0])
            guard let value=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {throw GraphConfigurationError("Enter a whole number for \(key.replacingOccurrences(of:"_",with:" ")).")}
            if pieces.count==2,let index=Int(pieces[1]),case .array(var items)=result[key],items.indices.contains(index) {items[index] = .integer(value);result[key] = .array(items)}
            else if pieces.count==1 {result[key] = .integer(value)}
            else {throw GraphConfigurationError("The edited axis is no longer part of this convolution configuration.")}
        }
        return try ConvolutionConfiguration(componentID:variant.rawValue,configurationVersion:2,values:result,registry:registry).values
    }
}

struct ConvolutionConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var allowedVariants:[ConvolutionConfiguration.Variant]=ConvolutionConfiguration.Variant.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(ConvolutionConfiguration.Variant,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=ConvolutionInspectorDraft(variant:.conv2d,values:[:])
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baselineType=""
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.layers.primary:ComponentVisualFamily.layers.secondary}
    private var axes:[String] {draft.variant.dimensions==1 ? ["Length"]:draft.variant.dimensions==2 ? ["Height","Width"]:["Depth","Height","Width"]}
    private var candidate:[String:JSONValue]? {try? draft.configuration(registry:registry)}
    private var changed:Bool {draft.variant.rawValue != baselineType || (candidate.map {$0 != baselineValues} ?? !draft.numeric.isEmpty)}
    private var selectorBusy:Bool {isBusy || !boundFields.isEmpty || (!draft.numeric.isEmpty && candidate==nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Tensor · in → out",systemImage:"circle.hexagongrid").foregroundStyle(accent)
                .frame(maxWidth:.infinity,alignment:.leading).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            choice("Operation",value:draft.variant.operation.rawValue,options:ConvolutionConfiguration.Operation.allCases.filter {operation in allowedVariants.contains {$0.operation==operation && $0.dimensions==draft.variant.dimensions}}.map {.init(id:$0.rawValue,title:$0.rawValue.capitalized)}) {operation in
                select {$0.operation.rawValue==operation && $0.dimensions==draft.variant.dimensions}
            }.disabled(selectorBusy)
            choice("Dimensions",value:String(draft.variant.dimensions),options:[1,2,3].filter {dimension in allowedVariants.contains {$0.dimensions==dimension && $0.operation==draft.variant.operation}}.map {.init(id:String($0),title:"\($0)D")}) {dimension in
                select {$0.dimensions==Int(dimension) && $0.operation==draft.variant.operation}
            }.disabled(selectorBusy)
            number("Input channels",key:"in_channels")
            if draft.variant.operation == .depthwise {
                number("Channel multiplier",key:"channel_multiplier")
                if let values=candidate,let input=values["in_channels"]?.integerValue,let multiplier=values["channel_multiplier"]?.integerValue {
                    Text("Output channels \(input*multiplier) · Groups \(input)").foregroundStyle(accent)
                        .frame(maxWidth:.infinity,alignment:.leading).padding(8).background(accent.opacity(0.08))
                }
            } else {
                number("Output channels",key:"out_channels")
                number("Groups",key:"groups")
            }
            geometry
            if draft.variant.operation != .transposed {
                choice("Padding",value:draft.values["padding_policy"]?.stringValue ?? "explicit",options:[.init(id:"explicit",title:"Explicit amount"),.init(id:"valid",title:"Valid · no padding"),.init(id:"same",title:draft.variant.dimensions==1 ? "Same length":"Same spatial size")]) {value in
                    draft.setPaddingPolicy(value);rejection=nil
                }.disabled(isBusy || boundFields.contains("padding_policy") || boundFields.contains("padding"))
                if draft.values["padding_policy"] == .string("explicit") {axisInputs("padding",title:"Padding")}
                choice("Padding mode",value:draft.values["padding_mode"]?.stringValue ?? "zeros",options:["zeros","reflect","replicate","circular"].map {.init(id:$0,title:$0.capitalized)}) {value in draft.values["padding_mode"] = .string(value);rejection=nil}.disabled(isBusy || boundFields.contains("padding_mode"))
            }
            flag("Use bias",key:"bias")
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("convolution.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("convolution.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Convolution changes")}
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
    private func select(_ predicate:(ConvolutionConfiguration.Variant)->Bool) {
        guard let variant=allowedVariants.first(where:predicate),variant != draft.variant else {return}
        do {try draft.select(variant,registry:registry);rejection=nil} catch {rejection=error.localizedDescription}
    }
    private func choice(_ title:String,value:String,options:[FlatChoiceOption],select:@escaping(String)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value,options:options,accent:accent,onSelect:select).accessibilityIdentifier("convolution."+title.lowercased())}
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil}))
            .toggleStyle(.checkbox).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("convolution."+key)
    }
    private var geometry:some View {
        VStack(alignment:.leading,spacing:6) {
            Text("Kernel geometry").fontWeight(.semibold)
            Grid(alignment:.leading,horizontalSpacing:6,verticalSpacing:6) {
                GridRow {Text("").frame(width:58);ForEach(axes,id:\.self) {Text($0).foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity)}}
                geometryRow("Kernel",key:"kernel_size")
                geometryRow("Stride",key:"stride")
                if draft.variant.operation == .transposed {
                    geometryRow("Padding",key:"padding")
                    geometryRow("Output pad",key:"output_padding")
                }
                geometryRow("Dilation",key:"dilation")
            }.padding(7).background(accent.opacity(0.04)).overlay(Rectangle().stroke(accent.opacity(0.3)))
        }
    }
    private func geometryRow(_ title:String,key:String)->some View {
        GridRow {Text(title).frame(width:58,alignment:.leading);ForEach(axes.indices,id:\.self) {axis in numericCell(title+" "+axes[axis],key:key,axis:draft.variant.dimensions==1 ? nil:axis)}}
    }
    private func axisInputs(_ key:String,title:String)->some View {
        HStack(spacing:6) {ForEach(axes.indices,id:\.self) {axis in number(title+" "+axes[axis],key:key,axis:draft.variant.dimensions==1 ? nil:axis)}}
    }
    private func number(_ title:String,key:String,axis:Int?=nil)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);numericCell(title,key:key,axis:axis)}
    }
    private func numericCell(_ title:String,key:String,axis:Int?=nil)->some View {
        TextField(title,text:Binding(get:{draft.text(key,axis)},set:{draft.setText($0,key:key,axis:axis);rejection=nil}))
            .flatField().frame(minWidth:35,maxWidth:.infinity).disabled(isBusy || boundFields.contains(key)).accessibilityLabel(title)
    }
    private func reload() {
        baselineType=node.type;baselineSource=node.config
        let variant=ConvolutionConfiguration.Variant(rawValue:node.type) ?? .conv2d
        do {let value=try ConvolutionConfiguration(componentID:node.type,configurationVersion:configurationVersion,values:node.config,registry:registry);draft = .init(variant:value.variant,values:value.values);rejection=nil}
        catch {draft = .init(variant:variant,values:node.config);rejection=error.localizedDescription}
        baselineValues=draft.values;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,allowedVariants.contains(draft.variant),let values=candidate else {return}
        guard onApply(draft.variant,values,baselineType,baselineSource) else {rejection="Convolution settings were not applied. Check parameter overrides, connected outputs and the containing block’s bindings.";return}
        baselineType=draft.variant.rawValue;baselineSource=values;baselineValues=values;draft.values=values;draft.numeric=[:];rejection=nil;onPending(false)
    }
}
