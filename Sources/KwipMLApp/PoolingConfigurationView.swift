import SwiftUI
import KwipMLCore

/// Unpublished strings stay local until the entire per-axis configuration validates.
struct PoolingInspectorDraft:Equatable {
    var variant:PoolingConfiguration.Variant
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    func text(_ key:String,_ axis:Int?=nil)->String {
        let token=key+(axis.map {".\($0)"} ?? "")
        if let value=numeric[token] {return value}
        if let axis,case .array(let items)=values[key],items.indices.contains(axis) {return items[axis].integerValue.map(String.init) ?? ""}
        return values[key]?.integerValue.map(String.init) ?? ""
    }
    mutating func setText(_ text:String,key:String,axis:Int?=nil) {numeric[key+(axis.map {".\($0)"} ?? "")]=text}
    mutating func select(_ next:PoolingConfiguration.Variant,registry:ComponentRegistry)throws {
        var defaults=try PoolingConfiguration.defaults(for:next,registry:registry)
        if let current=try? configuration(registry:registry) {
            for (key,value) in current where defaults[key] != nil {
                let sameGeometry=variant.dimensions==next.dimensions && variant.sizing==next.sizing
                let commonFlag=["ceil_mode","count_include_pad"].contains(key)
                let sameIndices=key=="return_indices" && variant.dimensions==next.dimensions
                if sameGeometry || commonFlag || sameIndices {defaults[key]=value}
            }
        }
        self = .init(variant:next,values:try PoolingConfiguration(componentID:next.rawValue,configurationVersion:2,values:defaults,registry:registry).values)
    }
    func configuration(registry:ComponentRegistry)throws->[String:JSONValue] {
        var result=values
        for (token,text) in numeric {
            let pieces=token.split(separator:".");let key=String(pieces[0])
            guard let value=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {throw GraphConfigurationError("Enter a whole number for \(key.replacingOccurrences(of:"_",with:" ")).")}
            if pieces.count==2,let index=Int(pieces[1]),case .array(var items)=result[key],items.indices.contains(index) {items[index] = .integer(value);result[key] = .array(items)}
            else if pieces.count==1 {result[key] = .integer(value)}
            else {throw GraphConfigurationError("The edited axis is no longer part of this pooling configuration.")}
        }
        return try PoolingConfiguration(componentID:variant.rawValue,configurationVersion:2,values:result,registry:registry).values
    }
}

struct PoolingConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var allowedVariants:[PoolingConfiguration.Variant]=PoolingConfiguration.Variant.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(PoolingConfiguration.Variant,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=PoolingInspectorDraft(variant:.maxPool2d,values:[:])
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baselineType=""
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.pooling.primary:ComponentVisualFamily.pooling.secondary}
    private var axes:[String] {draft.variant.dimensions==1 ? ["Length"]:draft.variant.dimensions==2 ? ["Height","Width"]:["Depth","Height","Width"]}
    private var candidate:[String:JSONValue]? {try? draft.configuration(registry:registry)}
    private var changed:Bool {draft.variant.rawValue != baselineType || (candidate.map {$0 != baselineValues} ?? !draft.numeric.isEmpty)}
    private var selectorBusy:Bool {isBusy || !boundFields.isEmpty || (!draft.numeric.isEmpty && candidate==nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Tensor · in → out",systemImage:"circle.hexagongrid").foregroundStyle(accent)
                .frame(maxWidth:.infinity,alignment:.leading).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            choice("Method",value:draft.variant.method.rawValue,options:PoolingConfiguration.Method.allCases.filter {method in allowedVariants.contains {$0.method==method && $0.sizing==draft.variant.sizing && $0.dimensions==draft.variant.dimensions}}.map {.init(id:$0.rawValue,title:$0 == .max ? "Max":"Average")}) {method in
                select { $0.method.rawValue==method && $0.sizing==draft.variant.sizing && $0.dimensions==draft.variant.dimensions }
            }.disabled(selectorBusy)
            choice("Sizing",value:draft.variant.sizing.rawValue,options:PoolingConfiguration.Sizing.allCases.filter {sizing in allowedVariants.contains {$0.sizing==sizing && $0.method==draft.variant.method && $0.dimensions==draft.variant.dimensions}}.map {.init(id:$0.rawValue,title:$0 == .fixed ? "Fixed window":"Adaptive output")}) {sizing in
                select {$0.sizing.rawValue==sizing && $0.method==draft.variant.method && $0.dimensions==draft.variant.dimensions}
            }.disabled(selectorBusy)
            choice("Dimensions",value:String(draft.variant.dimensions),options:[1,2,3].filter {dimension in allowedVariants.contains {$0.dimensions==dimension && $0.method==draft.variant.method && $0.sizing==draft.variant.sizing}}.map {.init(id:String($0),title:"\($0)D")}) {dimension in
                select {$0.dimensions==Int(dimension) && $0.method==draft.variant.method && $0.sizing==draft.variant.sizing}
            }.disabled(selectorBusy)
            if draft.variant.sizing == .fixed {
                geometry
                choice("Stride",value:draft.values["stride"] == .null ? "window":"custom",options:[.init(id:"window",title:"Match window"),.init(id:"custom",title:"Custom")]) {value in
                    draft.numeric=draft.numeric.filter {!$0.key.hasPrefix("stride")}
                    draft.values["stride"]=value=="window" ? .null:(candidate?["kernel_size"] ?? draft.values["kernel_size"])
                }.disabled(isBusy || boundFields.contains("stride"))
                if draft.values["stride"] != .null {axisInputs("stride",title:"Stride")}
                flag("Round partial windows up",key:"ceil_mode")
                if draft.variant.method == .max {flag("Expose winning indices",key:"return_indices")}
                else {
                    flag("Include padding in count",key:"count_include_pad")
                    if draft.variant == .avgPool2d {
                        choice("Average divisor",value:draft.values["divisor_override"] == .null ? "count":"override",options:[.init(id:"count",title:"Window count"),.init(id:"override",title:"Override")]) {value in
                            draft.numeric.removeValue(forKey:"divisor_override");draft.values["divisor_override"]=value=="count" ? .null:.integer(1)
                        }.disabled(isBusy || boundFields.contains("divisor_override"))
                        if draft.values["divisor_override"] != .null {number("Divisor",key:"divisor_override")}
                    }
                }
            } else {
                Text("Target sizes").fontWeight(.semibold)
                ForEach(Array(axes.enumerated()),id:\.offset) {index,name in
                    let key=draft.variant.dimensions==1 ? "output_size":index==0 ? "output_height":"output_width"
                    if draft.variant.dimensions>1 {
                        choice(name,value:draft.values[key] == .null ? "keep":"set",options:[.init(id:"keep",title:"Keep input size"),.init(id:"set",title:"Set size")]) {value in
                            draft.numeric.removeValue(forKey:key);draft.values[key]=value=="keep" ? .null:.integer(2)
                        }.disabled(isBusy || boundFields.contains(key))
                    }
                    if draft.values[key] != .null {number(name,key:key)}
                }
                if draft.variant.method == .max {flag("Expose winning indices",key:"return_indices")}
            }
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("pooling.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("pooling.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Pooling changes")}
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
    private func select(_ predicate:(PoolingConfiguration.Variant)->Bool) {
        guard let variant=allowedVariants.first(where:predicate),variant != draft.variant else {return}
        do {try draft.select(variant,registry:registry);rejection=nil} catch {rejection=error.localizedDescription}
    }
    private func choice(_ title:String,value:String,options:[FlatChoiceOption],select:@escaping(String)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value,options:options,accent:accent,onSelect:select).accessibilityIdentifier("pooling."+title.lowercased())}
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);rejection=nil}))
            .toggleStyle(.checkbox).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("pooling."+key)
    }
    private var geometry:some View {
        VStack(alignment:.leading,spacing:6) {
            Text("Window geometry").fontWeight(.semibold)
            Grid(alignment:.leading,horizontalSpacing:6,verticalSpacing:6) {
                GridRow {Text("").frame(width:58);ForEach(axes,id:\.self) {Text($0).foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity)}}
                geometryRow("Window",key:"kernel_size")
                geometryRow("Padding",key:"padding")
                if draft.variant.method == .max {geometryRow("Dilation",key:"dilation")}
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
        let variant=PoolingConfiguration.Variant(rawValue:node.type) ?? .maxPool2d
        do {let value=try PoolingConfiguration(componentID:node.type,configurationVersion:configurationVersion,values:node.config,registry:registry);draft = .init(variant:value.variant,values:value.values);rejection=nil}
        catch {draft = .init(variant:variant,values:node.config);rejection=error.localizedDescription}
        baselineValues=draft.values;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,allowedVariants.contains(draft.variant),let values=candidate else {return}
        guard onApply(draft.variant,values,baselineType,baselineSource) else {rejection="Pooling settings were not applied. Check connected outputs and the containing block’s bindings.";return}
        baselineType=draft.variant.rawValue;baselineSource=values;baselineValues=values;draft.values=values;draft.numeric=[:];rejection=nil;onPending(false)
    }
}
