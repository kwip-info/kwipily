import SwiftUI
import KwipMLCore

/// Unpublished strings stay local until the entire per-axis configuration validates.
struct NormalizationInspectorDraft:Equatable {
    var variant:NormalizationConfiguration.Variant
    var values:[String:JSONValue]
    var numeric:[String:String]=[:]
    func text(_ key:String,_ axis:Int?=nil)->String {
        let token=key+(axis.map {".\($0)"} ?? "")
        if let value=numeric[token] {return value}
        if let axis,case .array(let items)=values[key],items.indices.contains(axis) {return items[axis].integerValue.map(String.init) ?? ""}
        if let value=values[key]?.integerValue,key != "eps",key != "momentum" {return String(value)}
        return values[key]?.doubleValue.map {String($0)} ?? ""
    }
    mutating func setText(_ text:String,key:String,axis:Int?=nil) {numeric[key+(axis.map {".\($0)"} ?? "")]=text}
    mutating func select(_ next:NormalizationConfiguration.Variant,registry:ComponentRegistry)throws {
        let current=try configuration(registry:registry)
        var defaults=try NormalizationConfiguration.defaults(for:next,registry:registry)
        for key in ["eps","bias"] where defaults[key] != nil && current[key] != nil {defaults[key]=current[key]}
        let affine=current[[.layer,.rms].contains(variant.method) ? "elementwise_affine":"affine"]
        defaults[[.layer,.rms].contains(next.method) ? "elementwise_affine":"affine"]=affine
        let channels=current[variant.method == .group ? "num_channels":"num_features"]
        if ![.layer,.rms].contains(next.method),let channels {defaults[next.method == .group ? "num_channels":"num_features"]=channels}
        if [.layer,.rms].contains(variant.method),[.layer,.rms].contains(next.method) {defaults["normalized_shape"]=current["normalized_shape"]}
        // Running-stat defaults belong to the new method; do not suggest state transfer.
        if next.method==variant.method {
            for (key,value) in current where defaults[key] != nil {defaults[key]=value}
        }
        self = .init(variant:next,values:try NormalizationConfiguration(componentID:next.rawValue,configurationVersion:2,values:defaults,registry:registry).values)
    }
    mutating func addDimension()throws {
        guard [.layerNorm,.rmsNorm].contains(variant),case .array(var shape)=values["normalized_shape"],shape.count<7 else {throw GraphConfigurationError("Layer normalization supports at most seven trailing sample dimensions.")}
        shape.append(2);values["normalized_shape"] = .array(shape)
    }
    mutating func removeDimension(at index:Int)throws {
        guard [.layerNorm,.rmsNorm].contains(variant),case .array(var shape)=values["normalized_shape"],shape.count>1,shape.indices.contains(index) else {throw GraphConfigurationError("Keep at least one normalized sample dimension.")}
        shape.remove(at:index);values["normalized_shape"] = .array(shape)
        var shifted:[String:String]=[:]
        for (key,value) in numeric {
            if key.hasPrefix("normalized_shape."),let axis=Int(key.dropFirst("normalized_shape.".count)) {
                if axis != index {shifted["normalized_shape.\(axis>index ? axis-1:axis)"]=value}
            } else {shifted[key]=value}
        }
        numeric=shifted
    }
    func configuration(registry:ComponentRegistry)throws->[String:JSONValue] {
        var result=values
        for (token,text) in numeric {
            let pieces=token.split(separator:".");let key=String(pieces[0])
            let clean=text.trimmingCharacters(in:.whitespacesAndNewlines)
            if ["eps","momentum"].contains(key),pieces.count==1 {
                guard let value=Double(clean),value.isFinite else {throw GraphConfigurationError("Enter a finite number for \(key == "eps" ? "Epsilon":"Update weight").")}
                result[key] = .number(value)
            } else {
                guard let value=Int64(clean) else {throw GraphConfigurationError("Enter a whole number for \(key.replacingOccurrences(of:"_",with:" ")).")}
                if pieces.count==2,let index=Int(pieces[1]),case .array(var items)=result[key],items.indices.contains(index) {items[index] = .integer(value);result[key] = .array(items)}
                else if pieces.count==1 {result[key] = .integer(value)}
                else {throw GraphConfigurationError("The edited dimension is no longer part of this normalization configuration.")}
            }
        }
        return try NormalizationConfiguration(componentID:variant.rawValue,configurationVersion:2,values:result,registry:registry).values
    }
}

struct NormalizationConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var allowedVariants:[NormalizationConfiguration.Variant]=NormalizationConfiguration.Variant.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(NormalizationConfiguration.Variant,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=NormalizationInspectorDraft(variant:.layerNorm,values:[:])
    @State private var baselineValues:[String:JSONValue]=[:]
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baselineType=""
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.normalization.primary:ComponentVisualFamily.normalization.secondary}
    private var candidate:[String:JSONValue]? {try? draft.configuration(registry:registry)}
    private var changed:Bool {draft.variant.rawValue != baselineType || (candidate.map {$0 != baselineValues} ?? !draft.numeric.isEmpty)}
    private var selectorBusy:Bool {isBusy || !boundFields.isEmpty || (!draft.numeric.isEmpty && candidate==nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Tensor · in → out",systemImage:"circle.hexagongrid").foregroundStyle(accent)
                .frame(maxWidth:.infinity,alignment:.leading).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            choice("Method",value:draft.variant.method.rawValue,options:NormalizationConfiguration.Method.allCases.filter {method in allowedVariants.contains {$0.method==method}}.map {.init(id:$0.rawValue,title:$0 == .rms ? "RMS":$0.rawValue.capitalized)}) {method in
                select {$0.method.rawValue==method && ($0.dimensions==draft.variant.dimensions || $0.dimensions==nil || draft.variant.dimensions==nil)}
            }.disabled(selectorBusy)
            if let dimension=draft.variant.dimensions {
                choice("Dimensions",value:String(dimension),options:[1,2].filter {dimension in allowedVariants.contains {$0.dimensions==dimension && $0.method==draft.variant.method}}.map {.init(id:String($0),title:"\($0)D")}) {dimension in
                    select {$0.dimensions==Int(dimension) && $0.method==draft.variant.method}
                }.disabled(selectorBusy)
            }
            if [.layer,.rms].contains(draft.variant.method) {shapeGrid}
            else {
                number("Channels",key:draft.variant.method == .group ? "num_channels":"num_features")
                if draft.variant.method == .group {number("Groups",key:"num_groups")}
            }
            number("Epsilon",key:"eps")
            if [.batch,.instance].contains(draft.variant.method) {
                flag("Track running statistics",key:"track_running_stats")
                if draft.values["track_running_stats"] == true {
                    if draft.variant.method == .batch {
                        choice("Averaging",value:draft.values["averaging"]?.stringValue ?? "moving",options:[.init(id:"moving",title:"Moving summary"),.init(id:"cumulative",title:"Cumulative summary")]) {value in
                            draft.values["averaging"] = .string(value)
                            if value=="cumulative" {draft.numeric.removeValue(forKey:"momentum")}
                            rejection=nil
                        }.disabled(isBusy || boundFields.contains("averaging"))
                    }
                    if draft.variant.method == .instance || draft.values["averaging"] == .string("moving") {number("Update weight",key:"momentum")}
                }
            }
            Text("Learned adjustments").fontWeight(.semibold)
            flag("Learn scale",key:[.layer,.rms].contains(draft.variant.method) ? "elementwise_affine":"affine")
            if draft.variant != .rmsNorm,draft.values[[.layer,.rms].contains(draft.variant.method) ? "elementwise_affine":"affine"] == true {flag("Use bias",key:"bias")}
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || !changed || candidate==nil).accessibilityIdentifier("normalization.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("normalization.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Normalization changes")}
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
    private func select(_ predicate:(NormalizationConfiguration.Variant)->Bool) {
        guard let variant=allowedVariants.first(where:predicate),variant != draft.variant else {return}
        do {try draft.select(variant,registry:registry);rejection=nil} catch {rejection=error.localizedDescription}
    }
    private func choice(_ title:String,value:String,options:[FlatChoiceOption],select:@escaping(String)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value,options:options,accent:accent,onSelect:select).accessibilityIdentifier("normalization."+title.lowercased())}
    }
    private func flag(_ title:String,key:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.values[key]?.boolValue ?? false},set:{draft.values[key] = .bool($0);if key=="track_running_stats",!$0 {draft.numeric.removeValue(forKey:"momentum")};rejection=nil}))
            .toggleStyle(.checkbox).disabled(isBusy || boundFields.contains(key)).accessibilityIdentifier("normalization."+key)
    }
    private var shapeCount:Int {if case .array(let shape)=draft.values["normalized_shape"] {return shape.count};return 0}
    private var shapeGrid:some View {
        VStack(alignment:.leading,spacing:6) {
            Text("Normalized shape").fontWeight(.semibold)
            Grid(alignment:.leading,horizontalSpacing:6,verticalSpacing:6) {
                GridRow {Text("Order").frame(width:42);Text("Size").frame(maxWidth:.infinity,alignment:.leading);Text("").frame(width:22)}
                ForEach(0..<shapeCount,id:\.self) {index in
                    GridRow {
                        Text(String(index+1)).frame(width:42)
                        numericCell("Dimension \(index+1) size",key:"normalized_shape",axis:index)
                        Button {do {try draft.removeDimension(at:index);rejection=nil} catch {rejection=error.localizedDescription}} label:{Image(systemName:"xmark")}
                            .buttonStyle(.plain).disabled(isBusy || boundFields.contains("normalized_shape") || shapeCount<=1)
                            .accessibilityLabel("Remove dimension \(index+1)")
                    }
                }
            }.padding(7).background(accent.opacity(0.04)).overlay(Rectangle().stroke(accent.opacity(0.3)))
            Button("Add dimension",systemImage:"plus") {do {try draft.addDimension();rejection=nil} catch {rejection=error.localizedDescription}}
                .buttonStyle(.plain).foregroundStyle(accent).disabled(isBusy || boundFields.contains("normalized_shape") || shapeCount>=7)
                .accessibilityIdentifier("normalization.shape.add")
        }
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
        let variant=NormalizationConfiguration.Variant(rawValue:node.type) ?? .layerNorm
        do {let value=try NormalizationConfiguration(componentID:node.type,configurationVersion:configurationVersion,values:node.config,registry:registry);draft = .init(variant:value.variant,values:value.values);rejection=nil}
        catch {draft = .init(variant:variant,values:node.config);rejection=error.localizedDescription}
        baselineValues=draft.values;onPending(false)
    }
    private func apply() {
        guard !isBusy,changed,allowedVariants.contains(draft.variant),let values=candidate else {return}
        guard onApply(draft.variant,values,baselineType,baselineSource) else {rejection="Normalization settings were not applied. Check parameter overrides, connected outputs and the containing block’s bindings.";return}
        baselineType=draft.variant.rawValue;baselineSource=values;baselineValues=values;draft.values=values;draft.numeric=[:];rejection=nil;onPending(false)
    }
}
