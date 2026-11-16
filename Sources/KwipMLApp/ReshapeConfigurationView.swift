import SwiftUI
import KwipMLCore

struct ReshapeDimensionDraft:Identifiable,Equatable {
    let id=UUID()
    var size:String
    var inferred:Bool
}
struct ReshapeInspectorDraft:Equatable {
    var variant:ReshapeAuthoringConfiguration.Variant
    var rows:[ReshapeDimensionDraft]
    var startAxis="1",endAxis="-1"
    var hasDimensionDraft:Bool
    init(variant:ReshapeAuthoringConfiguration.Variant,values:[String:JSONValue])throws {
        let values=try TensorTransformConfiguration(operation:variant.rawValue,values:values).values
        self.variant=variant;hasDimensionDraft=variant == .dimensions
        if case .array(let shape)=values["shape"] {rows=shape.map {.init(size:$0.integerValue == -1 ? "1":String($0.integerValue!),inferred:$0.integerValue == -1)}}
        else {rows=[.init(size:"1",inferred:true)];startAxis=String(values["start_dim"]!.integerValue!);endAxis=String(values["end_dim"]!.integerValue!)}
    }
    mutating func select(_ next:ReshapeAuthoringConfiguration.Variant,input:BatchedTensorLayout?=nil)throws {
        guard next != variant else {return}
        let current=try configuration()
        if next == .dimensions,!hasDimensionDraft,let input {
            let output=try ReshapeAuthoringConfiguration.outputLayout(variant:variant,configuration:current,input:input)
            rows=output.sample.dimensions.map {.init(size:String($0),inferred:false)}
        }
        if next == .dimensions {hasDimensionDraft=true}
        variant=next
    }
    mutating func moveRow(id:UUID,offset:Int) {
        guard let index=rows.firstIndex(where:{$0.id==id}),rows.indices.contains(index+offset) else {return}
        rows.swapAt(index,index+offset)
    }
    func configuration()throws->[String:JSONValue] {
        func whole(_ value:String)throws->JSONValue {
            guard let number=Int64(value.trimmingCharacters(in:.whitespacesAndNewlines)) else {throw GraphConfigurationError("Enter a whole number for dimensions and axes.")}
            return .integer(number)
        }
        let values:[String:JSONValue]
        if variant == .dimensions {values=["shape":.array(try rows.map {$0.inferred ? .integer(-1):try whole($0.size)})]}
        else {values=["start_dim":try whole(startAxis),"end_dim":try whole(endAxis)]}
        return try TensorTransformConfiguration(operation:variant.rawValue,values:values).values
    }
}

struct ReshapeConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var inputLayout:BatchedTensorLayout?=nil
    var allowedVariants=ReshapeAuthoringConfiguration.Variant.allCases
    var boundFields:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(ReshapeAuthoringConfiguration.Variant,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=try! ReshapeInspectorDraft(variant:.dimensions,values:["shape":.array([-1])])
    @State private var baselineType=""
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baseline:[String:JSONValue]=[:]
    @State private var rejection:String?
    @State private var loadFailed=false
    @State private var customStart=false
    @State private var customEnd=false
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.tensorLayout.primary:ComponentVisualFamily.tensorLayout.secondary}
    private var candidate:[String:JSONValue]? {loadFailed ? nil:try? draft.configuration()}
    private var changed:Bool {draft.variant.rawValue != baselineType || (candidate.map {$0 != baseline} ?? true)}
    private var plannedOutput:BatchedTensorLayout? {guard let inputLayout,let candidate else {return nil};return try? ReshapeAuthoringConfiguration.outputLayout(variant:draft.variant,configuration:candidate,input:inputLayout)}
    private var validationMessage:String? {
        do {let values=try draft.configuration();if let inputLayout {_=try ReshapeAuthoringConfiguration.outputLayout(variant:draft.variant,configuration:values,input:inputLayout)};return nil}
        catch {return error.localizedDescription}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if loadFailed {
                Label(rejection ?? "Restore valid source settings before editing.",systemImage:"exclamationmark.circle").foregroundStyle(.red)
                Button("Reload settings") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy)
            } else {
                Text("Method")
                FlatChoiceControl(title:"Method",selection:draft.variant.rawValue,options:allowedVariants.map {.init(id:$0.rawValue,title:$0.title)},accent:accent) {value in
                    guard let next=ReshapeAuthoringConfiguration.Variant(rawValue:value) else {return}
                    do {try draft.select(next,input:inputLayout);rejection=nil} catch {rejection=error.localizedDescription}
                }.disabled(isBusy || !boundFields.isEmpty || candidate==nil)
                Label("Batch N · preserved",systemImage:"lock.fill").foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading).padding(8).background(accent.opacity(0.08))
                if draft.variant == .dimensions {dimensions}
                else {axis("Start axis",first:true,key:"start_dim");axis("End axis",first:false,key:"end_dim")}
                plannedShape
                HStack {
                    Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(isBusy || !changed || candidate==nil || validationMessage != nil).accessibilityIdentifier("reshape.apply")
                    Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("reshape.revert")
                    Spacer(minLength:0)
                    if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Reshape changes")}
                }
                if let message=rejection ?? validationMessage {Text(message).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            }
        }.font(.caption).tint(accent).toggleStyle(.checkbox)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.type != baselineType || next.config != baselineSource else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    private var dimensions:some View {
        VStack(alignment:.leading,spacing:8) {
            Text("Sample dimensions").fontWeight(.semibold).foregroundStyle(accent)
            HStack(spacing:6) {Text("Dimension").frame(width:58,alignment:.leading);Text("Size").frame(maxWidth:.infinity,alignment:.leading);Text("Infer").frame(width:35);Color.clear.frame(width:20,height:1)}.font(.caption2).foregroundStyle(MLTheme.muted)
            ForEach($draft.rows) {$row in
                let index=draft.rows.firstIndex {$0.id==row.id} ?? 0
                HStack(spacing:6) {
                    VStack(spacing:2) {
                        Button {draft.moveRow(id:row.id,offset:-1)} label:{Image(systemName:"chevron.up")}.disabled(index==0).accessibilityLabel("Move dimension \(index+1) up")
                        Button {draft.moveRow(id:row.id,offset:1)} label:{Image(systemName:"chevron.down")}.disabled(index==draft.rows.count-1).accessibilityLabel("Move dimension \(index+1) down")
                    }.buttonStyle(.plain).font(.system(size:8)).frame(width:15)
                    Text(String(index+1)).frame(width:37,alignment:.leading)
                    if row.inferred {Text("Auto").foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,alignment:.leading).padding(7).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))}
                    else {TextField("Size",text:$row.size).flatField().frame(minWidth:35,maxWidth:.infinity).accessibilityLabel("Dimension \(index+1) size")}
                    Toggle("Infer dimension \(index+1)",isOn:$row.inferred).labelsHidden().frame(width:35).disabled(!row.inferred && draft.rows.contains {$0.inferred})
                    Button {draft.rows.removeAll {$0.id==row.id}} label:{Image(systemName:"trash")}.buttonStyle(.plain).frame(width:20).disabled(draft.rows.count<=1).accessibilityLabel("Remove dimension \(index+1)")
                }.padding(7).background(accent.opacity(0.035)).overlay(Rectangle().stroke(MLTheme.rule))
            }
            Button("Add dimension") {draft.rows.append(.init(size:"1",inferred:false))}.buttonStyle(FlatButtonStyle()).disabled(draft.rows.count>=7).accessibilityIdentifier("reshape.add-dimension")
        }.disabled(isBusy || boundFields.contains("shape"))
    }
    private func axis(_ title:String,first:Bool,key:String)->some View {
        let value=first ? draft.startAxis:draft.endAxis
        let custom=first ? customStart:customEnd
        let axes=inputLayout.map {input in (1..<input.rank).map {FlatChoiceOption(id:String($0),title:"Axis \($0) · size \(input.sample.dimensions[$0-1])")}} ?? [.init(id:"1",title:"First sample axis (1)")]
        let options=axes+[.init(id:"-1",title:"Last sample axis (-1)"),.init(id:"custom",title:"Enter axis…")]
        let selected=custom || !options.contains(where:{$0.id==value}) ? "custom":value
        return VStack(alignment:.leading,spacing:4) {
            Text(title)
            FlatChoiceControl(title:title,selection:selected,options:options,accent:accent) {next in
                if first {customStart=next=="custom";if next != "custom" {draft.startAxis=next}}
                else {customEnd=next=="custom";if next != "custom" {draft.endAxis=next}}
                rejection=nil
            }
            if selected=="custom" {TextField(title,text:Binding(get:{first ? draft.startAxis:draft.endAxis},set:{if first {draft.startAxis=$0} else {draft.endAxis=$0};rejection=nil})).flatField().accessibilityLabel(title+" number")}
        }.disabled(isBusy || boundFields.contains(key))
    }
    private var plannedShape:some View {
        VStack(alignment:.leading,spacing:6) {
            Text("Planned shape").fontWeight(.semibold).foregroundStyle(accent)
            if let inputLayout {
                plannedRow("Input",inputLayout)
                if let plannedOutput {
                    plannedRow("Output",plannedOutput)
                    Text("\(inputLayout.sample.elementCount.formatted()) values per observation").foregroundStyle(MLTheme.muted)
                    if draft.variant == .dimensions,let index=draft.rows.firstIndex(where:{$0.inferred}) {Text("Inferred size · \(plannedOutput.sample.dimensions[index])").foregroundStyle(accent)}
                } else {Label("Output unresolved",systemImage:"exclamationmark.circle").foregroundStyle(.orange)}
            } else {Label("Input unresolved",systemImage:"questionmark.circle").foregroundStyle(MLTheme.muted)}
        }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background(accent.opacity(0.04)).overlay(Rectangle().stroke(accent.opacity(0.25)))
    }
    private func plannedRow(_ title:String,_ layout:BatchedTensorLayout)->some View {
        HStack(alignment:.top) {Text(title).frame(width:42,alignment:.leading);Text(layout.displayDimensions.joined(separator:" × ")).monospacedDigit().fixedSize(horizontal:false,vertical:true)}
    }
    private func reload() {
        baselineType=node.type;baselineSource=node.config;loadFailed=false
        do {
            guard let variant=ReshapeAuthoringConfiguration.Variant(rawValue:node.type) else {throw GraphConfigurationError("Choose an existing transform.")}
            let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:configurationVersion,registry:registry)
            draft=try .init(variant:variant,values:values);baseline=values;rejection=nil;customStart=false;customEnd=false;onPending(false)
        }catch {loadFailed=true;rejection=error.localizedDescription;onPending(false)}
    }
    private func apply() {
        guard !isBusy,changed,allowedVariants.contains(draft.variant),let values=candidate,validationMessage==nil else {return}
        guard onApply(draft.variant,values,baselineType,baselineSource) else {rejection="Settings were not applied. Check selection, incoming shape and the containing block’s bindings.";return}
        baselineType=draft.variant.rawValue;baselineSource=values;baseline=values;rejection=nil;onPending(false)
    }
}
