import SwiftUI
import KwipMLCore

struct TensorOperationRow:Identifiable,Equatable {let id=UUID();var value:String}
struct TensorOperationInspectorDraft:Equatable {
    var type:String
    var texts:[String:String]=[:]
    var order:[TensorOperationRow]=[]
    private var batchAxis=0
    private var batchRank=0
    var sections:[TensorOperationRow]=[]
    var splitMode="size",pattern="multiply",aLabels="ab",bLabels="bc",outputLabels="ac"
    var sampleBroadcast=false
    var hasB=true
    private var remembered:[String:[String:JSONValue]]=[:]
    init(type:String,values:[String:JSONValue])throws {
        self.type=type
        let canonical=try TensorOperationAuthoring.canonical(type,values)
        for (key,value) in canonical {
            if let integer=value.integerValue {texts[key]=String(integer)}
            if case .number(let number)=value {texts[key]=String(number)}
            if case .array(let items)=value {for (index,item) in items.enumerated() {texts["\(key).\(index)"]=String(item.integerValue!)}}
        }
        if case .array(let entries)=canonical["dims"] {batchAxis=entries[0].integerValue!;batchRank=entries.count;order=entries.dropFirst().map {.init(value:String($0.integerValue!))}}
        if case .array(let entries)=canonical["sections"] {sections=entries.map {.init(value:String($0.integerValue!))}}
        if case .bool(let enabled)=canonical["sample_broadcast"] {sampleBroadcast=enabled}
        splitMode=canonical["mode"]?.stringValue ?? "size"
        pattern=canonical["pattern"]?.stringValue ?? "multiply"
        if let equation=canonical["equation"]?.stringValue {
            let sides=equation.components(separatedBy:"->"),operands=sides[0].components(separatedBy:",")
            aLabels=String(operands[0].dropFirst());hasB=operands.count==2
            bLabels=hasB ? String(operands[1].dropFirst()):"";outputLabels=String(sides[1].dropFirst())
        }
    }
    mutating func select(_ next:String,registry:ComponentRegistry,input:BatchedTensorLayout?)throws {
        guard type != next,TensorOperationAuthoring.family(type)==TensorOperationAuthoring.family(next) else {return}
        let previous=try configuration();remembered[type]=previous
        var values=try remembered[next] ?? TensorOperationAuthoring.defaults(next,registry:registry)
        if remembered[next]==nil,next=="permute",type=="transpose",let input {
            let first=previous["dim0"]!.integerValue!,second=previous["dim1"]!.integerValue!
            var axes=Array(0..<input.rank)
            let a=first<0 ? first+input.rank:first,b=second<0 ? second+input.rank:second
            guard axes.indices.contains(a),axes.indices.contains(b),a>0,b>0 else {throw GraphConfigurationError("Choose valid sample axes before changing method.")}
            axes.swapAt(a,b);values=["dims":.array(axes.map {.integer(Int64($0))})]
        }
        if remembered[next]==nil,let dim=previous["dim"],TensorOperationAuthoring.family(next) == .parts {values["dim"]=dim}
        var replacement=try Self(type:next,values:values);replacement.remembered=remembered;self=replacement
    }
    mutating func moveOrder(id:UUID,offset:Int) {guard let index=order.firstIndex(where:{$0.id==id}),order.indices.contains(index+offset) else {return};order.swapAt(index,index+offset)}
    mutating func moveSection(id:UUID,offset:Int) {guard let index=sections.firstIndex(where:{$0.id==id}),sections.indices.contains(index+offset) else {return};sections.swapAt(index,index+offset)}
    func configuration()throws->[String:JSONValue] {
        func integer(_ text:String)throws->JSONValue {
            guard let number=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {throw GraphConfigurationError("Enter a whole number for axes, sizes and geometry.")};return .integer(number)
        }
        func field(_ key:String)throws->JSONValue {try integer(texts[key] ?? "")}
        func pair(_ key:String)throws->JSONValue {.array([try field(key+".0"),try field(key+".1")])}
        let values:[String:JSONValue]
        switch type {
        case "transpose":values=["dim0":try field("dim0"),"dim1":try field("dim1")]
        case "permute":values=["dims":.array([.integer(Int64(order.count+1==batchRank ? batchAxis:0))]+(try order.map {try integer($0.value)}))]
        case "split":values=["dim":try field("dim"),"mode":.string(splitMode),"split_size":splitMode=="size" ? try field("split_size"):2,"sections":splitMode=="sections" ? .array(try sections.map {try integer($0.value)}):.array([1,1])]
        case "chunk":values=["dim":try field("dim"),"chunks":try field("chunks")]
        case "add":
            guard let alpha=Double((texts["alpha"] ?? "").trimmingCharacters(in:.whitespacesAndNewlines)) else {throw GraphConfigurationError("Enter a finite number for B multiplier.")}
            values=["alpha":.number(alpha),"sample_broadcast":.bool(sampleBroadcast)]
        case "concat","stack","gather":values=["dim":try field("dim")]
        case "einsum":values=["pattern":.string(pattern),"equation":.string(pattern=="custom" ? "n"+aLabels+(hasB ? ",n"+bLabels:"")+"->n"+outputLabels:(TensorRoutingConfiguration.equationPresets[pattern] ?? ""))]
        case "unfold","fold":
            var geometry:[String:JSONValue]=[:]
            for key in ["kernel_size","stride","dilation","padding"] {geometry[key]=try pair(key)}
            if type=="fold" {geometry["output_size"]=try pair("output_size")};values=geometry
        default:values=[:]
        }
        return try TensorOperationAuthoring.canonical(type,values)
    }
}

struct TensorOperationConfigurationView:View {
    let node:GraphNode
    let configurationVersion:Int
    let registry:ComponentRegistry
    var inputLayouts:[String:BatchedTensorLayout]=[:]
    var inputLabels:[String:String]=[:]
    var inputContracts:[String:TypedTensorContract]=[:]
    var sourceIssue:String?
    var boundFields:Set<String>=[]
    var connectedInputs:Set<String>=[]
    var connectedOutputs:Set<String>=[]
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(String,[String:JSONValue],String,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=try! TensorOperationInspectorDraft(type:"transpose",values:["dim0":1,"dim1":2])
    @State private var baselineType=""
    @State private var baselineSource:[String:JSONValue]=[:]
    @State private var baseline:[String:JSONValue]=[:]
    @State private var rejection:String?
    @State private var loadFailed=false
    @State private var customAxes:Set<String>=[]
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.resolve(type:node.type).primary:ComponentVisualFamily.resolve(type:node.type).secondary}
    private var family:TensorOperationAuthoring.Family? {TensorOperationAuthoring.family(node.type)}
    private var candidate:[String:JSONValue]? {loadFailed ? nil:try? draft.configuration()}
    private var changed:Bool {draft.type != baselineType || (candidate.map {$0 != baseline} ?? true)}
    private var ports:[String] {(try? candidate.map {try TensorOperationAuthoring.inputPorts(draft.type,$0)}) ?? ["in"]}
    private var hasInputs:Bool {!ports.isEmpty && ports.allSatisfy {inputLayouts[$0] != nil}}
    private var planned:[String:BatchedTensorLayout]? {guard hasInputs,let candidate,(try? TensorOperationAuthoring.validateKnownTypes(type:draft.type,values:candidate,inputs:inputContracts)) != nil else {return nil};return try? TensorOperationAuthoring.outputs(type:draft.type,values:candidate,inputs:inputLayouts)}
    private var problem:String? {
        do {let values=try draft.configuration();try TensorOperationAuthoring.validateKnownTypes(type:draft.type,values:values,inputs:inputContracts);if hasInputs {_=try TensorOperationAuthoring.outputs(type:draft.type,values:values,inputs:inputLayouts)};if changed {_=try TensorOperationAuthoring.switching(node,configurationVersion:configurationVersion,to:draft.type,configuration:values,connectedInputs:connectedInputs,connectedOutputs:connectedOutputs,inputs:inputLayouts,registry:registry)};return nil}
        catch {return error.localizedDescription}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if loadFailed {Text(rejection ?? "Repair the saved tensor configuration before editing.").foregroundStyle(.red);Button("Reload settings") {reload()}.buttonStyle(FlatButtonStyle())}
            else {
                if let family,family.types.count>1 {
                    Text("Method")
                    FlatChoiceControl(title:"Method",selection:draft.type,options:family.types.filter {registry.component(type:$0) != nil}.map {.init(id:$0,title:TensorOperationAuthoring.methodTitle($0))},accent:accent) {next in
                        do {try draft.select(next,registry:registry,input:inputLayouts["in"]);customAxes=[];rejection=nil} catch {rejection=error.localizedDescription}
                    }.disabled(isBusy || !boundFields.isEmpty || candidate==nil)
                }
                if draft.type != "concat",draft.type != "add",draft.type != "stack" {Label("Batch N · preserved",systemImage:"lock.fill").foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading).padding(8).background(accent.opacity(0.08))}
                controls
                if draft.type=="concat" {concatenateShape} else if draft.type=="add" {additionShape} else if draft.type=="stack" {stackShape} else {plannedShape}
                if draft.type != "matmul" {
                    HStack {
                        Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(isBusy || !changed || candidate==nil || problem != nil).accessibilityIdentifier("tensor-operation.apply")
                        Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("tensor-operation.revert")
                        Spacer(minLength:0)
                        if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied tensor changes")}
                    }
                }
                if let message=rejection ?? problem ?? (hasInputs ? nil:sourceIssue) {Text(message).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            }
        }.font(.caption).tint(accent).toggleStyle(.checkbox)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node) {_,next in
                guard next.type != baselineType || next.config != baselineSource else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    @ViewBuilder private var controls:some View {
        switch draft.type {
        case "add":additionControls;concatenateInputs;broadcastPlan
        case "stack":stackAxis;concatenateInputs
        case "concat":axis("Join axis",key:"dim",port:"a");concatenateInputs
        case "transpose":axis("First axis",key:"dim0");axis("Second axis",key:"dim1")
        case "permute":axisOrder
        case "split":
            axis("Split axis",key:"dim")
            Text("Sizing")
            FlatChoiceControl(title:"Sizing",selection:draft.splitMode,options:[.init(id:"size",title:"Fixed part size"),.init(id:"sections",title:"Explicit section sizes")],accent:accent) {draft.splitMode=$0;rejection=nil}.disabled(isBusy || boundFields.contains("mode"))
            if draft.splitMode=="size" {number("Part size",key:"split_size")}
            else {sectionSizes}
        case "chunk":axis("Split axis",key:"dim");number("Requested chunks",key:"chunks")
        case "gather":axis("Gather axis",key:"dim",port:"source")
        case "einsum":equationEditor
        case "unfold","fold":patchGeometry
        default:EmptyView()
        }
    }
    private func textBinding(_ key:String)->Binding<String> {Binding(get:{draft.texts[key] ?? ""},set:{draft.texts[key]=$0;rejection=nil})}
    private func number(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:textBinding(key)).flatField().accessibilityLabel(title)}.disabled(isBusy || boundFields.contains(key))
    }
    private func axis(_ title:String,key:String,port:String="in")->some View {
        let value=draft.texts[key] ?? "",input=inputLayouts[port]
        let axes:[FlatChoiceOption]=input.map {input in (1..<input.rank).map {axis in
            let a=input.sample.dimensions[axis-1]
            let detail:String
            if draft.type=="concat",let b=inputLayouts["b"],b.rank==input.rank {let width=b.sample.dimensions[axis-1];detail="\(a) + \(width) → \(a+width)"} else {detail="size \(a)"}
            return FlatChoiceOption(id:String(axis),title:"Axis \(axis) · \(detail)")
        }} ?? [.init(id:"1",title:"First sample axis (1)")]
        let options=axes+[.init(id:"-1",title:"Last sample axis (-1)"),.init(id:"custom",title:"Enter axis…")]
        let selected=customAxes.contains(key) || !options.contains(where:{$0.id==value}) ? "custom":value
        return VStack(alignment:.leading,spacing:4) {
            Text(title)
            FlatChoiceControl(title:title,selection:selected,options:options,accent:accent) {next in
                if next=="custom" {customAxes.insert(key)} else {customAxes.remove(key);draft.texts[key]=next};rejection=nil
            }
            if selected=="custom" {TextField(title,text:textBinding(key)).flatField().accessibilityLabel(title+" number")}
            if draft.type=="concat",let input,let axis=Int(value),axis != 0 {
                let resolved=axis<0 ? axis+input.rank:axis
                if resolved>0,resolved<input.rank {Text("Axis \(resolved)"+(resolved==input.rank-1 ? " · last":"")).foregroundStyle(accent)}
            }
        }.disabled(isBusy || boundFields.contains(key))
    }
    private var stackAxis:some View {
        let value=draft.texts["dim"] ?? ""
        let rank=inputLayouts["a"].map {$0.rank+1}
        let positions:[FlatChoiceOption]=rank.map {rank in (1..<rank).map {axis in
            .init(id:String(axis),title:"Axis \(axis) · "+(axis==1 ? "before sample dimensions":axis==rank-1 ? "after sample dimensions":"between sample dimensions \(axis-1) and \(axis)"))
        }} ?? [.init(id:"1",title:"First sample position (1)")]
        let options=[FlatChoiceOption(id:"-1",title:"Last position (-1)")]+positions+[.init(id:"custom",title:"Custom position…")]
        let selected=customAxes.contains("dim") || !options.contains(where:{$0.id==value}) ? "custom":value
        return VStack(alignment:.leading,spacing:4) {
            Text("New axis position")
            FlatChoiceControl(title:"New axis position",selection:selected,options:options,accent:accent) {next in
                if next=="custom" {customAxes.insert("dim")} else {customAxes.remove("dim");draft.texts["dim"]=next};rejection=nil
            }
            if selected=="custom" {TextField("Axis number",text:textBinding("dim")).flatField().accessibilityLabel("New axis number")}
        }.disabled(isBusy || boundFields.contains("dim"))
    }
    private var stackShape:some View {
        VStack(alignment:.leading,spacing:7) {
            additionShape
            if let output=planned?["out"],let dim=candidate?["dim"]?.integerValue {
                let axis=dim<0 ? dim+output.rank:dim
                HStack(spacing:0) {
                    ForEach(Array(output.displayDimensions.enumerated()),id:\.offset) {index,size in
                        VStack(spacing:2) {
                            Text(size).fontWeight(index==axis ? .semibold:.regular).monospacedDigit()
                            if index==axis {Text("new").font(.system(size:9))}
                        }.frame(maxWidth:.infinity).frame(height:38).background(accent.opacity(index==axis ? 0.22:0.05)).overlay(Rectangle().stroke(accent.opacity(0.2)))
                    }
                }.foregroundStyle(accent).accessibilityElement(children:.ignore).accessibilityLabel("New axis \(axis), size 2. "+shapeText(output))
                Text("New axis \(axis) · size 2").foregroundStyle(accent)
            }
        }
    }
    private var additionControls:some View {
        VStack(alignment:.leading,spacing:8) {
            number("B multiplier",key:"alpha")
            Text(additionFormula).fontWeight(.semibold).foregroundStyle(accent).frame(maxWidth:.infinity).padding(8).background(accent.opacity(0.07)).accessibilityLabel("Planned calculation: "+additionFormula)
            Toggle("Broadcast sample dimensions",isOn:Binding(get:{draft.sampleBroadcast},set:{draft.sampleBroadcast=$0;rejection=nil})).disabled(isBusy || boundFields.contains("sample_broadcast"))
        }
    }
    private var additionFormula:String {
        guard let value=Double(draft.texts["alpha"] ?? ""),value.isFinite else {return "A + multiplier × B"}
        if value==1 {return "A + B"};if value == -1 {return "A − B"}
        return "A "+(value<0 ? "−":"+")+" "+String(abs(value))+" × B"
    }
    @ViewBuilder private var broadcastPlan:some View {
        if draft.sampleBroadcast,let output=planned?["out"] {
            VStack(alignment:.leading,spacing:5) {
                ForEach(["a","b"],id:\.self) {port in
                    if let input=inputLayouts[port],input.sample.dimensions != output.sample.dimensions {
                        let padded=Array(repeating:1,count:output.sample.dimensions.count-input.sample.dimensions.count)+input.sample.dimensions
                        VStack(alignment:.leading,spacing:3) {
                            Text(port.uppercased()+" sample").foregroundStyle(MLTheme.muted)
                            Text("["+input.sample.dimensions.map(String.init).joined(separator:", ")+"] → ["+padded.map(String.init).joined(separator:", ")+"] → ["+output.sample.dimensions.map(String.init).joined(separator:", ")+"]")
                                .monospacedDigit().fixedSize(horizontal:false,vertical:true).frame(maxWidth:.infinity,alignment:.leading)
                        }.padding(7).background(accent.opacity(0.04)).overlay(Rectangle().stroke(accent.opacity(0.2)))
                    }
                }
            }
        }
    }
    private var additionShape:some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {Text("Planned shape").fontWeight(.semibold).foregroundStyle(accent);Spacer();Text("Declared").foregroundStyle(MLTheme.muted)}
            Text(shapeText(planned?["out"])).fontWeight(.semibold).monospacedDigit().foregroundStyle(accent).frame(maxWidth:.infinity).padding(9).background(accent.opacity(0.045)).overlay(Rectangle().stroke(accent.opacity(0.4)))
        }
    }
    private func shapeText(_ layout:BatchedTensorLayout?)->String {layout.map {"["+$0.displayDimensions.joined(separator:", ")+"]"} ?? "Unresolved"}
    private var concatenateInputs:some View {
        VStack(alignment:.leading,spacing:0) {
            Text("Connected inputs").fontWeight(.semibold).foregroundStyle(accent).padding(.bottom,6)
            HStack {Text("Port").frame(width:30,alignment:.leading);Text("Component").frame(maxWidth:.infinity,alignment:.leading);Text("Declared shape").frame(width:115,alignment:.leading)}.foregroundStyle(MLTheme.muted).padding(7).background(accent.opacity(0.08))
            ForEach(["a","b"],id:\.self) {port in
                HStack(alignment:.top) {Text(port).frame(width:30,alignment:.leading);Text(inputLabels[port] ?? "Not connected").frame(maxWidth:.infinity,alignment:.leading).fixedSize(horizontal:false,vertical:true);Text(shapeText(inputLayouts[port])).monospacedDigit().frame(width:115,alignment:.leading).fixedSize(horizontal:false,vertical:true)}.padding(7).overlay(Rectangle().stroke(MLTheme.rule))
            }
        }
    }
    private var concatenateShape:some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {Text("Planned shape").fontWeight(.semibold).foregroundStyle(accent);Spacer();Text("Declared").foregroundStyle(MLTheme.muted)}
            Text(shapeText(planned?["out"])).fontWeight(.semibold).monospacedDigit().foregroundStyle(accent).frame(maxWidth:.infinity).padding(9).background(accent.opacity(0.045)).overlay(Rectangle().stroke(accent.opacity(0.4)))
            if let a=inputLayouts["a"],let b=inputLayouts["b"],let dim=candidate?["dim"]?.integerValue,
               planned != nil {
                let selected=(dim<0 ? dim+a.rank:dim)-1
                let first=a.sample.dimensions[selected],second=b.sample.dimensions[selected]
                GeometryReader {geometry in
                    HStack(spacing:0) {
                        Text("a · \(first)").lineLimit(1).minimumScaleFactor(0.6).clipped().frame(width:geometry.size.width*Double(first)/Double(first+second),height:28).background(accent.opacity(0.25))
                        Text("b · \(second)").lineLimit(1).minimumScaleFactor(0.6).frame(maxWidth:.infinity).frame(height:28).background(accent.opacity(0.12))
                    }.foregroundStyle(accent).monospacedDigit()
                }.frame(height:28)
            }
        }
    }
    private var axisOrder:some View {
        VStack(alignment:.leading,spacing:7) {
            Text("Sample axis order").fontWeight(.semibold).foregroundStyle(accent)
            HStack {Text("Output").frame(width:70,alignment:.leading);Text("Input axis");Spacer()}.foregroundStyle(MLTheme.muted)
            HStack {Text("0").frame(width:70,alignment:.leading);Label("Batch N",systemImage:"lock.fill");Spacer()}.padding(7).background(accent.opacity(0.06))
            ForEach($draft.order) {$row in
                let index=draft.order.firstIndex {$0.id==row.id} ?? 0
                HStack(spacing:6) {
                    VStack(spacing:2) {
                        Button {draft.moveOrder(id:row.id,offset:-1)} label:{Image(systemName:"chevron.up")}.disabled(index==0).accessibilityLabel("Move axis row \(index+1) up")
                        Button {draft.moveOrder(id:row.id,offset:1)} label:{Image(systemName:"chevron.down")}.disabled(index==draft.order.count-1).accessibilityLabel("Move axis row \(index+1) down")
                    }.font(.system(size:8)).buttonStyle(.plain).frame(width:15)
                    Text(String(index+1)).frame(width:35,alignment:.leading)
                    TextField("Input axis",text:$row.value).flatField().accessibilityLabel("Input axis for output \(index+1)")
                    Button {draft.order.removeAll {$0.id==row.id}} label:{Image(systemName:"trash")}.buttonStyle(.plain).disabled(draft.order.count<=1).accessibilityLabel("Remove axis row \(index+1)")
                }.padding(7).background(accent.opacity(0.035)).overlay(Rectangle().stroke(MLTheme.rule))
            }
            Button("Add sample axis") {draft.order.append(.init(value:String(draft.order.count+1)))}.buttonStyle(FlatButtonStyle()).disabled(draft.order.count>=7 || inputLayouts["in"].map {draft.order.count >= $0.sample.dimensions.count} == true)
        }.disabled(isBusy || boundFields.contains("dims"))
    }
    private var sectionSizes:some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {Text("Part").frame(width:45,alignment:.leading);Text("Size");Spacer()}.foregroundStyle(MLTheme.muted)
            ForEach($draft.sections) {$row in
                let index=draft.sections.firstIndex {$0.id==row.id} ?? 0
                HStack(spacing:6) {
                    VStack(spacing:2) {
                        Button {draft.moveSection(id:row.id,offset:-1)} label:{Image(systemName:"chevron.up")}.disabled(index==0).accessibilityLabel("Move section \(index+1) up")
                        Button {draft.moveSection(id:row.id,offset:1)} label:{Image(systemName:"chevron.down")}.disabled(index==draft.sections.count-1).accessibilityLabel("Move section \(index+1) down")
                    }.font(.system(size:8)).buttonStyle(.plain).frame(width:15)
                    Text(String(index+1)).frame(width:24,alignment:.leading)
                    TextField("Size",text:$row.value).flatField().accessibilityLabel("Part \(index+1) size")
                    Button {draft.sections.removeAll {$0.id==row.id}} label:{Image(systemName:"trash")}.buttonStyle(.plain).disabled(draft.sections.count<=1).accessibilityLabel("Remove part \(index+1)")
                }.padding(7).background(accent.opacity(0.035)).overlay(Rectangle().stroke(MLTheme.rule))
            }
            Button("Add section") {draft.sections.append(.init(value:"1"))}.buttonStyle(FlatButtonStyle()).disabled(draft.sections.count>=8)
        }.disabled(isBusy || boundFields.contains("sections"))
    }
    private var equationEditor:some View {
        VStack(alignment:.leading,spacing:7) {
            Text("Arrangement")
            FlatChoiceControl(title:"Arrangement",selection:draft.pattern,options:[.init(id:"multiply",title:"Matrix product"),.init(id:"outer",title:"Outer product"),.init(id:"diagonal",title:"Diagonal"),.init(id:"reorder",title:"Reorder axes"),.init(id:"custom",title:"Custom labels")],accent:accent) {draft.pattern=$0;rejection=nil}.disabled(isBusy || boundFields.contains("pattern") || boundFields.contains("equation"))
            if draft.pattern=="custom" {
                equationField("Input A sample labels",value:$draft.aLabels)
                Toggle("Use input B",isOn:$draft.hasB)
                if draft.hasB {equationField("Input B sample labels",value:$draft.bLabels)}
                equationField("Output sample labels",value:$draft.outputLabels)
            }
            Text(candidate?["equation"]?.stringValue ?? "Invalid arrangement").font(.system(.caption,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(8).background(accent.opacity(0.06))
        }.disabled(isBusy || boundFields.contains("equation"))
    }
    private func equationField(_ title:String,value:Binding<String>)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);HStack(spacing:6) {Text("n").foregroundStyle(accent);TextField(title,text:value).flatField().accessibilityLabel(title)}}
    }
    private var patchGeometry:some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {Text("Geometry").frame(maxWidth:.infinity,alignment:.leading);Text("Height").frame(width:70);Text("Width").frame(width:70)}.foregroundStyle(MLTheme.muted)
            if draft.type=="fold" {geometryRow("Output size",key:"output_size")}
            geometryRow("Patch size",key:"kernel_size");geometryRow("Stride",key:"stride");geometryRow("Dilation",key:"dilation");geometryRow("Padding",key:"padding")
        }
    }
    private func geometryRow(_ title:String,key:String)->some View {
        HStack(spacing:6) {Text(title).frame(maxWidth:.infinity,alignment:.leading);TextField("Height",text:textBinding(key+".0")).flatField().frame(width:70).accessibilityLabel(title+" height");TextField("Width",text:textBinding(key+".1")).flatField().frame(width:70).accessibilityLabel(title+" width")}.padding(7).background(accent.opacity(0.035)).overlay(Rectangle().stroke(MLTheme.rule)).disabled(isBusy || boundFields.contains(key))
    }
    private var plannedShape:some View {
        VStack(alignment:.leading,spacing:6) {
            Text("Planned shapes").fontWeight(.semibold).foregroundStyle(accent)
            ForEach(ports,id:\.self) {port in shapeRow(port.capitalized,layout:inputLayouts[port])}
            if let planned {
                ForEach(planned.keys.sorted(),id:\.self) {port in shapeRow(port.hasPrefix("part") ? port:"Output",layout:planned[port])}
                if family == .parts {Text("\(planned.count) actual parts").foregroundStyle(accent)}
            } else {Label(hasInputs ? "Output unresolved":"Inputs unresolved",systemImage:hasInputs ? "exclamationmark.circle":"questionmark.circle").foregroundStyle(MLTheme.muted)}
        }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background(accent.opacity(0.04)).overlay(Rectangle().stroke(accent.opacity(0.25)))
    }
    private func shapeRow(_ title:String,layout:BatchedTensorLayout?)->some View {
        HStack(alignment:.top) {Text(title).frame(width:48,alignment:.leading);Text(layout?.displayDimensions.joined(separator:" × ") ?? "Unresolved").monospacedDigit().fixedSize(horizontal:false,vertical:true)}
    }
    private func reload() {
        baselineType=node.type;baselineSource=node.config;loadFailed=false
        do {let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:configurationVersion,registry:registry);draft=try .init(type:node.type,values:values);baseline=try TensorOperationAuthoring.canonical(node.type,values);customAxes=[];rejection=nil;onPending(false)}
        catch {loadFailed=true;rejection=error.localizedDescription;onPending(false)}
    }
    private func apply() {
        guard !isBusy,changed,let values=candidate,problem==nil else {return}
        guard onApply(draft.type,values,baselineType,baselineSource) else {rejection="Settings were not applied. Check connections, source selection and containing block bindings.";return}
        baselineType=draft.type;baselineSource=values;baseline=values;rejection=nil;onPending(false)
    }
}
