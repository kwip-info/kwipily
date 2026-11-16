import SwiftUI
import KwipMLCore

struct ReshapeTableConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var scheme
    @State private var original:[String:JSONValue]=[:]
    @State private var draft=ReshapeTableConfiguration.defaults
    @State private var search=""
    @State private var origin=""
    @State private var selected:[String:String]=[:]
    @State private var selectedValue:Int?
    @State private var error:String?
    @State private var fillText=""
    @State private var fillType="int64"
    private var accent:Color {scheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var schema:[DataColumnSchema] {context.upstreamSchema(forInput:"table")}
    private var changed:Bool {draft != original}
    private func string(_ key:String)->String {if case .string(let s)=draft[key] {return s};return ""}
    private func list(_ key:String)->[String] {if case .array(let a)=draft[key] {return a.compactMap {if case .string(let s)=$0 {return s};return nil}};return []}
    private var fields:[[String:JSONValue]] {if case .array(let a)=draft["values"] {return a.compactMap {if case .object(let row)=$0 {return row};return nil}};return []}
    private func textBinding(_ key:String)->Binding<String> {Binding(get:{string(key)},set:{draft[key] = .string($0)})}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            choice("Mode",key:"mode",options:[("long","Long · columns to rows"),("wide","Wide · rows to columns")])
            TextField("Find column",text:$search).flatField().accessibilityLabel("Find Reshape Table column")
            if string("mode")=="long" {
                columnGrid("Identifier columns",key:"idColumns",ordered:false)
                columnGrid("Value columns · output order",key:"valueColumns",ordered:true)
                field("Field-name output",key:"fieldName");field("Value output",key:"valueName")
                choice("Types",key:"typePolicy",options:[("strict","Require same types"),("losslessNumeric","Checked numeric widening")])
                choice("Missing values",key:"missingValues",options:[("keep","Keep"),("drop","Drop emitted row")])
            } else {
                columnGrid("Row keys · output order",key:"rowKeys",ordered:true)
                sourcePicker("Column names from",selection:textBinding("columnKey"))
                valueGrid
                choice("Repeated key pairs",key:"duplicates",options:[("stop","Stop run"),("aggregate","Aggregate")])
                choice("Missing column names",key:"nullColumnNames",options:[("stop","Stop run"),("drop","Drop row"),("label","Use label")])
                if string("nullColumnNames")=="label" {field("Missing-name label",key:"nullLabel")}
                choice("Missing combinations",key:"missingCombinations",options:[("missing","Missing"),("fill","Fixed fill")])
                if string("missingCombinations")=="fill" {
                    VStack(alignment:.leading,spacing:4) {
                        Text("Fixed fill type")
                        FlatChoiceControl(title:"Fixed fill type",selection:fillType,options:[("string","Text"),("int64","Integer"),("float64","Number"),("bool","Boolean")].map {.init(id:$0.0,title:$0.1)},accent:accent) {fillType=$0;updateFill()}
                        TextField("Fixed fill",text:$fillText).flatField().onChange(of:fillText) {_,_ in updateFill()}
                    }
                }
                choice("Column order",key:"columnOrder",options:[("firstSeen","First seen"),("sorted","Sorted by key value")])
                VStack(alignment:.leading,spacing:4) {Text("Maximum output columns · includes row keys");TextField("1–128",text:Binding(get:{if case .integer(let n)=draft["maxColumns"] {return String(n)};return ""},set:{draft["maxColumns"] = Int64($0).map(JSONValue.integer) ?? .string($0)})).flatField()}
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).disabled(!changed)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Spacer();if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Reshape Table changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
        .onChange(of:node.id) {_,_ in reload()}
        .onChange(of:node.config) {_,next in if next != original {if !changed {reload()} else {error="Saved settings changed. Revert to reload; your draft is preserved."}}}
        .onChange(of:draft) {_,_ in publish()}
        .onDisappear {context.setNumericPending(origin:origin,key:"reshapeTableDraft",pending:false)}
    }
    private func choice(_ title:String,key:String,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(accent);FlatChoiceControl(title:title,selection:string(key),options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {draft[key] = .string($0)}}
    }
    private func field(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:textBinding(key)).flatField().accessibilityLabel(title)}}
    private func sourcePicker(_ title:String,selection:Binding<String>)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:selection.wrappedValue,options:[.init(id:"",title:"Choose column")]+schema.map {.init(id:$0.name,title:"\($0.name) · \($0.dtype)")},accent:accent) {selection.wrappedValue=$0}}
    }
    private func columnGrid(_ title:String,key:String,ordered:Bool)->some View {
        let chosen=list(key)
        let names=ordered ? chosen+schema.map(\.name).filter {!chosen.contains($0)}:schema.map(\.name)
        let visible=names.filter {search.isEmpty || $0.localizedCaseInsensitiveContains(search)}
        return VStack(alignment:.leading,spacing:4) {
            Text(title).fontWeight(.semibold).foregroundStyle(accent)
            VStack(spacing:0) {
                HStack {Text("Column");Spacer();Text("Type · inherited")}.font(.caption2).padding(7).background(MLTheme.panel)
                ThemedScrollView {
                    VStack(spacing:0) {ForEach(visible,id:\.self) {name in
                        HStack {
                            if ordered {Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).gesture(DragGesture(minimumDistance:5).onEnded {v in moveColumn(key,name:name,delta:Int((v.translation.height/32).rounded()))})}
                            Toggle(name,isOn:Binding(get:{list(key).contains(name)},set:{on in var a=list(key);if on {a.append(name)} else {a.removeAll {$0==name}};draft[key] = .array(a.map(JSONValue.string))})).toggleStyle(FileInputCheckboxStyle(tint:accent))
                            Spacer();Text(schema.first(where:{$0.name==name})?.dtype ?? "Unavailable").foregroundStyle(MLTheme.muted)
                        }.padding(7).frame(height:32).background(selected[key]==name ? accent.opacity(0.1):MLTheme.background).contentShape(Rectangle()).onTapGesture {selected[key]=name}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                    }}
                }.frame(height:min(180,max(32,CGFloat(visible.count)*32)))
                if ordered {
                    HStack {Spacer();Button {if let name=selected[key] {moveColumn(key,name:name,delta:-1)}} label:{Image(systemName:"arrow.up")};Button {if let name=selected[key] {moveColumn(key,name:name,delta:1)}} label:{Image(systemName:"arrow.down")}}.buttonStyle(FlatButtonStyle()).padding(4)
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private func moveColumn(_ key:String,name:String,delta:Int) {var a=list(key);guard let i=a.firstIndex(of:name),!a.isEmpty else {return};let target=max(0,min(a.count-1,i+delta));a.remove(at:i);a.insert(name,at:target);draft[key] = .array(a.map(JSONValue.string))}
    private var valueGrid:some View {
        VStack(alignment:.leading,spacing:4) {
            Text("Values · output order").fontWeight(.semibold).foregroundStyle(accent)
            ThemedScrollView(.horizontal) {
                VStack(spacing:0) {
                    HStack {Color.clear.frame(width:16);Text("Value column").frame(width:140);Text("Aggregation").frame(width:120);Text("Prefix").frame(width:110);Text("Missing values").frame(width:120);Color.clear.frame(width:25)}.font(.caption2).padding(6).background(MLTheme.panel)
                    ForEach(fields.indices,id:\.self) {i in
                        HStack {
                            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:16).gesture(DragGesture(minimumDistance:5).onEnded {value in selectedValue=i;moveValue(Int((value.translation.height/40).rounded()))}).accessibilityLabel("Reorder value row")
                            FlatChoiceControl(title:"Value column",selection:rowText(i,"column"),options:schema.map {.init(id:$0.name,title:"\($0.name) · \($0.dtype)")},accent:accent) {setRow(i,"column",$0)}.frame(width:140)
                            FlatChoiceControl(title:"Aggregation",selection:rowText(i,"aggregation"),options:allowedReducers(i).map {.init(id:$0,title:reducerTitle($0))},accent:accent) {setRow(i,"aggregation",$0)}.frame(width:120)
                            TextField("Prefix",text:Binding(get:{rowText(i,"prefix")},set:{setRow(i,"prefix",$0)})).flatField().frame(width:110)
                            FlatChoiceControl(title:"Missing values",selection:rowText(i,"nullPolicy"),options:[("skip","Skip"),("propagate","Return missing"),("error","Stop run")].map {.init(id:$0.0,title:$0.1)},accent:accent) {setRow(i,"nullPolicy",$0)}.frame(width:120)
                            Button {var a=fields;a.remove(at:i);saveFields(a);selectedValue=nil} label:{Image(systemName:"minus")}.buttonStyle(.plain).frame(width:25)
                        }.padding(6).background(selectedValue==i ? accent.opacity(0.1):MLTheme.background).contentShape(Rectangle()).onTapGesture {selectedValue=i}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                    }
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
            HStack {
                Button("Add value") {guard let first=schema.first else {return};var a=fields;a.append(["column":.string(first.name),"aggregation":.string("first"),"prefix":.string(first.name+"_"),"nullPolicy":.string("skip")]);saveFields(a);selectedValue=a.count-1}.disabled(fields.count>=128 || schema.isEmpty)
                Spacer();Button {moveValue(-1)} label:{Image(systemName:"arrow.up")};Button {moveValue(1)} label:{Image(systemName:"arrow.down")}
            }.buttonStyle(FlatButtonStyle())
        }
    }
    private func allowedReducers(_ index:Int)->[String] {
        let dtype=schema.first(where:{$0.name==rowText(index,"column")})?.dtype
        if dtype=="image" {return ["first","last","countRows","countNonNull"]}
        if dtype != "int64" && dtype != "float64" {return ReshapeTableConfiguration.reducers.filter {!["sum","mean"].contains($0)}}
        return ReshapeTableConfiguration.reducers
    }
    private func reducerTitle(_ op:String)->String {switch op {case "countRows":return "Count rows";case "countNonNull":return "Count nonmissing";case "min":return "Minimum";case "max":return "Maximum";default:return op.capitalized}}
    private func rowText(_ index:Int,_ key:String)->String {guard fields.indices.contains(index),case .string(let s)=fields[index][key] else {return ""};return s}
    private func saveFields(_ rows:[[String:JSONValue]]) {draft["values"] = .array(rows.map(JSONValue.object))}
    private func setRow(_ index:Int,_ key:String,_ value:String) {var a=fields;guard a.indices.contains(index) else {return};a[index][key] = .string(value);saveFields(a)}
    private func moveValue(_ delta:Int) {var a=fields;guard let i=selectedValue,a.indices.contains(i+delta) else {return};a.swapAt(i,i+delta);saveFields(a);selectedValue=i+delta}
    private func updateFill() {switch fillType {case "int64":draft["fillValue"] = Int64(fillText).map(JSONValue.integer) ?? .null;case "float64":draft["fillValue"] = Double(fillText).flatMap {$0.isFinite ? JSONValue.number($0):nil} ?? .null;case "bool":draft["fillValue"] = fillText=="true" ? .bool(true):fillText=="false" ? .bool(false):.null;default:draft["fillValue"] = .string(fillText)}}
    private func publish() {context.setNumericPending(origin:origin,key:"reshapeTableDraft",pending:changed)}
    private func reload() {
        context.setNumericPending(origin:origin,key:"reshapeTableDraft",pending:false);original=node.config;draft=node.config;selected=[:];selectedValue=nil;error=nil
        switch draft["fillValue"] {case .integer(let n):fillType="int64";fillText=String(n);case .number(let n):fillType="float64";fillText=String(n);case .bool(let b):fillType="bool";fillText=b ? "true":"false";case .string(let s):fillType="string";fillText=s;default:fillText=""}
        origin=context.numericOrigin(key:"reshapeTableDraft");publish()
    }
    private func apply() {
        do {_ = try ReshapeTableSchema.validate(draft,schema:schema)} catch {self.error=error.localizedDescription;return}
        if context.applyTableConfiguration(draft,expected:original) {original=draft;error=nil;publish()} else {error=context.statusMessage}
    }
}
