import SwiftUI
import KwipMLCore

struct AggregateConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var original:[String:JSONValue]=[:]
    @State private var groups:[String]=[]
    @State private var reducers:[SummaryReducerDraft]=[]
    @State private var nullKeys="group"
    @State private var maxRows="100000"
    @State private var search=""
    @State private var origin=""
    @State private var selectedGroup:Int?
    @State private var selectedSummary:UUID?
    @State private var dragOrigin:Int?
    @State private var groupDragOrigin:Int?
    @State private var error:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var schema:[DataColumnSchema] {context.upstreamSchema}
    private var configuration:[String:JSONValue] {
        var result:[String:JSONValue]=["groupBy":.array(groups.map(JSONValue.string)),"nullKeys":.string(nullKeys),"aggregations":.array(reducers.map(\.configuration))]
        result["maxRows"] = Int64(maxRows).map(JSONValue.integer);return result
    }
    private var changed:Bool {configuration != original || maxRows != number(original["maxRows"])}
    private var selectedIndex:Int? {reducers.firstIndex {$0.id == selectedSummary}}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Label("Input",systemImage:"tablecells");Spacer();Text(schema.isEmpty ? "Unresolved":"\(schema.count) columns")}.padding(8).foregroundStyle(accent).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.3)))
            groupGrid
            choice("Missing group keys",selection:$nullKeys,options:[("group","Group together"),("drop","Drop rows"),("error","Stop run")])
            summaryGrid
            if let index=selectedIndex {selectedControls(index)}
            VStack(alignment:.leading,spacing:4) {Text("Maximum output groups");TextField("Maximum output groups",text:$maxRows).flatField().accessibilityLabel("Maximum output groups")}
            Text("\(groups.count) group columns · \(reducers.count) summaries").font(.caption2).foregroundStyle(MLTheme.muted)
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!changed)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Spacer();if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Summarize changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
        .onChange(of:node.id) {_,_ in reload()}
        .onChange(of:node.config) {_,next in if next == original {return};if !changed {reload()} else {error="Saved settings changed. Revert to load them; your draft is preserved."}}
        .onChange(of:configuration) {_,_ in publish()}.onChange(of:maxRows) {_,_ in publish()}
        .onDisappear {context.setNumericPending(origin:origin,key:"aggregateDraft",pending:false)}
    }
    private var groupGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Group by").fontWeight(.semibold).foregroundStyle(accent)
            VStack(spacing:0) {
                HStack {Text("Column").frame(maxWidth:.infinity,alignment:.leading);Text("Type").frame(width:72,alignment:.leading);Color.clear.frame(width:20)}.font(.system(size:10,weight:.semibold)).padding(7).background(MLTheme.panel)
                if groups.isEmpty {Text("Global summary").foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading).padding(9)}
                ForEach(groups.indices,id:\.self) {index in
                    HStack(spacing:6) {
                        let name=groups[index]
                        Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:16,height:28).contentShape(Rectangle()).gesture(DragGesture(minimumDistance:4,coordinateSpace:.global).onChanged {value in
                            guard context.isCurrent,!context.isBusy,!name.isEmpty,groups.filter({$0 == name}).count == 1,let current=groups.firstIndex(of:name) else {return}
                            if groupDragOrigin == nil {groupDragOrigin=current}
                            let target=min(groups.count-1,max(0,(groupDragOrigin ?? current)+Int((value.translation.height/42).rounded())))
                            let moved=groups.remove(at:current);groups.insert(moved,at:target);selectedGroup=target
                        }.onEnded {_ in groupDragOrigin=nil}).accessibilityLabel("Drag group \(name) to reorder")
                        columnChoice("Grouping column \(index+1)",selection:Binding(get:{groups.indices.contains(index) ? groups[index]:""},set:{if groups.indices.contains(index) {groups[index]=$0;selectedGroup=index}}),columns:schema.filter {$0.dtype != "image"})
                        Text(schema.first {$0.name == groups[index]}?.dtype ?? "Unresolved").font(.system(size:10)).foregroundStyle(MLTheme.muted).frame(width:72,alignment:.leading)
                        Button {groups.remove(at:index);selectedGroup=nil} label:{Image(systemName:"minus")}.buttonStyle(.plain).frame(width:20).accessibilityLabel("Remove grouping column \(index+1)")
                    }.padding(7).background(selectedGroup == index ? accent.opacity(0.08):Color.clear).contentShape(Rectangle()).onTapGesture {selectedGroup=index}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                }
                HStack {
                    Button("Add group") {groups.append("");selectedGroup=groups.count-1}.disabled(groups.count+reducers.count>=128)
                    Spacer()
                    Button {moveGroup(-1)} label:{Image(systemName:"arrow.up")}.disabled(selectedGroup == nil || selectedGroup == 0).accessibilityLabel("Move selected grouping column earlier")
                    Button {moveGroup(1)} label:{Image(systemName:"arrow.down")}.disabled(selectedGroup == nil || selectedGroup == groups.count-1).accessibilityLabel("Move selected grouping column later")
                }.buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private var filtered:[SummaryReducerDraft] {reducers.filter {search.isEmpty || $0.output.localizedCaseInsensitiveContains(search) || $0.column.localizedCaseInsensitiveContains(search) || title($0.op).localizedCaseInsensitiveContains(search)}}
    private var summaryGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Aggregations").fontWeight(.semibold).foregroundStyle(accent)
            TextField("Find summary",text:$search).flatField().accessibilityLabel("Find summary")
            VStack(spacing:0) {
                ThemedScrollView(.horizontal) {
                    VStack(spacing:0) {
                        HStack(spacing:0) {heading("",20);heading("Column",74);heading("Operation",96);heading("Output name",100);heading("",22)}.padding(.horizontal,4).background(MLTheme.panel)
                        ThemedScrollView {
                            LazyVStack(spacing:0) {
                                ForEach(filtered) {reducer in summaryRow(reducer)}
                                if filtered.isEmpty {Text(reducers.isEmpty ? "Add a summary":"No matching summaries").foregroundStyle(MLTheme.muted).padding(10)}
                            }
                        }.frame(height:min(250,max(72,CGFloat(filtered.count)*40)))
                    }.frame(width:320)
                }
                HStack {
                    Button("Add summary") {let reducer=SummaryReducerDraft();reducers.append(reducer);selectedSummary=reducer.id;search=""}.disabled(groups.count+reducers.count>=128)
                    Spacer();Text("\(reducers.count) summaries").foregroundStyle(MLTheme.muted)
                    Button {moveSummary(-1)} label:{Image(systemName:"arrow.up")}.disabled(selectedIndex == nil || selectedIndex == 0).accessibilityLabel("Move selected summary earlier")
                    Button {moveSummary(1)} label:{Image(systemName:"arrow.down")}.disabled(selectedIndex == nil || selectedIndex == reducers.count-1).accessibilityLabel("Move selected summary later")
                }.font(.system(size:10)).buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private func heading(_ text:String,_ width:CGFloat)->some View {Text(text).font(.system(size:9,weight:.semibold)).frame(width:width,height:26,alignment:.leading)}
    private func summaryRow(_ reducer:SummaryReducerDraft)->some View {
        HStack(spacing:0) {
            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:20,height:36).contentShape(Rectangle()).gesture(DragGesture(minimumDistance:4,coordinateSpace:.global).onChanged {value in
                guard context.isCurrent,!context.isBusy,let index=reducers.firstIndex(where:{$0.id == reducer.id}) else {return}
                if dragOrigin == nil {dragOrigin=index};selectedSummary=reducer.id
                let target=min(reducers.count-1,max(0,(dragOrigin ?? index)+Int((value.translation.height/40).rounded())))
                let row=reducers.remove(at:index);reducers.insert(row,at:target)
            }.onEnded {_ in dragOrigin=nil}).accessibilityLabel("Drag summary \(reducer.output) to reorder")
            Group {
                if reducer.op == "countRows" {Text("All rows").foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,alignment:.leading).padding(.horizontal,5)}
                else {columnChoice("Source for summary \(summaryNumber(reducer.id))",selection:binding(reducer.id,\.column),columns:reducer.eligible(schema))}
            }.frame(width:70).padding(.trailing,4)
            FlatChoiceControl(title:"Operation for summary \(summaryNumber(reducer.id))",selection:reducer.op,options:SummaryReducerDraft.operations.map {.init(id:$0.0,title:$0.1)},compact:true,accent:accent) {value in update(reducer.id) {$0.setOperation(value,schema:schema)}}.frame(width:92).padding(.trailing,4)
            TextField("Output name",text:binding(reducer.id,\.output)).flatField().frame(width:96).padding(.trailing,4).accessibilityLabel("Output name for summary \(summaryNumber(reducer.id))")
            Button {reducers.removeAll {$0.id == reducer.id};if selectedSummary == reducer.id {selectedSummary=reducers.first?.id}} label:{Image(systemName:"minus")}.buttonStyle(.plain).frame(width:22).accessibilityLabel("Remove summary \(summaryNumber(reducer.id))")
        }.font(.system(size:10)).padding(.horizontal,4).frame(height:40).background(selectedSummary == reducer.id ? accent.opacity(0.1):MLTheme.background).contentShape(Rectangle()).onTapGesture {selectedSummary=reducer.id}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    @ViewBuilder private func selectedControls(_ index:Int)->some View {
        let row=reducers[index]
        VStack(alignment:.leading,spacing:8) {
            Text("Selected summary · \(row.output.isEmpty ? "Unnamed":row.output)").fontWeight(.semibold).foregroundStyle(accent).lineLimit(2)
            if !["countRows","countNonNull"].contains(row.op) {choice("Missing values",selection:binding(row.id,\.nullPolicy),options:[("skip","Skip missing values"),("propagate","Return missing"),("error","Stop run")])}
            if row.op == "joinText" {
                VStack(alignment:.leading,spacing:4) {
                    Text("Separator")
                    FlatChoiceControl(title:"Text separator",selection:row.separatorChoice,options:[("newline","Newline"),("commaSpace","Comma + space"),("comma","Comma"),("space","Space"),("tab","Tab"),("none","None"),("custom","Custom")].map {.init(id:$0.0,title:$0.1)},accent:accent) {value in update(row.id) {$0.setSeparatorChoice(value)}}
                    if row.separatorChoice == "custom" {TextField("Custom separator",text:binding(row.id,\.separator)).flatField().accessibilityLabel("Custom text separator")}
                }
                Toggle("Remove duplicate text",isOn:binding(row.id,\.distinct)).toggleStyle(FileInputCheckboxStyle(tint:accent))
                Toggle("Skip empty text",isOn:binding(row.id,\.skipEmpty)).toggleStyle(FileInputCheckboxStyle(tint:accent))
            }
            HStack {Text("Result type");Spacer();Text(resultType(row)).foregroundStyle(accent)}.padding(8).background(accent.opacity(0.06)).overlay(Rectangle().stroke(accent.opacity(0.3)))
        }.padding(.vertical,3)
    }
    private func resultType(_ row:SummaryReducerDraft)->String {
        if row.op == "sumDecimal" {return "Exact decimal text"}
        if ["countRows","countNonNull"].contains(row.op) {return "Integer"};if row.op == "mean" {return "Number"};if row.op == "joinText" {return "Text"}
        switch schema.first(where:{$0.name == row.column})?.dtype {case "int64":return "Integer";case "float64":return "Number";case "string":return "Text";case "bool":return "Boolean";default:return "Unresolved"}
    }
    private func summaryNumber(_ id:UUID)->Int {(reducers.firstIndex {$0.id == id} ?? 0)+1}
    private func title(_ op:String)->String {SummaryReducerDraft.operations.first {$0.0 == op}?.1 ?? op}
    private func update(_ id:UUID,_ change:(inout SummaryReducerDraft)->Void) {if let index=reducers.firstIndex(where:{$0.id == id}) {change(&reducers[index]);selectedSummary=id}}
    private func binding<T>(_ id:UUID,_ path:WritableKeyPath<SummaryReducerDraft,T>)->Binding<T> {
        let fallback=reducers.first {$0.id == id}![keyPath:path]
        return Binding(get:{reducers.first {$0.id == id}.map {$0[keyPath:path]} ?? fallback},set:{value in update(id) {$0[keyPath:path]=value}})
    }
    private func choice(_ title:String,selection:Binding<String>,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:selection.wrappedValue,options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {selection.wrappedValue=$0}}}
    @ViewBuilder private func columnChoice(_ title:String,selection:Binding<String>,columns:[DataColumnSchema])->some View {
        if schema.isEmpty {TextField("Exact column",text:selection).flatField().accessibilityLabel(title)}
        else {FlatChoiceControl(title:title,selection:selection.wrappedValue,options:columns.map {.init(id:$0.name,title:$0.name)},placeholder:"Column",compact:true,accent:accent) {selection.wrappedValue=$0}}
    }
    private func moveGroup(_ delta:Int) {guard let i=selectedGroup,groups.indices.contains(i+delta) else {return};groups.swapAt(i,i+delta);selectedGroup=i+delta}
    private func moveSummary(_ delta:Int) {guard let i=selectedIndex,reducers.indices.contains(i+delta) else {return};reducers.swapAt(i,i+delta)}
    private func number(_ value:JSONValue?)->String {if case .integer(let value)=value {return String(value)};return ""}
    private func publish() {context.setNumericPending(origin:origin,key:"aggregateDraft",pending:changed)}
    private func reload() {
        context.setNumericPending(origin:origin,key:"aggregateDraft",pending:false);original=node.config
        if case .array(let values)=original["groupBy"] {groups=values.compactMap {if case .string(let name)=$0 {return name};return nil}} else {groups=[]}
        do {guard case .array(let values)=original["aggregations"] else {throw DataFlowContractError("Saved summaries are unavailable.")};reducers=try values.map {try SummaryReducerDraft.decode($0)};error=nil} catch {reducers=[];self.error=error.localizedDescription}
        if case .string(let value)=original["nullKeys"] {nullKeys=value}
        maxRows=number(original["maxRows"]);selectedSummary=reducers.first?.id;selectedGroup=nil;origin=context.numericOrigin(key:"aggregateDraft");publish()
    }
    private func apply() {if context.applyTableConfiguration(configuration,expected:original) {original=configuration;error=nil;publish()} else {error=context.statusMessage}}
}
