import SwiftUI
import KwipMLCore

struct JoinConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var saved:[String:JSONValue]=[:]
    @State private var leftKeys:[String]=[]
    @State private var rightKeys:[String]=[]
    @State private var rows:[JoinOutputColumn]=[]
    @State private var maximumRows=""
    @State private var search=""
    @State private var selectedRow:String?
    @State private var selectedPair:Int?
    @State private var dragOrigin:Int?
    @State private var error:String?
    @State private var origin=""
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var left:[DataColumnSchema] {context.upstreamSchema(forInput:"left")}
    private var right:[DataColumnSchema] {context.upstreamSchema(forInput:"right")}
    private var changed:Bool {configuration != saved}
    private var defaults:[JoinOutputColumn] {JoinOutputColumn.defaults(left:left,right:right,leftPrefix:text(draft["leftPrefix"]),rightPrefix:text(draft["rightPrefix"]))}
    private var configuration:[String:JSONValue] {
        var next=draft
        next["leftKeys"] = .array(leftKeys.map(JSONValue.string));next["rightKeys"] = .array(rightKeys.map(JSONValue.string))
        next["maxRows"] = Int64(maximumRows).map(JSONValue.integer)
        next["outputColumns"] = .array(rows.map(\.json))
        return next
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack(spacing:6) {sourceBadge("Left",schema:left);sourceBadge("Right",schema:right)}
            choice("Join mode",key:"mode",options:[("inner","Inner · matching rows"),("left","Left · keep all left rows"),("right","Right · keep all right rows"),("full","Full · keep both sides")])
            keyGrid
            outputGrid
            choice("Missing keys",key:"nullPolicy",options:[("neverMatch","Never match"),("match","Match missing keys"),("error","Stop run")])
            choice("Key relationship",key:"cardinality",options:[("oneToOne","Both keys unique"),("oneToMany","Left keys unique"),("manyToOne","Right keys unique"),("manyToMany","Both sides may repeat")])
            choice("New source columns",key:"newColumns",options:[("keep","Include unchanged"),("drop","Exclude"),("error","Require grid choices")])
            Text("Name collisions").fontWeight(.semibold).foregroundStyle(accent)
            prefixField("Left prefix",key:"leftPrefix");prefixField("Right prefix",key:"rightPrefix")
            VStack(alignment:.leading,spacing:4) {Text("Maximum output rows");TextField("Maximum output rows",text:$maximumRows).flatField().accessibilityLabel("Maximum output rows")}
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!changed)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Spacer();if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Join changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
        .onChange(of:node.id) {_,_ in reload()}
        .onChange(of:node.config) {_,next in if next == original {return};if !changed {reload()} else {error="Saved configuration changed. Revert to load it; your draft is preserved."}}
        .onChange(of:left) {_,_ in reconcile()}.onChange(of:right) {_,_ in reconcile()}
        .onChange(of:configuration) {_,_ in context.setNumericPending(origin:origin,key:"joinDraft",pending:changed)}
        .onDisappear {context.setNumericPending(origin:origin,key:"joinDraft",pending:false)}
    }
    private func sourceBadge(_ title:String,schema:[DataColumnSchema])->some View {
        HStack(spacing:4) {Text(title).fontWeight(.semibold);Text(schema.isEmpty ? "· Unresolved":"· \(schema.count) columns")}.frame(maxWidth:.infinity,alignment:.leading).padding(8).foregroundStyle(accent).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.3)))
    }
    private var keyGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Matching keys").fontWeight(.semibold).foregroundStyle(accent)
            VStack(spacing:0) {
                HStack(spacing:6) {Text("Left column").frame(maxWidth:.infinity,alignment:.leading);Text("Right column").frame(maxWidth:.infinity,alignment:.leading);Color.clear.frame(width:20)}.font(.system(size:10,weight:.semibold)).padding(7).background(MLTheme.panel)
                ForEach(leftKeys.indices,id:\.self) {index in
                    HStack(alignment:.top,spacing:6) {
                        keyField(side:"Left",index:index,schema:left,values:$leftKeys)
                        keyField(side:"Right",index:index,schema:right,values:$rightKeys)
                        Button {leftKeys.remove(at:index);rightKeys.remove(at:index);selectedPair=nil} label:{Image(systemName:"minus")}.buttonStyle(.plain).frame(width:20,height:28).accessibilityLabel("Remove key pair \(index+1)")
                    }.padding(7).background(selectedPair == index ? accent.opacity(0.08):Color.clear).contentShape(Rectangle()).onTapGesture {selectedPair=index}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                }
                HStack {
                    Button("Add key pair") {leftKeys.append("");rightKeys.append("");selectedPair=leftKeys.count-1}.disabled(leftKeys.count>=128)
                    Spacer()
                    Button {movePair(-1)} label:{Image(systemName:"arrow.up")}.disabled(selectedPair == nil || selectedPair == 0).accessibilityLabel("Move selected key pair earlier")
                    Button {movePair(1)} label:{Image(systemName:"arrow.down")}.disabled(selectedPair == nil || selectedPair == leftKeys.count-1).accessibilityLabel("Move selected key pair later")
                }.buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private var filtered:[JoinOutputColumn] {rows.filter {search.isEmpty || $0.source.localizedCaseInsensitiveContains(search) || $0.name.localizedCaseInsensitiveContains(search) || $0.side.localizedCaseInsensitiveContains(search)}}
    private var outputGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Output columns").fontWeight(.semibold).foregroundStyle(accent)
            TextField("Find column",text:$search).flatField().accessibilityLabel("Find Join output column")
            VStack(spacing:0) {
                ThemedScrollView(.horizontal) {
                    VStack(spacing:0) {
                        HStack(spacing:0) {heading("",20);heading("Keep",30);heading("From",28);heading("Column",104);heading("Output name",138)}.padding(.horizontal,4).background(MLTheme.panel)
                        ThemedScrollView {
                            LazyVStack(spacing:0) {
                                ForEach(filtered) {row in outputRow(row)}
                                if filtered.isEmpty {Text(rows.isEmpty ? "Connect tables to choose columns":"No matching columns").foregroundStyle(MLTheme.muted).padding(10)}
                            }
                        }.frame(height:min(240,max(72,CGFloat(filtered.count)*42)))
                    }.frame(width:328)
                }
                HStack(spacing:5) {
                    Button("Select all") {for i in rows.indices {rows[i].included=true}}
                    Button("Clear") {for i in rows.indices {rows[i].included=false}}
                    Spacer(minLength:0)
                    Button {moveRow(-1)} label:{Image(systemName:"arrow.up")}.disabled(selectedIndex == nil || selectedIndex == 0).accessibilityLabel("Move selected output column earlier")
                    Button {moveRow(1)} label:{Image(systemName:"arrow.down")}.disabled(selectedIndex == nil || selectedIndex == rows.count-1).accessibilityLabel("Move selected output column later")
                    Text("\(rows.filter(\.included).count) / \(rows.count) kept").foregroundStyle(MLTheme.muted)
                }.font(.system(size:10)).buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private func heading(_ title:String,_ width:CGFloat)->some View {Text(title).font(.system(size:9,weight:.semibold)).frame(width:width,height:26,alignment:.leading)}
    private func outputRow(_ row:JoinOutputColumn)->some View {
        let schema=row.side == "left" ? left:right
        let source=schema.first {$0.name == row.source}
        let missing = !schema.isEmpty && source == nil
        return HStack(spacing:0) {
            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:20,height:40).contentShape(Rectangle()).gesture(DragGesture(minimumDistance:4,coordinateSpace:.global).onChanged {value in
                guard context.isCurrent,!context.isBusy,let index=rows.firstIndex(where:{$0.id == row.id}) else {return}
                if dragOrigin == nil {dragOrigin=index};selectedRow=row.id
                let target=min(rows.count-1,max(0,(dragOrigin ?? index)+Int((value.translation.height/42).rounded())))
                let moved=rows.remove(at:index);rows.insert(moved,at:target)
            }.onEnded {_ in dragOrigin=nil}).accessibilityLabel("Drag \(row.side) \(row.source) to reorder")
            Toggle("",isOn:Binding(get:{row.included},set:{value in update(row.id) {$0.included=value}})).toggleStyle(FileInputCheckboxStyle(tint:accent)).accessibilityLabel("Keep \(row.side) \(row.source)").frame(width:30)
            Text(row.side == "left" ? "L":"R").foregroundStyle(accent).frame(width:28)
            VStack(alignment:.leading,spacing:2) {Text(row.source).lineLimit(1).truncationMode(.middle);Text(missing ? "Missing":(source?.dtype ?? "Unresolved")).font(.system(size:9)).foregroundStyle(missing ? MLTheme.rose:MLTheme.muted)}.frame(width:100,alignment:.leading).padding(.trailing,4).help(row.source)
            HStack(spacing:2) {
                TextField("Output name",text:Binding(get:{row.name},set:{value in update(row.id) {$0.name=value}})).flatField().disabled(!row.included).accessibilityLabel("Output name for \(row.side) \(row.source)")
                if missing {Button {rows.removeAll {$0.id == row.id}} label:{Image(systemName:"xmark.circle")}.buttonStyle(.plain).foregroundStyle(MLTheme.rose).accessibilityLabel("Remove missing \(row.side) \(row.source)")}
            }.frame(width:138)
        }.font(.system(size:10)).padding(.horizontal,4).frame(height:42).background(selectedRow == row.id ? accent.opacity(0.1):row.included ? MLTheme.background:MLTheme.panel).contentShape(Rectangle()).onTapGesture {selectedRow=row.id}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    @ViewBuilder private func keyField(side:String,index:Int,schema:[DataColumnSchema],values:Binding<[String]>)->some View {
        VStack(alignment:.leading,spacing:3) {
            if schema.isEmpty {TextField("Exact column",text:Binding(get:{values.wrappedValue[index]},set:{values.wrappedValue[index]=$0;selectedPair=index})).flatField().accessibilityLabel("\(side) key for pair \(index+1)")}
            else {FlatChoiceControl(title:"\(side) key for pair \(index+1)",selection:values.wrappedValue[index],options:schema.filter {$0.dtype != "image"}.map {FlatChoiceOption(id:$0.name,title:$0.name)},placeholder:"Choose column",compact:true,accent:accent) {values.wrappedValue[index]=$0;selectedPair=index}}
            Text(schema.first {$0.name == values.wrappedValue[index]}?.dtype ?? "Unresolved").font(.system(size:9)).foregroundStyle(MLTheme.muted)
        }.frame(maxWidth:.infinity)
    }
    private func choice(_ title:String,key:String,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:text(draft[key]),options:options.map {FlatChoiceOption(id:$0.0,title:$0.1)},accent:accent) {draft[key] = .string($0)}}
    }
    private func prefixField(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:Binding(get:{text(draft[key])},set:{value in
            let before=defaults;draft[key] = .string(value);let after=defaults
            for i in rows.indices {if let old=before.first(where:{$0.id == rows[i].id}),rows[i].name == old.name,let next=after.first(where:{$0.id == rows[i].id}) {rows[i].name=next.name}}
        })).flatField().accessibilityLabel(title)}
    }
    private var selectedIndex:Int? {rows.firstIndex {$0.id == selectedRow}}
    private func update(_ id:String,_ change:(inout JoinOutputColumn)->Void) {if let index=rows.firstIndex(where:{$0.id == id}) {change(&rows[index]);selectedRow=id}}
    private func moveRow(_ delta:Int) {guard let i=selectedIndex,rows.indices.contains(i+delta) else {return};rows.swapAt(i,i+delta)}
    private func movePair(_ delta:Int) {guard let i=selectedPair,leftKeys.indices.contains(i+delta) else {return};leftKeys.swapAt(i,i+delta);rightKeys.swapAt(i,i+delta);selectedPair=i+delta}
    private func text(_ value:JSONValue?)->String {if case .string(let value)=value {return value};return ""}
    private func number(_ value:JSONValue?)->String {if case .integer(let value)=value {return String(value)};return ""}
    private func names(_ value:JSONValue?)->[String] {if case .array(let values)=value {return values.map {text($0)}};return []}
    private func reconcile() {for row in defaults where !rows.contains(where:{$0.id == row.id}) {var next=row;next.included=draft["newColumns"] != .string("drop");rows.append(next)}}
    private func reload() {
        context.setNumericPending(origin:origin,key:"joinDraft",pending:false)
        original=node.config;draft=node.config;leftKeys=names(draft["leftKeys"]);rightKeys=names(draft["rightKeys"])
        while rightKeys.count<leftKeys.count {rightKeys.append("")};while leftKeys.count<rightKeys.count {leftKeys.append("")}
        maximumRows=number(draft["maxRows"]);draft["newColumns"] = draft["newColumns"] ?? .string("keep")
        do {rows=try draft["outputColumns"].map {try JoinOutputColumn.decode($0)} ?? defaults;error=nil} catch {rows=[];self.error=error.localizedDescription}
        saved=configuration;origin=context.numericOrigin(key:"joinDraft");selectedRow=nil;selectedPair=nil
    }
    private func apply() {
        if context.applyTableConfiguration(configuration,expected:original) {original=configuration;saved=configuration;error=nil;context.setNumericPending(origin:origin,key:"joinDraft",pending:false)}
        else {error=context.statusMessage}
    }
}
