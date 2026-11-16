import SwiftUI
import KwipMLCore

struct SortConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    var body:some View {SortUniqueConfigurationEditor(context:context,node:node,isSort:true)}
}
struct UniqueConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    var body:some View {SortUniqueConfigurationEditor(context:context,node:node,isSort:false)}
}
private struct SortUniqueConfigurationEditor<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    let isSort:Bool
    @State private var original:[String:JSONValue]=[:]
    @State private var keys:[SortConfiguration.Key]=[]
    @State private var keep="first"
    @State private var missing="equal"
    @State private var comparison="exact"
    @State private var search=""
    @State private var selected:Int?
    @State private var origin=""
    @State private var error:String?
    @State private var dragOrigin:Int?
    @State private var draggedColumn:String?
    @Environment(\.colorScheme) private var colorScheme
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var schema:[DataColumnSchema] {context.upstreamSchema(forInput:"table")}
    private var available:[DataColumnSchema] {schema.filter {["string","int64","float64","bool","date","datetime","time"].contains($0.dtype)}}
    private var values:[String:JSONValue] {isSort ? ["keys":.array(keys.map(\.value))]:["keys":.array(keys.map { .string($0.column)}),"keep":.string(keep),"missing":.string(missing),"comparison":.string(comparison)]}
    private var changed:Bool {values != original}
    private var draftKey:String {isSort ? "sortDraft":"uniqueDraft"}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text(isSort ? "Sort keys":"Key columns").fontWeight(.semibold).foregroundStyle(accent)
            TextField("Find column",text:$search).flatField()
            if isSort {sortGrid} else {uniqueGrid}
            if !isSort {
                choice("Retain",selection:$keep,options:[("first","First"),("last","Last")])
                choice("Missing keys",selection:$missing,options:[("equal","Equal"),("distinct","Distinct rows"),("stop","Stop run")])
                choice("Text comparison",selection:$comparison,options:[("exact","Exact"),("casefold","Unicode casefold")])
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Spacer()
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
        .onChange(of:node.id) {_,_ in reload()}
        .onChange(of:node.config) {_,next in if next != original {if !changed {reload()} else {error="Saved settings changed. Revert to load them; your draft is preserved."}}}
        .onChange(of:values) {_,_ in publish()}
        .onDisappear {context.setNumericPending(origin:origin,key:draftKey,pending:false)}
    }
    private var sortGrid:some View {
        VStack(spacing:0) {
            ThemedScrollView(.horizontal) {
                VStack(spacing:0) {
                    HStack {Text("Order").frame(width:26);Text("Column").frame(width:120);Text("Type").frame(width:64);Text("Direction").frame(width:100);Text("Missing").frame(width:75);Text("Text comparison").frame(width:110);Color.clear.frame(width:20)}.font(.caption2).padding(6).background(MLTheme.panel)
                    ThemedScrollView {
                        VStack(spacing:0) {ForEach(keys.indices,id:\.self) {i in
                            if search.isEmpty || keys[i].column.localizedCaseInsensitiveContains(search) {
                                HStack {
                                    Image(systemName:"line.3.horizontal").frame(width:26).gesture(DragGesture(minimumDistance:4).onChanged {value in
                                        guard context.isCurrent,!context.isBusy else {return}
                                        if dragOrigin == nil {dragOrigin=i;draggedColumn=keys[i].column}
                                        guard let current=keys.firstIndex(where:{$0.column==draggedColumn}) else {return};selected=current
                                        let target=min(keys.count-1,max(0,(dragOrigin ?? i)+Int((value.translation.height/40).rounded())))
                                        if target != current {let key=keys.remove(at:current);keys.insert(key,at:target);selected=target}
                                    }.onEnded {_ in dragOrigin=nil;draggedColumn=nil})
                                    FlatChoiceControl(title:"Column",selection:keys[i].column,options:available.filter {col in col.name==keys[i].column || !keys.contains {$0.column==col.name}}.map {.init(id:$0.name,title:$0.name)},accent:accent) {keys[i].column=$0}.frame(width:120)
                                    Text(schema.first {$0.name==keys[i].column}?.dtype ?? "—").frame(width:64)
                                    picker("Direction",value:$keys[i].direction,options:[("ascending","Ascending"),("descending","Descending")]).frame(width:100)
                                    picker("Missing",value:$keys[i].missing,options:[("first","First"),("last","Last")]).frame(width:75)
                                    if schema.first(where:{$0.name==keys[i].column})?.dtype == "string" {picker("Text comparison",value:$keys[i].comparison,options:[("exact","Exact"),("casefold","Casefold")]).frame(width:110)} else {Color.clear.frame(width:110)}
                                    Button {keys.remove(at:i);selected=nil} label:{Image(systemName:"minus")}.buttonStyle(.plain).frame(width:20)
                                }.padding(6).frame(height:40).background(selected==i ? accent.opacity(0.1):MLTheme.background).onTapGesture {selected=i}
                            }
                        }}
                    }.frame(height:min(240,max(40,CGFloat(keys.count)*40)))
                }.frame(width:675)
            }
            HStack {
                Button("Add key") {if let next=available.first(where:{col in !keys.contains {$0.column==col.name}}) {keys.append(.init(column:next.name));selected=keys.count-1}}.disabled(keys.count>=available.count)
                Spacer()
                Button {move(-1)} label:{Image(systemName:"arrow.up")}.disabled(selected == nil || selected==0)
                Button {move(1)} label:{Image(systemName:"arrow.down")}.disabled(selected == nil || selected==keys.count-1)
            }.buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private var uniqueGrid:some View {
        VStack(spacing:0) {
            HStack {Text("Column");Spacer();Text("Type").frame(width:65)}.padding(7).background(MLTheme.panel)
            ThemedScrollView {
                VStack(spacing:0) {ForEach(available.filter {search.isEmpty || $0.name.localizedCaseInsensitiveContains(search)},id:\.name) {column in
                    HStack {
                        Toggle(column.name,isOn:Binding(get:{keys.contains {$0.column==column.name}},set:{enabled in if enabled {keys.append(.init(column:column.name))} else {keys.removeAll {$0.column==column.name}}})).toggleStyle(FileInputCheckboxStyle(tint:accent))
                        Spacer();Text(column.dtype).foregroundStyle(MLTheme.muted).frame(width:65)
                    }.padding(7).overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                }}
            }.frame(height:min(240,max(40,CGFloat(available.count)*34)))
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func picker(_ title:String,value:Binding<String>,options:[(String,String)])->some View {FlatChoiceControl(title:title,selection:value.wrappedValue,options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {value.wrappedValue=$0}}
    private func choice(_ title:String,selection:Binding<String>,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);picker(title,value:selection,options:options)}}
    private func move(_ delta:Int) {guard let i=selected,keys.indices.contains(i+delta) else {return};keys.swapAt(i,i+delta);selected=i+delta}
    private func publish() {context.setNumericPending(origin:origin,key:draftKey,pending:changed)}
    private func reload() {
        context.setNumericPending(origin:origin,key:draftKey,pending:false);original=node.config;keys=[]
        if isSort {keys=(try? SortConfiguration(original))?.keys ?? []}
        else if let c=try? UniqueConfiguration(original) {keys=c.keys.map {.init(column:$0)};keep=c.keep;missing=c.missing;comparison=c.comparison}
        origin=context.numericOrigin(key:draftKey);error=nil;publish()
    }
    private func apply() {
        do {
            _ = try isSort ? SortTableConfiguration.validate(values,schema:schema):UniqueTableConfiguration.validate(values,schema:schema)
            if context.applyTableConfiguration(values,expected:original) {original=values;error=nil;publish()} else {error=context.statusMessage}
        } catch {self.error=error.localizedDescription}
    }
}
