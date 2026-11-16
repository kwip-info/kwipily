import SwiftUI
import KwipMLCore

struct UnionConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var original:[String:JSONValue]=[:]
    @State private var ports=["table1","table2"]
    @State private var match="name"
    @State private var coverage="all"
    @State private var typePolicy="strict"
    @State private var sourceColumn=""
    @State private var search=""
    @State private var origin=""
    @State private var identifier=false
    @State private var selected:String?
    @State private var dragOrigin:Int?
    @State private var error:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var config:[String:JSONValue] {["inputs":.array(ports.map(JSONValue.string)),"match":.string(match),"coverage":.string(coverage),"typePolicy":.string(typePolicy),"sourceColumn":.string(identifier ? sourceColumn:"")]}
    private var changed:Bool {config != original || (identifier && sourceColumn.isEmpty)}
    private var schemas:[String:[DataColumnSchema]] {Dictionary(uniqueKeysWithValues:ports.map {($0,context.upstreamSchema(forInput:$0))})}
    private var selectedIndex:Int? {ports.firstIndex {$0==selected}}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            inputGrid
            VStack(alignment:.leading,spacing:4) {
                Text("Match columns").fontWeight(.semibold).foregroundStyle(accent)
                HStack(spacing:0) {matchButton("name","By name");matchButton("position","By position")}.overlay(Rectangle().stroke(accent.opacity(0.35)))
            }
            alignmentGrid
            choice("Column coverage",selection:$coverage,options:match == "name" ? [("all","Include all · fill missing"),("shared","Shared columns only"),("exact","Require identical names")]:[("all","Include all · fill missing"),("shared","Shared positions only"),("exact","Require identical counts")])
            choice("Type policy",selection:$typePolicy,options:[("strict","Require same types"),("losslessNumeric","Checked numeric widening")])
            Toggle("Add input identifier",isOn:$identifier).toggleStyle(FileInputCheckboxStyle(tint:accent))
            if identifier {TextField("Identifier column name",text:$sourceColumn).flatField().accessibilityLabel("Input identifier column name")}
            HStack {Image(systemName:"checkmark.circle");Text("Duplicates retained");Spacer();Text("\(ports.count) inputs")}.font(.caption2).foregroundStyle(accent)
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!changed)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Spacer();if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Union changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
        .onChange(of:node.id) {_,_ in reload()}
        .onChange(of:node.config) {_,next in if next == original {return};if !changed {reload()} else {error="Saved settings changed. Revert to load them; your draft is preserved."}}
        .onChange(of:config) {_,_ in publish()}.onChange(of:identifier) {_,_ in publish()}
        .onDisappear {context.setNumericPending(origin:origin,key:"unionDraft",pending:false)}
    }
    private var inputGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Input order").fontWeight(.semibold).foregroundStyle(accent)
            VStack(spacing:0) {
                HStack(spacing:6) {Color.clear.frame(width:16);Text("Port").frame(width:55,alignment:.leading);Text("Source").frame(maxWidth:.infinity,alignment:.leading);Text("Cols").frame(width:32);Color.clear.frame(width:20)}.font(.system(size:10,weight:.semibold)).padding(7).background(MLTheme.panel)
                ThemedScrollView {
                    VStack(spacing:0) {ForEach(ports,id:\.self) {port in inputRow(port)}}
                }.frame(height:min(240,CGFloat(ports.count)*40))
                HStack {
                    Button("Add input") {if let next=(try? UnionConfiguration(config))?.nextPort() {ports.append(next);selected=next}}.disabled(ports.count>=32)
                    Spacer()
                    Button {move(-1)} label:{Image(systemName:"arrow.up")}.disabled(selectedIndex == nil || selectedIndex == 0).accessibilityLabel("Move selected input earlier")
                    Button {move(1)} label:{Image(systemName:"arrow.down")}.disabled(selectedIndex == nil || selectedIndex == ports.count-1).accessibilityLabel("Move selected input later")
                }.buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private func inputRow(_ port:String)->some View {
        HStack(spacing:6) {
            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:16,height:30).contentShape(Rectangle()).gesture(DragGesture(minimumDistance:4,coordinateSpace:.global).onChanged {value in
                guard context.isCurrent,!context.isBusy,let current=ports.firstIndex(of:port) else {return}
                if dragOrigin == nil {dragOrigin=current};selected=port
                let target=min(ports.count-1,max(0,(dragOrigin ?? current)+Int((value.translation.height/40).rounded())))
                ports.remove(at:current);ports.insert(port,at:target)
            }.onEnded {_ in dragOrigin=nil}).accessibilityLabel("Drag \(UnionConfiguration.title(port)) to reorder")
            Text(UnionConfiguration.title(port)).frame(width:55,alignment:.leading).foregroundStyle(accent)
            Text(context.inputSourceName(forInput:port) ?? "Not connected").lineLimit(1).truncationMode(.middle).frame(maxWidth:.infinity,alignment:.leading).help(context.inputSourceName(forInput:port) ?? "Not connected")
            Text(schemas[port]?.isEmpty == false ? "\(schemas[port]!.count)":"—").foregroundStyle(MLTheme.muted).frame(width:32)
            Button {
                guard context.inputSourceName(forInput:port)==nil else {error="Disconnect \(UnionConfiguration.title(port)) before removing its port.";return}
                ports.removeAll {$0==port};selected=nil;error=nil
            } label:{Image(systemName:"minus")}.buttonStyle(.plain).frame(width:20).disabled(ports.count<=2).accessibilityLabel("Remove \(UnionConfiguration.title(port))")
        }.padding(.horizontal,7).frame(height:40).background(selected==port ? accent.opacity(0.1):MLTheme.background).contentShape(Rectangle()).onTapGesture {selected=port}.accessibilityElement(children:.contain).accessibilityLabel("Union input \(UnionConfiguration.title(port))").overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    private var alignmentGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            HStack {Text("Column alignment").fontWeight(.semibold).foregroundStyle(accent);Spacer();Image(systemName:"lock").foregroundStyle(MLTheme.muted)}
            TextField("Find output column",text:$search).flatField().accessibilityLabel("Find Union output column")
            VStack(spacing:0) {
                switch alignmentResult {
                case .failure(let failure):Text(failure.localizedDescription).foregroundStyle(MLTheme.rose).frame(maxWidth:.infinity,alignment:.leading).padding(9)
                case .success(let all):
                    let rows=all.filter {search.isEmpty || $0.output.name.localizedCaseInsensitiveContains(search) || $0.sources.compactMap {$0}.contains {$0.localizedCaseInsensitiveContains(search)}}
                    if all.isEmpty {Text("Alignment unresolved").foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,alignment:.leading).padding(9)}
                    else {
                        ThemedScrollView(.horizontal) {
                            VStack(spacing:0) {
                                HStack(spacing:0) {cell("Output",width:95);ForEach(ports,id:\.self) {cell(UnionConfiguration.title($0),width:match == "position" ? 100:53)};cell("Type",width:65)}.fontWeight(.semibold).background(MLTheme.panel)
                                ThemedScrollView {
                                    LazyVStack(spacing:0) {
                                        ForEach(rows.indices,id:\.self) {i in
                                            HStack(spacing:0) {cell(rows[i].output.name,width:95);ForEach(ports.indices,id:\.self) {p in cell(match == "name" ? (rows[i].sources[p] == nil ? "—":"✓"):(rows[i].sources[p] ?? "—"),width:match == "position" ? 100:53)};cell(rows[i].output.dtype,width:65)}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                                        }
                                        if rows.isEmpty {Text("No matching columns").padding(10).foregroundStyle(MLTheme.muted)}
                                    }
                                }.frame(height:min(220,max(40,CGFloat(rows.count)*32)))
                            }.frame(width:160+CGFloat(ports.count)*(match == "position" ? 100:53))
                        }
                    }
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
        }
    }
    private var alignmentResult:Result<[UnionTableConfiguration.Alignment],Error> {Result {try UnionTableConfiguration.alignment(config,schemas:schemas)}}
    private func cell(_ text:String,width:CGFloat)->some View {Text(text).font(.system(size:10)).lineLimit(1).truncationMode(.middle).frame(width:width-12,height:32,alignment:.leading).padding(.horizontal,6).help(text)}
    private func matchButton(_ value:String,_ title:String)->some View {Button {match=value} label:{Text(title).frame(maxWidth:.infinity).padding(8).foregroundStyle(match == value ? accent:MLTheme.muted).background(match == value ? accent.opacity(0.12):MLTheme.panel)}.buttonStyle(.plain).accessibilityLabel("Match Union columns \(title)")}
    private func choice(_ title:String,selection:Binding<String>,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:selection.wrappedValue,options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {selection.wrappedValue=$0}}}
    private func move(_ delta:Int) {guard let i=selectedIndex,ports.indices.contains(i+delta) else {return};ports.swapAt(i,i+delta)}
    private func publish() {context.setNumericPending(origin:origin,key:"unionDraft",pending:changed)}
    private func reload() {
        context.setNumericPending(origin:origin,key:"unionDraft",pending:false);original=node.config
        do {let c=try UnionConfiguration(original);ports=c.inputs;match=c.match;coverage=c.coverage;typePolicy=c.typePolicy;identifier = !c.sourceColumn.isEmpty;sourceColumn=identifier ? c.sourceColumn:"input";error=nil} catch {self.error=error.localizedDescription}
        selected=ports.first;origin=context.numericOrigin(key:"unionDraft");publish()
    }
    private func apply() {
        guard !identifier || !sourceColumn.isEmpty else {error="Name the input identifier column.";return}
        if context.applyTableConfiguration(config,expected:original) {original=config;error=nil;publish()} else {error=context.statusMessage}
    }
}
