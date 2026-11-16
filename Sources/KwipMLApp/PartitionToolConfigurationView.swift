import SwiftUI
import KwipMLCore

struct PartitionToolConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=try! PartitionToolDraft(configuration:PartitionToolDraft.defaults)
    @State private var saved=try! PartitionToolDraft(configuration:PartitionToolDraft.defaults)
    @State private var original:[String:JSONValue]=[:]
    @State private var origin=""
    @State private var error:String?
    @State private var keyText=""
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var pending:Bool {draft != saved}
    private var schema:[DataColumnSchema] {context.upstreamSchema}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Label("Input schema",systemImage:"tablecells");Spacer();Text(schema.isEmpty ? "Unresolved":"\(schema.count) columns")}.foregroundStyle(accent).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            HStack(spacing:0) {
                ForEach([("random","Random"),("entity","By entity"),("time","By time")],id:\.0) {mode,title in
                    Button {draft.switchStrategy(mode)} label:{Text(title).frame(maxWidth:.infinity).padding(.vertical,8).background(draft.strategy == mode ? accent.opacity(0.14):Color.clear)}.buttonStyle(.plain).foregroundStyle(draft.strategy == mode ? accent:Color.primary).accessibilityLabel("Partition strategy \(title)").accessibilityValue(draft.strategy == mode ? "Selected":"")
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
            if draft.strategy == "time" {timeControls} else {allocationControls}
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer();if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Partition changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent)
            .onAppear(perform:reload).onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node.config) {_,next in if next == original {return};if !pending {reload()} else {error="Saved configuration changed. Revert to load it; your draft is preserved."}}
            .onChange(of:draft) {_,_ in publish()}
            .onDisappear {context.setNumericPending(origin:origin,key:"partitionTool",pending:false)}
    }
    private var allocationControls:some View {
        VStack(alignment:.leading,spacing:9) {
            Text(draft.strategy == "entity" ? "Entity allocation":"Row allocation").fontWeight(.semibold).foregroundStyle(accent)
            HStack {Text("Training");Spacer();Text(training).monospacedDigit()}.padding(9).background(accent.opacity(0.08))
            field("Validation (%)",key:"validationPercent");field("Test (%)",key:"testPercent");field("Seed",key:"seed")
            if draft.strategy == "entity" {keys("Entity keys",key:"groupBy")}
            else {
                Toggle("Stratify by column",isOn:Binding(get:{draft.value("stratifyColumn") != nil},set:{if $0 {draft.set("stratifyColumn",.string(schema.first(where:{$0.dtype != "image"})?.name ?? ""))} else {draft.remove("stratifyColumn")}})).toggleStyle(.checkbox)
                if draft.value("stratifyColumn") != nil {column("Stratification column",key:"stratifyColumn")}
            }
        }
    }
    private var training:String {guard let validation=Int(draft.text("validationPercent")),let test=Int(draft.text("testPercent")),validation > 0,test > 0,validation+test < 100 else {return "—"};return "\(100-validation-test)%"}
    private var timeControls:some View {
        VStack(alignment:.leading,spacing:9) {
            column("Time column",key:"timeColumn")
            choice("Time encoding",key:"timeEncoding",options:[("iso8601Offset","ISO 8601 · explicit offset"),("epochMilliseconds","Epoch milliseconds")])
            field("Training ends (inclusive)",key:"trainingEnd");field("Validation ends (inclusive)",key:"validationEnd")
            Text("Embargo after each cutoff")
            TextField("Duration",text:$draft.embargoText).flatField().accessibilityLabel("Embargo duration")
            FlatChoiceControl(title:"Embargo unit",selection:draft.embargoUnit,options:["milliseconds","seconds","minutes","hours","days"].map{.init(id:$0,title:$0.capitalized)},accent:accent) {draft.setEmbargoUnit($0)}
            choice("Entities across time",key:"entityPolicy",options:[("allowAcrossTime","Allow across time"),("requireDisjoint","Require disjoint entities")])
            keys("Entity keys",key:"entityColumns")
        }
    }
    private func binding(_ key:String)->Binding<String> {Binding(get:{draft.text(key)},set:{draft.set(key,.string($0))})}
    private func field(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:binding(key)).flatField().accessibilityLabel(title)}}
    private func choice(_ title:String,key:String,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:draft.text(key),options:options.map{.init(id:$0.0,title:$0.1)},accent:accent) {draft.set(key,.string($0))}}}
    @ViewBuilder private func column(_ title:String,key:String) -> some View {
        if schema.isEmpty {field(title,key:key)} else {choice(title,key:key,options:schema.filter{$0.dtype != "image"}.map{($0.name,$0.name)})}
    }
    private func keys(_ title:String,key:String)->some View {
        let names=draft.names(key)
        return VStack(alignment:.leading,spacing:5) {
            Text(title).fontWeight(.semibold).foregroundStyle(accent)
            ForEach(Array(names.enumerated()),id:\.offset) {index,name in
                HStack(spacing:8) {
                    Text("\(index+1)").foregroundStyle(MLTheme.muted);Text(name).lineLimit(1).truncationMode(.middle);Spacer()
                    Button {move(key,index,-1)} label:{Image(systemName:"arrow.up")}.disabled(index==0).accessibilityLabel("Move \(name) up")
                    Button {move(key,index,1)} label:{Image(systemName:"arrow.down")}.disabled(index==names.count-1).accessibilityLabel("Move \(name) down")
                    Button {draft.set(key,.array(names.enumerated().filter{$0.offset != index}.map{.string($0.element)}))} label:{Image(systemName:"minus")}.accessibilityLabel("Remove \(name)")
                }.buttonStyle(.plain).padding(8).background(accent.opacity(0.06)).overlay(Rectangle().stroke(MLTheme.rule))
            }
            if schema.isEmpty {
                TextField("Exact column name",text:$keyText).flatField()
                Button("Add key") {if !keyText.isEmpty,!names.contains(keyText) {draft.set(key,.array((names+[keyText]).map(JSONValue.string)));keyText=""}}.buttonStyle(FlatButtonStyle())
            } else {
                FlatChoiceControl(title:"Add entity key",selection:"",options:schema.filter{$0.dtype != "image" && !names.contains($0.name)}.map{.init(id:$0.name,title:$0.name)},placeholder:"Add entity key",accent:accent) {draft.set(key,.array((names+[$0]).map(JSONValue.string)))}
            }
        }
    }
    private func move(_ key:String,_ index:Int,_ delta:Int) {var names=draft.names(key);names.swapAt(index,index+delta);draft.set(key,.array(names.map(JSONValue.string)))}
    private func publish() {context.setNumericPending(origin:origin,key:"partitionTool",pending:pending)}
    private func reload() {
        context.setNumericPending(origin:origin,key:"partitionTool",pending:false)
        do {draft=try .init(configuration:node.config);saved=draft;original=node.config;origin=context.numericOrigin(key:"partitionTool");error=nil;publish()} catch {self.error=error.localizedDescription}
    }
    private func apply() {
        do {let config=try draft.configuration(schema:schema);if context.applyTableConfiguration(config,expected:original) {original=config;saved=draft;error=nil;publish()} else {error=context.statusMessage}} catch {self.error=error.localizedDescription}
    }
}
