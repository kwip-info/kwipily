import SwiftUI
import KwipMLCore

struct SequenceWindowsConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var origin=""
    @State private var failure:String?
    @State private var pendingFields=Set<String>()
    @State private var fieldGeneration=UUID()
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var pending:Bool {draft != original || !pendingFields.isEmpty}
    private var schema:[DataColumnSchema] {context.upstreamSchema(forInput:draft["mode"] == .string("partitions") ? "training":"table")}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            InspectorSection("Source",symbol:"tablecells",identifier:"sequence-windows.source",initiallyExpanded:true,accent:accent) {
                choice("Inputs",key:"mode",options:[("table","Table"),("partitions","Training / Validation / Test")])
                column("Token column",key:"tokenColumn",types:["int64Sequence"])
                choice("Sequence scope",key:"scope",options:[("row","Each row"),("document","Ordered rows in each document")])
                if draft["scope"] == .string("document") {
                    column("Document column",key:"documentColumn",types:["string","int64"])
                    column("Order column",key:"orderColumn",types:["string","int64"])
                    choice("Direction",key:"direction",options:[("ascending","Ascending"),("descending","Descending")])
                }
            }
            InspectorSection("Windows",symbol:"rectangle.split.3x1",identifier:"sequence-windows.windows",initiallyExpanded:true,accent:accent) {
                number("Context length",key:"contextLength",range:1...4096)
                number("Stride",key:"stride",range:1...65536)
                HStack {Text("Target").foregroundStyle(MLTheme.muted);Spacer();Text("Next token (+1)")}
                choice("Incomplete window",key:"incomplete",options:[("drop","Drop"),("pad","Pad")])
                if draft["incomplete"] == .string("pad") {
                    number("Padding ID",key:"paddingID",range:0...65535)
                    Label("Loss mask · valid targets only",systemImage:"checkmark.shield").foregroundStyle(accent)
                }
            }
            InspectorSection("Output",symbol:"tablecells",identifier:"sequence-windows.output",initiallyExpanded:true,accent:accent) {
                VStack(spacing:0) {
                    HStack {Text("Role").frame(width:80,alignment:.leading);Text("Column");Spacer()}.fontWeight(.semibold).padding(8).background(accent.opacity(0.1))
                    output("Inputs",key:"inputColumn");output("Targets",key:"targetColumn");output("Valid length",key:"lengthColumn")
                    if draft["incomplete"] == .string("pad") {output("Loss mask",key:"maskColumn")}
                }.overlay(Rectangle().stroke(MLTheme.rule,lineWidth:1))
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending || !pendingFields.isEmpty)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer();if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Sequence windows changes")}
            }
            if let failure {Text(failure).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).accessibilityIdentifier("sequence-windows.configuration")
            .onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}
            .onChange(of:draft){_,_ in publish()}
            .onChange(of:node.config){_,next in if !pending {reload()} else if next != original {failure="Saved settings changed. Revert to load them; your draft is preserved."}}
            .onDisappear {context.setNumericPending(origin:origin,key:"sequenceWindows",pending:false)}
    }
    private func column(_ title:String,key:String,types:[String])->some View {
        VStack(alignment:.leading,spacing:4) {
            choice(title,key:key,options:schema.filter {types.contains($0.dtype)}.map {($0.name,$0.name)})
            if schema.isEmpty {text("Exact "+title.lowercased(),key:key)}
        }
    }
    private func output(_ title:String,key:String)->some View {
        HStack {Text(title).frame(width:80,alignment:.leading);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField().accessibilityLabel("Output "+title)}.padding(6)
    }
    private func choice(_ title:String,key:String,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {
        Text(title).foregroundStyle(MLTheme.muted)
        FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? "",options:options.map {.init(id:$0.0,title:$0.1)},accent:accent) {value in
            draft[key] = .string(value)
        }.accessibilityIdentifier("sequence-windows."+key)
        }
    }
    private func text(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}.accessibilityIdentifier("sequence-windows."+key)
    }
    private func number(_ title:String,key:String,range:ClosedRange<Int>)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:.wholeDisplay(Double(draft[key]?.integerValue ?? range.lowerBound)),domain:.integer(minimum:Double(range.lowerBound),maximum:Double(range.upperBound)),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{origin,pending in
                if pending {pendingFields.insert(key)} else {pendingFields.remove(key)}
                context.setNumericPending(origin:origin,key:key,pending:pending)
            }) {value in guard case .integer(let n)=value else {return false};draft[key] = .integer(n);return true}.id(fieldGeneration)
        }.accessibilityIdentifier("sequence-windows."+key)
    }
    private func check(_ title:String,key:String)->some View {Toggle(title,isOn:Binding(get:{draft[key] == .bool(true)},set:{draft[key] = .bool($0)})).toggleStyle(.checkbox).accessibilityIdentifier("sequence-windows."+key)}
    private func publish(){context.setNumericPending(origin:origin,key:"sequenceWindows",pending:pending)}
    private func reload(){fieldGeneration=UUID();pendingFields=[];context.setNumericPending(origin:origin,key:"sequenceWindows",pending:false);draft=node.config;original=node.config;origin=context.numericOrigin(key:"sequenceWindows");failure=nil;publish()}
    private func apply(){do {try SequenceWindowsConfiguration.validate(draft,schema:schema.map {.init(name:$0.name,dtype:$0.dtype,nullable:$0.nullable ?? true)});if context.applyTableConfiguration(draft,expected:original) {original=draft;failure=nil;publish()} else {failure=context.statusMessage}} catch {failure=error.localizedDescription}}
}

struct SequenceWindowsInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            LessonTextPanel(title:"Aligned next-token examples",text:"Context length is the number of input/target positions. Each target is the following token. A full window needs one extra source token; stride controls the next start position.")
            LessonTextPanel(title:"Document and partition boundaries",text:"Each row stays independent by default. Grouped mode concatenates tokens in the explicit document/order sequence, without inserting separators. Order values must be unique within a document. Training, Validation and Test never mix. Split documents or contiguous corpus regions before tokenization/windowing; do not randomly partition overlapping windows afterward.")
            LessonTextPanel(title:"Padding and memory",text:"Pad emits a loss mask and valid-target count; zero-target windows are omitted. Padding ID is explicit and must match your vocabulary/model before training. Context is bounded to 4,096, stride to 65,536. Fragments are ordered in owned disk storage and windows use a bounded buffer. Existing scratch/artifact budgets still apply; no source data is silently truncated.")
            LessonTextPanel(title:"Inspect and learn",text:"Inspect opens aligned captured input/target tokens and IDs in Results; Back to Output returns to the table. Labels use the captured vocabulary when available. Sequence objective and mask-aware Train support follow in the next checkpoint; this tool does not train a language model or generate text.")
        }
    }
}
