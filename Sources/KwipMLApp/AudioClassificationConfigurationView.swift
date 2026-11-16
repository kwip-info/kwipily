import SwiftUI
import KwipMLCore

struct AudioClassificationConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var error=""
    @State private var fieldGeneration=0
    private var accent:Color {ComponentVisualFamily.data.primary}
    private var pending:Bool {draft != original}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Source & model",symbol:"waveform",identifier:"classify-audio.source",initiallyExpanded:true,accent:accent) {
                let columns=context.upstreamSchema(forInput:"table").filter {$0.dtype=="audio" || $0.dtype=="tensor"}
                if columns.isEmpty {text("Waveform column",key:"inputColumn")} else {choice("Waveform column",key:"inputColumn",values:columns.map{($0.name,$0.name)})}
                Label("Connect Load model → Model",systemImage:"point.3.connected.trianglepath.dotted").foregroundStyle(accent)
                Text("AST · CPU · mono float32 · 16,000 Hz").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Windows",symbol:"rectangle.split.3x1",identifier:"classify-audio.windows",initiallyExpanded:true,accent:accent) {
                number("Window length · seconds · 0.025…10",key:"windowSeconds")
                number("Window hop · seconds · no longer than window",key:"hopSeconds")
                choice("Short final window",key:"finalWindow",values:[("pad","Pad computation · retain actual length"),("refuse","Refuse incomplete window")])
                Text("Times describe window coverage.").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Label reporting",symbol:"tag",identifier:"classify-audio.reporting",initiallyExpanded:true,accent:accent) {
                number("Minimum reported score · 0…1",key:"minimumScore")
                number("Maximum reported labels per window",key:"reportedLabels")
                text("Prediction column prefix",key:"outputPrefix")
                Text("Independent scores · all classes retained").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Compute & output limits",symbol:"gauge",identifier:"classify-audio.limits",initiallyExpanded:false,accent:accent) {
                number("Windows per CPU batch · 1…8",key:"batchWindows")
                number("Maximum complete-run windows · 1…4,096",key:"maximumWindows")
                number("Maximum complete-run score rows · 1…150,000",key:"maximumScoreRows")
                Text("Predictions + Scores → standard tables").foregroundStyle(MLTheme.muted)
            }
            HStack {Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending);Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)}
            if !error.isEmpty {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}.onChange(of:draft){_,_ in publish()}
            .onChange(of:node.config){_,next in if !pending {reload()} else if next != original {error="Saved settings changed. Revert to load them; the draft is preserved."}}
            .onDisappear{context.setNumericPending(origin:context.numericOrigin(key:"classify-audio"),key:"classify-audio",pending:false)}
    }
    private func text(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}}
    private func number(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);FileInputDraftEntry(title:title,value:Binding(get:{draft[key] ?? .integer(0)},set:{draft[key]=$0}),format:FileInputDraftText.display,parse:{raw in if let value=Int64(raw) {return .integer(value)};if let value=Double(raw),value.isFinite {return .number(value)};return .string(raw)}).id("\(key)-\(fieldGeneration)")}}
    private func choice(_ title:String,key:String,values:[(String,String)])->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? "",options:values.map{.init(id:$0.0,title:$0.1)}){draft[key] = .string($0)}}}
    private func publish(){context.setNumericPending(origin:context.numericOrigin(key:"classify-audio"),key:"classify-audio",pending:pending)}
    private func reload(){draft=node.config;original=node.config;fieldGeneration &+= 1;error="";publish()}
    private func apply(){do {guard let definition=context.operatorDefinition else {return};try AudioClassificationConfiguration.validate(draft,definition:definition);if context.applyTableConfiguration(draft,expected:original){original=draft;error="";publish()}else{error=context.statusMessage}}catch{self.error=error.localizedDescription}}
}
