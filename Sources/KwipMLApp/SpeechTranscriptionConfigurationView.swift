import SwiftUI
import KwipMLCore

struct SpeechTranscriptionConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var error=""
    private var accent:Color {ComponentVisualFamily.data.primary}
    private var pending:Bool {draft != original}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Source & model",symbol:"waveform",identifier:"speech.source",initiallyExpanded:true,accent:accent) {
                let columns=context.upstreamSchema(forInput:"table").filter {$0.dtype=="audio" || $0.dtype=="tensor"}
                if columns.isEmpty {text("Waveform column",key:"inputColumn")} else {choice("Waveform column",key:"inputColumn",values:columns.map{($0.name,$0.name)})}
                Label("Connect Load model → Model",systemImage:"point.3.connected.trianglepath.dotted").foregroundStyle(accent)
                Text("Whisper · CPU · mono float32 · 16,000 Hz").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Recognition",symbol:"text.bubble",identifier:"speech.recognition",initiallyExpanded:true,accent:accent) {
                choice("Spoken language",key:"language",values:[("auto","Detect per chunk"),("en","English"),("es","Spanish"),("fr","French"),("de","German"),("it","Italian"),("pt","Portuguese"),("ja","Japanese"),("ko","Korean"),("zh","Chinese")])
                number("Chunk length · seconds · 5…30",key:"chunkSeconds")
                number("Maximum new tokens · per chunk",key:"maximumTokens")
                number("Beam width · 1…5",key:"beamWidth")
                Text("Sequential chunks · estimated segment times").foregroundStyle(MLTheme.muted)
            }
            InspectorSection("Output & limits",symbol:"tablecells",identifier:"speech.output",initiallyExpanded:true,accent:accent) {
                text("Transcript column",key:"outputColumn")
                number("Maximum complete-run text characters",key:"maximumCharacters")
                number("Maximum complete-run segments",key:"maximumSegments")
                Text("Transcripts + Segments → standard tables").foregroundStyle(MLTheme.muted)
            }
            HStack {Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending);Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)}
            if !error.isEmpty {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}.onChange(of:draft){_,_ in publish()}
            .onChange(of:node.config){_,next in if !pending {reload()} else if next != original {error="Saved settings changed. Revert to load them; the draft is preserved."}}
            .onDisappear{context.setNumericPending(origin:context.numericOrigin(key:"speech"),key:"speech",pending:false)}
    }
    private func text(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}}
    private func number(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);FileInputDraftEntry(title:title,value:Binding(get:{draft[key] ?? .integer(0)},set:{draft[key]=$0}),format:FileInputDraftText.display,parse:{Int64($0).map(JSONValue.integer) ?? .string($0)})}}
    private func choice(_ title:String,key:String,values:[(String,String)])->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? "",options:values.map{.init(id:$0.0,title:$0.1)}){draft[key] = .string($0)}}}
    private func publish(){context.setNumericPending(origin:context.numericOrigin(key:"speech"),key:"speech",pending:pending)}
    private func reload(){draft=node.config;original=node.config;error="";publish()}
    private func apply(){do {guard let definition=context.operatorDefinition else {return};try SpeechTranscriptionConfiguration.validate(draft,definition:definition);if context.applyTableConfiguration(draft,expected:original){original=draft;error="";publish()}else{error=context.statusMessage}}catch{self.error=error.localizedDescription}}
}
