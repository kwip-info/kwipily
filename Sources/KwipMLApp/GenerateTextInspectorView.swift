import SwiftUI
import KwipMLCore

struct GenerateTextInspectorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @Environment(\.colorScheme) private var colorScheme
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.evaluation.primary:ComponentVisualFamily.evaluation.secondary}
    @State private var draft:[String:JSONValue]=[:]
    @State private var numbers:[String:String]=[:]
    @State private var baseline:[String:JSONValue]=[:]
    @State private var message=""
    @State private var columns:[String]=[]
    @State private var modelMetadata:WorkflowGenerationModelMetadata?
    @State private var modelMessage=""
    private let numericKeys=["temperature","topK","topP","seed","maxNewTokens","maxPrompts","maxTotalTokens"]
    private var dirty:Bool {draft != baseline || !numbersMatch}
    private var numbersMatch:Bool {numericKeys.allSatisfy {numbers[$0] == numericText(baseline[$0])}}
    private func numericText(_ v:JSONValue?)->String {switch v {case .integer(let n):return String(n);case .number(let n):return String(n);default:return ""}}
    private func reset() {
        baseline=TextGenerationConfiguration.defaults.merging(context.configuration){_,new in new};draft=baseline
        numbers=Dictionary(uniqueKeysWithValues:numericKeys.map{($0,numericText(baseline[$0]))});message="";pending(false)
    }
    private func pending(_ value:Bool) {context.authority.setNumericPending(origin:context.authority.numericOrigin(key:"generationDraft"),key:"generationDraft",pending:value)}
    private func text(_ key:String)->Binding<String> {.init(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})}
    private func flag(_ key:String)->Binding<Bool> {.init(get:{draft[key] == .bool(true)},set:{draft[key] = .bool($0)})}
    private func choices(_ title:String,_ key:String,_ options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:5) {Text(title).font(.caption)
            FlatChoiceControl(title:title,selection:draft[key]?.stringValue ?? "",options:options.map{.init(id:$0.0,title:$0.1)}) {draft[key] = .string($0)}
        }
    }
    private func number(_ title:String,_ key:String)->some View {
        VStack(alignment:.leading,spacing:5) {Text(title).font(.caption);TextField(title,text:Binding(get:{numbers[key] ?? ""},set:{numbers[key]=$0})).flatField().accessibilityIdentifier("generation."+key)}
    }
    private func apply() {
        do {
            var c=draft
            for key in numericKeys {
                if ["temperature","topP"].contains(key) {guard let n=Double(numbers[key] ?? ""),n.isFinite else {throw DataFlowContractError("Enter a finite \(key).")};c[key] = .number(n)}
                else {guard let n=Int64(numbers[key] ?? "") else {throw DataFlowContractError("Enter a whole number for \(key).")};c[key] = .integer(n)}
            }
            let settings=try TextGenerationConfiguration(c)
            if !settings.tableMode,host.owner.document.edges.contains(where:{$0.to.address==context.authority.address && $0.to.port=="prompts"}) {throw DataFlowContractError("Disconnect the Prompts table before choosing One prompt.")}
            pending(false)
            if context.authority.configure(c,expected:context.configuration) {reset()} else {message=context.authority.message ?? "Configuration was not applied.";pending(dirty)}
        } catch {message=error.localizedDescription;pending(dirty)}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Label(modelMetadata == nil ? "Model input":"Captured model",systemImage:modelMetadata == nil ? "square.stack":"checkmark.shield").font(.caption.weight(.semibold)).foregroundStyle(accent)
            if let modelMetadata {Text("\(modelMetadata.vocabularySize) tokens · context \(modelMetadata.context)").font(.caption).foregroundStyle(MLTheme.muted)}
            else if !modelMessage.isEmpty {Text(modelMessage).font(.caption).foregroundStyle(MLTheme.muted)}
            choices("Prompt source","promptMode",[("one","One prompt"),("table","Table column")])
            if draft["promptMode"] == .string("table") {
                choices("Prompt column","promptColumn",columns.map{($0,$0)})
            } else {
                Text("Prompt").font(.caption)
                TextEditor(text:text("prompt")).font(.system(.body,design:.monospaced)).frame(minHeight:75,maxHeight:140).overlay(Rectangle().stroke(MLTheme.muted.opacity(0.3))).accessibilityIdentifier("generation.prompt")
            }
            choices("Long prompt","longPrompt",[("refuse","Refuse"),("latest","Keep latest context")])
            InspectorSection("Decoding",symbol:"text.append",identifier:"generation.section.decoding",initiallyExpanded:true,accent:accent) {
                choices("Method","method",[("greedy","Greedy"),("sample","Sample")])
                if draft["method"] == .string("sample") {number("Temperature","temperature");number("Top-k · 0 disables","topK");number("Top-p","topP");number("Seed","seed")}
                if let modelMetadata {
                    if modelMetadata.hasUnknown {Toggle("Allow unknown tokens",isOn:flag("allowUnknown")).tint(ComponentVisualFamily.evaluation.secondary)}
                    if modelMetadata.hasEOS {Toggle("Stop at EOS",isOn:flag("stopAtEOS")).tint(ComponentVisualFamily.evaluation.secondary)}
                }
                choices("Device","device",[("cpu","CPU"),("mps","Apple GPU")])
            }
            InspectorSection("Limits",symbol:"gauge.with.dots.needle.50percent",identifier:"generation.section.limits",initiallyExpanded:true,accent:accent) {
                number("New tokens / prompt","maxNewTokens")
                number("Maximum prompts","maxPrompts")
                number("Total new-token budget","maxTotalTokens")
            }
            InspectorSection("Output columns",symbol:"tablecells",identifier:"generation.section.output-columns",initiallyExpanded:true,accent:accent) {
                ForEach([("Text","textColumn"),("IDs","idsColumn"),("Count","countColumn"),("Stop","stopColumn")],id:\.1) {title,key in
                    HStack {Text(title).font(.caption).frame(width:40,alignment:.leading);TextField(title,text:text(key)).flatField()}
                }
            }
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!dirty);Button("Revert",action:reset).buttonStyle(FlatButtonStyle()).disabled(!dirty)}
            Button("Generate") {_=host.generateText(address:context.authority.address)}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(dirty || host.owner.isBusy).accessibilityIdentifier("generation.generate")
            if host.modelLoader.activeAddress==context.authority.address || host.modelLoader.generationRecord?.address==context.authority.address {Text(host.modelLoader.message(for:context.authority.address)).font(.caption).foregroundStyle(MLTheme.muted)}
            else if let generation=host.generation,generation.address==context.authority.address {Text(generation.statusMessage).font(.caption).foregroundStyle(MLTheme.muted)}
        }
        .onAppear(perform:reset)
        .onChange(of:draft) {_,_ in pending(dirty)}
        .onChange(of:numbers) {_,_ in pending(dirty)}
        .onDisappear {pending(false)}
        .task(id:"\(context.contextIdentity)-\(host.outcomes.count)-\(host.owner.isBusy)-\(host.modelLoader.bindings.revision)") {
            guard !host.owner.isBusy else {return}
            columns=await host.generationPromptColumns(for:context.authority.address)
            do {let metadata=try await host.generationModelMetadata(for:context.authority.address);guard !Task.isCancelled else {return};modelMetadata=metadata;modelMessage=""}
            catch {guard !Task.isCancelled else {return};modelMetadata=nil;modelMessage="Select a verified next-token model."}
        }
    }
}
