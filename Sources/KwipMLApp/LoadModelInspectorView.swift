import SwiftUI
import KwipMLCore

struct LoadModelInspectorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var store:ModelLoaderStore
    let history:WorkflowHistoryStore
    @State private var draft=LoadModelConfiguration()
    @State private var baseline:[String:JSONValue]=[:]
    @State private var maximum="512"
    @State private var previewText="A quiet morning."
    @State private var error:String?
    @State private var privateAccess=false
    @State private var credential=""
    @State private var credentialNames:[String]=[]
    @State private var credentialSheet=false
    @State private var token=""
    @Environment(\.colorScheme) private var colorScheme
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.boundaries.primary:ComponentVisualFamily.boundaries.secondary}
    private var address:WorkflowNodeAddress {context.authority.address}
    private var dirty:Bool {draft.values != baseline || maximum != String(draft.maximumTokens)}
    private func pending(_ value:Bool) {context.authority.setNumericPending(origin:context.authority.numericOrigin(key:"loadModelDraft"),key:"loadModelDraft",pending:value)}
    private var credentialSelection:ModelHubCredentialSelection {.init(root:store.root,project:store.owner.projectID)}
    private func reset() {
        error=nil
        do {credentialNames=try ModelHubCredentials.names();credential=try credentialSelection.name(for:address) ?? "";privateAccess = !credential.isEmpty} catch {self.error=error.localizedDescription}
        baseline=context.configuration;draft=(try? .decode(baseline)) ?? .init();maximum=String(draft.maximumTokens);pending(false)}
    @discardableResult private func apply()->Bool {
        do {
            guard let n=Int(maximum) else {throw DataFlowContractError("Maximum tokens must be a whole number.")};draft.maximumTokens=n
            try draft.validate();pending(false)
            guard draft.values==baseline || context.authority.configure(draft.values,expected:baseline) else {throw DataFlowContractError(context.statusMessage)}
            baseline=context.configuration;error=nil;return true
        } catch {self.error=error.localizedDescription;pending(dirty);return false}
    }
    private func saveCredentialSelection()->Bool {
        do {if privateAccess {guard credentialNames.contains(credential) else {throw WorkflowArchiveError("Choose an available saved Hub credential.")}}
            try credentialSelection.select(privateAccess ? credential:nil,for:address);return true
        } catch {self.error=error.localizedDescription;return false}
    }
    private func choice(_ title:String,selection:String,options:[(String,String)],onSelect:@escaping(String)->Void)->some View {
        VStack(alignment:.leading,spacing:5) {Text(title).font(.caption);FlatChoiceControl(title:title,selection:selection,options:options.map{.init(id:$0.0,title:$0.1)},accent:accent,onSelect:onSelect)}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            choice("Source",selection:draft.source.rawValue,options:[("savedRun","Saved run"),("localFolder","Local folder"),("huggingFace","Hugging Face"),("documentPackage","Prebuilt / document model package")]) {if let source=LoadModelConfiguration.Source(rawValue:$0) {draft.source=source}}
            if draft.source == .savedRun {
                if baseline.isEmpty {
                    if host.selfPlayBindings[address]==nil {WorkflowFrozenModelSelectionView(context:context,host:host,history:history).onAppear{host.reloadSelfPlayRecords()}}else{Button("Clear paired model selection"){_=host.clearSelfPlayModel(address)}.buttonStyle(FlatButtonStyle())}
                    if !host.selfPlayRecords.filter({$0.completed}).isEmpty {
                        Text("Game policy/value checkpoints").font(.caption).foregroundStyle(MLTheme.muted)
                        FlatChoiceControl(title:"Paired model",selection:host.selfPlayBindings[address] ?? "",options:[.init(id:"",title:"Choose paired checkpoint")]+host.selfPlayRecords.filter{$0.completed}.map{.init(id:$0.id,title:String($0.id.prefix(8))+" · "+$0.finishedAt.formatted(date:.abbreviated,time:.shortened))}){id in if let r=host.selfPlayRecords.first(where:{$0.id==id}){_=host.selectSelfPlayModel(r,for:address)}}
                        Text("Verified same-project paired weights. Game session consumes the policy head; value remains available for the task. Inference selection does not resume learning.").font(.caption).foregroundStyle(MLTheme.muted)
                    }
                }
                else {Text("Apply to choose a saved run.").font(.caption).foregroundStyle(MLTheme.muted)}
            } else {
                if draft.source == .localFolder || draft.source == .documentPackage {
                    Button("Choose folder…") {guard apply() else {return};host.chooseLocalModel(address)}.buttonStyle(FlatButtonStyle())
                } else {
                    Button("Use reviewed Laya typed-decisions snapshot") {
                        draft.repository="convaiinnovations/laya";draft.revision="7b928d828b7b0e022f929d9bd2e44165aa270148";draft.subfolder="typed-decisions"
                    }.buttonStyle(FlatButtonStyle())
                    Text("Optional typed-decision encoder/head · about843 MB safe weights plus tokenizer. Choice, ordinal and yes/no tasks; no conversational generation. Nothing downloads until requested.").font(.caption).foregroundStyle(MLTheme.muted)
                    Button("Use reviewed AudioSet AST snapshot") {
                        draft.repository="MIT/ast-finetuned-audioset-10-10-0.4593";draft.revision="f826b80d28226b62986cc218e5cec390b1096902";draft.subfolder=""
                    }.buttonStyle(FlatButtonStyle())
                    Text("Optional 527-class audio classifier · safe weights and matching processor · review license and download size before requesting.").font(.caption).foregroundStyle(MLTheme.muted)
                    Button("Use reviewed Whisper Tiny snapshot") {
                        draft.repository="openai/whisper-tiny";draft.revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af";draft.subfolder=""
                    }.buttonStyle(FlatButtonStyle())
                    Text("Optional speech recognition · model download only when requested. Review the pinned repository license and files before downloading.").font(.caption).foregroundStyle(MLTheme.muted)
                    field("Repository",text:$draft.repository)
                    field("Revision",text:$draft.revision)
                    field("Subfolder (optional)",text:$draft.subfolder)
                    Toggle("Use private Hub access",isOn:$privateAccess).toggleStyle(.checkbox).tint(accent).font(.caption)
                    if privateAccess {
                        FlatChoiceControl(title:"Saved Hub credential",selection:credential,options:([("","Choose credential")]+credentialNames.map{($0,$0)}+(!credential.isEmpty && !credentialNames.contains(credential) ? [(credential,credential+" · unavailable")]:[])).map{.init(id:$0.0,title:$0.1)},accent:accent){credential=$0}
                        if !credential.isEmpty && !credentialNames.contains(credential) {Text("This credential is unavailable. Choose another or add it again.").font(.caption).foregroundStyle(MLTheme.rose)}
                        Button("Manage credential…") {credentialSheet=true}.buttonStyle(FlatButtonStyle())
                    }
                    Button("Resolve revision") {guard apply() else {return};guard saveCredentialSelection() else {return};host.resolveModelHub(address,credential:privateAccess ? credential:nil)}.buttonStyle(FlatButtonStyle())
                    if let snapshot=store.resolved[address] {
                        VStack(alignment:.leading,spacing:6) {
                            Text("Pinned snapshot").font(.caption.weight(.semibold))
                            Text(String(snapshot.revision.prefix(16))).font(.caption.monospaced()).textSelection(.enabled)
                            Text("\(snapshot.files.count) files · \(ByteCountFormatter.string(fromByteCount:snapshot.files.reduce(Int64(0)){$0+Int64($1.bytes)},countStyle:.file))").font(.caption).foregroundStyle(MLTheme.muted)
                            DisclosureGroup("Files") {ForEach(snapshot.files,id:\.path) {file in HStack {Text(file.path).lineLimit(1);Spacer();Text(ByteCountFormatter.string(fromByteCount:Int64(file.bytes),countStyle:.file))}.font(.caption2)}}
                        }.padding(9).overlay(Rectangle().stroke(MLTheme.rule))
                        Button("Download & verify") {guard apply() else {return};guard saveCredentialSelection() else {return};host.downloadModelHub(address,credential:privateAccess ? credential:nil)}.buttonStyle(FlatButtonStyle(prominent:true))
                    }
                }
                if let reference=store.reference(address) {
                    metadata(reference)
                    if reference.metadata["task"] == .string("typedDecisions") {
                        Text("Laya · typed decisions · CPU reference\nMatching encoder, head and tokenizer retained. Connect Train for adaptation or Decide for complete Decisions/Scores tables. Encoder limit \(reference.metadata["maximumTokens"]?.integerValue ?? 0) · decision-head limit \(reference.metadata["decisionHeadTokens"]?.integerValue ?? 0) tokens. State capacity depends on all questions/options; no silent truncation.").font(.caption).foregroundStyle(MLTheme.muted)
                        if case .array(let warnings)=reference.metadata["calibrationWarnings"],!warnings.isEmpty {Text("Imported calibration diagnostic: "+warnings.compactMap(\.stringValue).joined(separator:"; ")).font(.caption).foregroundStyle(MLTheme.rose)}
                        Text("Imported temperatures do not establish calibration for your data. Act/escalate probability is not used. RLCD, Apple GPU and exact optimizer resume are not qualified.").font(.caption).foregroundStyle(MLTheme.muted)
                    } else if reference.metadata["task"] == .string("audioClassification") {
                        Text("AST · multi-label audio classification · CPU\n16 kHz mono waveform → Classify audio\nMatching processor retained. No tokenizer or classification fine-tuning. Independent sigmoid scores do not sum to one.").font(.caption).foregroundStyle(MLTheme.muted)
                    } else if reference.metadata["task"] == .string("automaticSpeechRecognition") {
                        Text("Whisper · speech-to-text · CPU\n16 kHz mono waveform → Transcribe audio\nMatching audio processor and tokenizer retained. ASR training, translation and speech synthesis are not supported.").font(.caption).foregroundStyle(MLTheme.muted)
                    } else if reference.metadata["task"] == .string("prebuilt") {
                        Text("Frozen numeric model · prepared inputs in saved feature order. Connect Prebuilt models → Predict/Assign new rows or Evaluate; use the loaded model settings. Upstream transforms are not implicitly included.").font(.caption).foregroundStyle(MLTheme.muted)
                    } else if draft.source == .documentPackage {
                        Text("Frozen vocabulary, IDF and class order · no tokenizer download").font(.caption).foregroundStyle(MLTheme.muted)
                        Text("Connect this model to Prebuilt models → Predict unseen documents, plus a text table. Prediction never refits preparation.").font(.caption)
                    } else {
                    InspectorSection("Tokenizer",symbol:"textformat",identifier:"load-model.tokenizer",initiallyExpanded:true,accent:accent) {
                        Text("From model · vocabulary matched").font(.caption).foregroundStyle(MLTheme.muted)
                        field("Maximum tokens",text:$maximum)
                        choice("Truncation",selection:draft.truncation,options:[("reject","Reject over-length"),("truncate","Truncate at maximum")]) {draft.truncation=$0}
                        choice("Padding",selection:draft.padding,options:[("none","None"),("longest","Longest in batch"),("maximum","Maximum tokens")]) {draft.padding=$0}
                        choice("Padding side",selection:draft.paddingSide,options:[("right","Right"),("left","Left")]) {draft.paddingSide=$0}
                        Text("Special tokens · Model defaults").font(.caption).foregroundStyle(MLTheme.muted)
                        field("Sample text",text:$previewText)
                        Button("Preview tokens") {guard apply() else {return};host.previewModelTokens(address,text:previewText)}.buttonStyle(FlatButtonStyle())
                        if let packet=store.tokenPreview[address],packet["modelIdentity"] == .string(reference.identity),packet["sampleText"] == .string(previewText),packet["configuration"] == .object(draft.values),case .array(let rows)=packet["tokens"],case .array(let tokens)=rows.first,case .array(let idsRows)=packet["input_ids"],case .array(let ids)=idsRows.first,case .array(let masksRows)=packet["attention_mask"],case .array(let masks)=masksRows.first {
                            Grid(alignment:.leading,horizontalSpacing:12,verticalSpacing:5) {
                                GridRow {Text("Token");Text("ID");Text("Mask")}.fontWeight(.semibold)
                                ForEach(Array(tokens.prefix(32).indices),id:\.self) {index in GridRow {Text(tokens[index].stringValue ?? "").lineLimit(1);Text(ids.indices.contains(index) ? String(ids[index].integerValue ?? 0):"—");Text(masks.indices.contains(index) ? String(masks[index].integerValue ?? 0):"—")}}
                            }.font(.caption.monospaced()).padding(8).frame(maxWidth:.infinity,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))
                            if tokens.count>32 {Text("First 32 of \(tokens.count) tokens").font(.caption2).foregroundStyle(MLTheme.muted)}
                        }
                    }
                    }
                }
            }
            if let error {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
            if !context.statusMessage.isEmpty {Text(context.statusMessage).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {Button("Apply") {_=apply()}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(!dirty);Button("Revert",action:reset).buttonStyle(FlatButtonStyle()).disabled(!dirty)}
            if !store.message(for:address).isEmpty {Text(store.message(for:address)).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)}
        }.disabled(store.isBusy)
        .safeAreaInset(edge:.bottom,spacing:6) {if store.isBusy {Button("Cancel") {store.cancel()}.buttonStyle(FlatButtonStyle()).frame(maxWidth:.infinity,alignment:.leading)}}
        .sheet(isPresented:$credentialSheet) {
            VStack(alignment:.leading,spacing:12) {
                Text("Hugging Face credential").font(.headline)
                field("Credential name",text:$credential)
                VStack(alignment:.leading,spacing:5) {Text("Access token").font(.caption);SecureField("hf_…",text:$token).flatField()}
                Text("Stored in this Mac’s Keychain. Only the credential name is shown here.").font(.caption).foregroundStyle(MLTheme.muted)
                if let error {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
                HStack {Button("Cancel") {token="";credentialSheet=false}.buttonStyle(FlatButtonStyle());Button("Save credential") {do {try ModelHubCredentials.save(name:credential,token:token);credentialNames=try ModelHubCredentials.names();token="";credentialSheet=false;error=nil} catch {self.error=error.localizedDescription}}.buttonStyle(FlatButtonStyle(prominent:true))}
            }.padding(20).frame(width:360).onDisappear {token=""}
        }
        .onAppear(perform:reset).onChange(of:draft) {_,_ in pending(dirty)}.onChange(of:maximum) {_,_ in pending(dirty)}.onDisappear {pending(false)}
    }
    private func field(_ title:String,text:Binding<String>)->some View {VStack(alignment:.leading,spacing:5) {Text(title).font(.caption);TextField(title,text:text).flatField()}}
    private func metadata(_ reference:ModelAssetReference)->some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {Text(reference.metadata["architecture"]?.stringValue ?? "Model").fontWeight(.semibold);Spacer();Text("Verified").foregroundStyle(accent)}
            if reference.metadata["task"] == .string("typedDecisions") {
                Text("Typed decisions · choice / ordinal / yes-no").foregroundStyle(MLTheme.muted)
                Text("CPU · encoder \(reference.metadata["maximumTokens"]?.integerValue ?? 0) tokens · head \(reference.metadata["decisionHeadTokens"]?.integerValue ?? 0) tokens").foregroundStyle(MLTheme.muted)
                Text("Inspect · Decide · Train head/full · safe package export").foregroundStyle(MLTheme.muted)
                Text("Each question produces complete option probabilities, not next-token text.").foregroundStyle(MLTheme.muted)
            } else if reference.metadata["task"] == .string("audioClassification") {
                if case .array(let labels)=reference.metadata["labels"] {Text("Audio classification · \(labels.count) classes").foregroundStyle(MLTheme.muted)}
                Text("Inspect · Classify · Export · CPU").foregroundStyle(MLTheme.muted)
                Text("16 kHz mono · up to 10 seconds/window · independent scores").foregroundStyle(MLTheme.muted)
            } else if reference.metadata["task"] == .string("automaticSpeechRecognition") {
                Text("Speech recognition · 16 kHz mono waveform").foregroundStyle(MLTheme.muted)
                Text("Inspect · Transcribe · matching processor and tokenizer").foregroundStyle(MLTheme.muted)
                Text("Local CPU inference · no ASR training or causal generation").foregroundStyle(MLTheme.muted)
            } else if reference.metadata["task"] == .string("prebuilt") {
                Text("Prepared-input numeric model · CPU").foregroundStyle(MLTheme.muted)
                if case .array(let names)=reference.metadata["featureColumns"] {Text("Feature order · "+names.compactMap(\.stringValue).joined(separator:", ")).foregroundStyle(MLTheme.muted)}
                Text("Frozen inference · no implicit preprocessing or optimizer resume").foregroundStyle(MLTheme.muted)
            } else if reference.metadata["task"] == .string("documentClassification") {
                Text("Document classifier · \(reference.metadata["vocabularySize"]?.integerValue ?? 0) fitted terms").foregroundStyle(MLTheme.muted)
                if case .array(let classes)=reference.metadata["classes"] {Text("Categories · "+classes.compactMap(\.stringValue).joined(separator:", ")).foregroundStyle(MLTheme.muted)}
                Text("Text table → category + probabilities · Predict · Evaluate").foregroundStyle(MLTheme.muted)
            } else {
            Text("Causal language model · \(reference.metadata["vocabularySize"]?.integerValue ?? 0) vocabulary IDs").foregroundStyle(MLTheme.muted)
            Text("Context · \(reference.metadata["maximumTokens"]?.integerValue ?? 0) tokens").foregroundStyle(MLTheme.muted)
            Text("Inspect · Tokenize · Generate · Export").foregroundStyle(MLTheme.muted)
            InspectorSection("Inputs & outputs",symbol:"arrow.left.arrow.right",identifier:"load-model.signature",initiallyExpanded:false,accent:accent) {
                Grid(alignment:.leading) {GridRow {Text("Name");Text("Shape");Text("Type")}.fontWeight(.semibold);GridRow {Text("input_ids");Text("N × T");Text("int64")};GridRow {Text("attention_mask");Text("N × T");Text("int64")};GridRow {Text("logits");Text("N × T × V");Text("float")}}.font(.caption.monospaced())
                Text("N observations · T tokens · V vocabulary").font(.caption2).foregroundStyle(MLTheme.muted)
            }
            }
        }.font(.caption).padding(9).overlay(Rectangle().stroke(MLTheme.rule))
    }
}
