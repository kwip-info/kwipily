import SwiftUI
import KwipMLCore

struct FineTuningSaveEditorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    let requestClose:()->Void
    let registerDiscard:(@escaping ()->Void,@escaping ()->Bool)->Void
    @State private var original:[String:JSONValue]=[:]
    @State private var draft:[String:JSONValue]=[:]
    @State private var checked:[String:JSONValue]?
    @State private var message=""
    @State private var closing=false
    private var reference:ModelAssetReference? {(try? WorkflowTrainedModelSource.resolve(in:host.owner.document,consumer:context.authority.address)).flatMap{host.modelLoader.reference($0.address)}}
    private var typed:Bool {reference?.metadata["task"] == .string("typedDecisions")}
    private var lora:Bool {reference?.metadata["fineTuningMode"] == .string("lora")}
    private var content:String {(try? WorkflowSaveModelConfiguration.decode(draft).effectiveContent(isLoRA:lora).rawValue) ?? "full"}
    var body:some View {
        VStack(spacing:0){
            HStack{VStack(alignment:.leading,spacing:5){Text("Save model").font(.title2.weight(.semibold));Text("Choose what the saved artifact can do").font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close",action:close).buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider()
            HStack(alignment:.top,spacing:0){
                VStack(alignment:.leading,spacing:12){purpose("Standalone inference model",lora ? "merged":"full");if lora {purpose("Adapter + base identity","adapter")};if reference?.metadata["continuation"] == .string("cpuEpochBoundary"){purpose("Training resume package","continuation")};Text("Options follow the connected result's verified capabilities.").font(.caption).foregroundStyle(MLTheme.muted);Spacer()}.padding(16).frame(width:270)
                Divider()
                ScrollView{VStack(alignment:.leading,spacing:18){
                    Text(content=="continuation" ? "Latest completed training boundary":content=="adapter" ? "Adapter with exact base dependency":"Selected inference model").font(.title3.weight(.semibold))
                    group("Source") {Text("Connected result · "+(reference?.directoryID.prefix(8).description ?? "Run the connected Train first")).font(.caption);Text(content=="continuation" ? "Latest epoch \(reference?.metadata["completedEpoch"]?.integerValue ?? 0) · CPU-qualified optimizer continuation":"Selected inference epoch \(reference?.metadata["selectedEpoch"]?.integerValue ?? 0)").font(.caption);Text("Selected best inference weights and the terminal optimizer boundary are different artifacts.").font(.caption).foregroundStyle(MLTheme.muted)}
                    group("Format & destination") {Text("Format").font(.caption);Text(typed ? "Laya typed-decision inference package":"Hugging Face safe model folder"+(content=="continuation" ? " with Kwip resume metadata":"")).font(.callout);if !typed && (content=="full" || content=="merged") {Text("Maximum inference shard size").font(.caption);FlatChoiceControl(title:"Maximum shard size",selection:String(draft["shardBytes"]?.integerValue ?? 2_000_000_000),options:[100_000_000,500_000_000,1_000_000_000,2_000_000_000].map{.init(id:String($0),title:"\($0/1_000_000) MB")}){if let bytes=Int64($0){draft["shardBytes"] = .integer(bytes)}}};Text("Destination on this Mac").font(.caption);Text(host.modelDestinationSummary(for:context.authority.address)).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled);Button("Choose folder…"){if !host.selectSaveModelDestination(for:context.authority.address,lease:context.authority.token){message=host.statusMessage}}.buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent || draft != original);if draft != original {Text("Apply or Revert these settings before choosing a folder.").font(.caption).foregroundStyle(MLTheme.muted)}}
                    group("Required contents · read-only") {if typed{Text("Encoder + decision head\nMatching tokenizer and exact budgets\nQuestion-set revision and state mapping\nCalibration + review policy\nBase/source hashes and notices").font(.callout);Text("One safe weights file, within the2 GB model limit. Weight reuse starts a fresh optimizer. No exact resume or raw training records included.").font(.caption).foregroundStyle(MLTheme.muted)}else{Text(content=="continuation" ? "✓ Latest model / adapter\n✓ Optimizer, scheduler and RNG\n✓ Original base + tokenizer for LoRA\n✓ Data, runtime and settings identities":content=="adapter" ? "✓ Adapter and additional modules\n✓ Matching tokenizer and configuration\n✓ Exact original base-file identity":"✓ Full or merged safe weights\n✓ Matching tokenizer and configuration\n✓ Task identity and provenance").font(.callout);Text(content=="continuation" ? "Requires the matching original or retained Training and Validation inputs. Raw documents are not included by default. CPU runtime pins must match.":content=="adapter" ? "Loading needs the exact matching local base. No automatic download. No optimizer-resume claim.":"No original base required at inference. No optimizer state or raw training documents included.").font(.caption).foregroundStyle(MLTheme.muted)}
                    }
                    Text("Apply saves configuration. Run Flow or Save model writes to an unused destination, with verified atomic publication.").font(.caption).foregroundStyle(MLTheme.muted)
                }.padding(24).frame(maxWidth:.infinity,alignment:.leading)}
            }
            Divider()
            VStack(alignment:.leading,spacing:10){Text(message.isEmpty ? "Draft · validate selected contents":message).font(.caption).foregroundStyle(MLTheme.muted);HStack{Button("Revert",action:load).buttonStyle(FlatButtonStyle());Spacer();Button("Validate",action:validate).buttonStyle(FlatButtonStyle()).disabled(!context.isCurrent);Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!context.isCurrent || checked != draft)}}.padding(16)
        }.frame(minWidth:800,idealWidth:1000,minHeight:610,idealHeight:760).background(MLTheme.background).onAppear{load();registerDiscard({discard()}, {!closing && draft != original})}.onDisappear{discard()}.onChange(of:draft){_,_ in report()}
    }
    private func group<C:View>(_ title:String,@ViewBuilder content:()->C)->some View {VStack(alignment:.leading,spacing:12){Text(title).font(.headline);content()}.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))}
    private func purpose(_ title:String,_ value:String)->some View {Button{draft["format"] = .string("huggingFace");draft["content"] = .string(value)}label:{Text(title).frame(maxWidth:.infinity,alignment:.leading).padding(12).background(content==value ? MLTheme.rose.opacity(0.16):MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))}.buttonStyle(.plain)}
    private func load(){guard context.isCurrent else{return};original=context.configuration;draft=original;checked=nil;message="";report()}
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"fineSaveDraft"),key:"fineSaveDraft",pending:!closing && draft != original)}
    private func validate(){do{_=try WorkflowSaveModelConfiguration.decode(draft);guard let reference,reference.metadata["trainingOperation"] == .bool(true) else{throw WorkflowArchiveError("Run and verify the connected imported Train first.")};guard content != "continuation" || reference.metadata["continuation"] == .string("cpuEpochBoundary") else{throw WorkflowArchiveError("Exact resume export requires a CPU boundary.")};checked=draft;message="Contents selected. Export independently verifies all artifact pins before writing."}catch{checked=nil;message=error.localizedDescription}}
    private func apply(){guard checked==draft else{return};context.setNumericPending(origin:context.numericOrigin(key:"fineSaveDraft"),key:"fineSaveDraft",pending:false);guard context.authority.configure(draft,expected:original)else{message=context.statusMessage;report();return};closing=true;requestClose()}
    private func discard(){guard !closing else{return};closing=true;context.setNumericPending(origin:context.numericOrigin(key:"fineSaveDraft"),key:"fineSaveDraft",pending:false)}
    private func close(){requestClose()}
}
