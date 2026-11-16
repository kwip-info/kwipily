import SwiftUI
import KwipMLCore

struct SaveModelInspectorView:View {
    @State private var fineEditor:ExpandedConfigurationWindowController?
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    private var active:Bool {
        context.isCurrent && !context.isBusy && !host.isRetired
            && host.owner === context.authority.owner && !host.owner.hasPendingInspectorEdits
    }
    private var isDocument:Bool {(try? WorkflowTrainedModelSource.resolve(in:host.owner.document,consumer:context.authority.address).isEstimator) == true}
    private var isPrebuilt:Bool {guard let source=try? WorkflowTrainedModelSource.resolve(in:host.owner.document,consumer:context.authority.address),let node=host.owner.document.nodes.first(where:{$0.address==source.address}),case .estimator(let c)=node.payload else {return false};return PrebuiltModelConfiguration.isPrebuilt(c)}
    private var isTyped:Bool {
        guard let source=try? WorkflowTrainedModelSource.resolve(in:host.owner.document,consumer:context.authority.address) else {return false}
        if host.modelLoader.reference(source.address)?.metadata["task"] == .string("typedDecisions") {return true}
        guard let node=host.owner.document.nodes.first(where:{$0.address==source.address}),case .train(let config)=node.payload,case .object(let fine)=config["fineTuning"] else {return false}
        return fine["task"] == .string("typedDecisions")
    }
    var body:some View {
        let expected=context.configuration
        VStack(alignment:.leading,spacing:10) {
            if let source=try? WorkflowTrainedModelSource.resolve(in:host.owner.document,consumer:context.authority.address),host.modelLoader.reference(source.address)?.metadata["trainingOperation"] == .bool(true) {
                Button("Configure saved contents…") {
                    if let fineEditor,!fineEditor.didClose{fineEditor.present()}else{let appearance=host.configurationAppearance;let next=ExpandedConfigurationWindowController(title:"Save model",identifier:"info.kwip.KwipML.fine-save-editor"){controller in AnyView(FineTuningSaveEditorView(context:context,host:host,requestClose:{[weak controller] in controller?.close()},registerDiscard:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty}).windowActivatingControls().appAppearance(appearance))};fineEditor=next;next.present()}
                }.buttonStyle(FlatButtonStyle())
            }
            Text("Format").font(.caption)
            FlatChoiceControl(title:"Model format",selection:context.saveModelConfiguration?.format.rawValue ?? "",options:isTyped ? [.init(id:"huggingFace",title:"Laya typed-decision package")]:isDocument ? [.init(id:"package",title:isPrebuilt ? "Prepared-input inference package":"Classifier + text preparation")]:host.isSelfPlaySource(context.authority.address) ? [.init(id:"package",title:"Paired policy/value inference package")]:[
                .init(id:"package",title:"Portable package"),.init(id:"checkpoint",title:"Checkpoint (.pt)"),.init(id:"huggingFace",title:"Hugging Face folder")]) {value in
                guard let format=WorkflowSaveModelConfiguration.Format(rawValue:value) else {return}
                _=context.setSaveModelFormat(format,expected:expected)
            }
            if isPrebuilt {
                Text("Preparation").font(.caption)
                FlatChoiceControl(title:"Saved preparation",selection:context.saveModelConfiguration?.preparation.rawValue ?? "none",options:[.init(id:"none",title:"Prepared inputs only")]+(host.learnedPreparationAvailable(for:context.authority.address) ? [.init(id:"learned",title:"Model + learned preparation")]:[])) {value in
                    var next=expected;next["preparation"] = .string(value);_=context.authority.configure(next,expected:expected)
                }
                if !host.learnedPreparationAvailable(for:context.authority.address) {Text("Learned preparation requires a completed fit with a supported Preprocess chain.").font(.caption).foregroundStyle(MLTheme.muted)}
            }
            if host.isSelfPlaySource(context.authority.address) {
                Text("✓ Explicit policy and value heads\n✓ Shared architecture and trained weights\n✓ Task version, rule/encoding provenance\nPortable inference only · optimizer/replay resume stays in recorded Self-play History.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            if isDocument {
                Text(isPrebuilt ? "✓ Inert model state\n✓ Exact feature and class order\n✓ Algorithm, options and runtime identity":"✓ Vocabulary and IDF\n✓ Class names and exact order\n✓ Coefficients, feature settings and provenance").font(.caption).foregroundStyle(ComponentVisualFamily.training.primary)
                Text(isPrebuilt ? (context.saveModelConfiguration?.preparation == .learned ? "Includes frozen learned rules. Reuse through Load model → Preprocess → Apply saved rules. Earlier fixed typing/cleanup stays explicit; no training rows included.":"Prepared numeric inputs only · upstream transforms are not bundled. No training rows or optimizer resume."):"Required inference contents · no optimizer resume. Reload this package to classify unseen text without fitting again. Source documents are not included.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            if context.saveModelConfiguration?.format == .huggingFace {
                VStack(alignment:.leading,spacing:6) {
                    if let source=try? WorkflowTrainedModelSource.resolve(in:host.owner.document,consumer:context.authority.address),let reference=host.modelLoader.reference(source.address),reference.metadata["trainingOperation"] == .bool(true) {
                        let lora=reference.metadata["fineTuningMode"] == .string("lora")
                        let options:[FlatChoiceOption] = (lora ? [.init(id:"adapter",title:"LoRA adapter + base identity"),.init(id:"merged",title:"Merged standalone model")]:[.init(id:"full",title:"Full fine-tuned model")]) + (reference.metadata["continuation"] == .string("cpuEpochBoundary") ? [.init(id:"continuation",title:"Latest training boundary")]:[])
                        Text("Contents").font(.caption)
                        FlatChoiceControl(title:"Save contents",selection:context.saveModelConfiguration?.effectiveContent(isLoRA:lora).rawValue ?? "full",options:options) {value in var next=expected;next["content"] = .string(value);_=context.authority.configure(next,expected:expected)}
                        if context.saveModelConfiguration?.content == .continuation {Text("Latest completed epoch · consistent model, optimizer and RNG").font(.caption).foregroundStyle(MLTheme.muted)}
                    }
                    Text("Original dtype · safe tensors").font(.caption)
                    Text(isTyped ? "✓ Encoder/head configuration and weights\n✓ Matching tokenizer\n✓ Immutable questions and calibration provenance\nWeights reuse · fresh optimizer; matching Kwip adapter required":"✓ Model configuration\n✓ Matching tokenizer\n✓ Model identity and provenance").font(.caption).foregroundStyle(ComponentVisualFamily.training.primary)
                    if !isTyped {
                    Text("Maximum shard size").font(.caption)
                    FlatChoiceControl(title:"Maximum shard size",selection:String(context.saveModelConfiguration?.shardBytes ?? 2_000_000_000),options:[.init(id:"100000000",title:"100 MB"),.init(id:"500000000",title:"500 MB"),.init(id:"1000000000",title:"1 GB"),.init(id:"2000000000",title:"2 GB")]) {value in
                        guard let bytes=Int64(value) else {return};var next=expected;next["shardBytes"] = .integer(bytes);_=context.authority.configure(next,expected:expected)
                    }
                    }
                }
            }
            Text("Destination on this Mac").font(.caption)
            Text(host.modelDestinationSummary(for:context.authority.address)).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            HStack {
                Button("Choose destination…") {_=host.selectSaveModelDestination(for:context.authority.address,lease:context.authority.token)}.buttonStyle(FlatButtonStyle())
                Button("Clear") {_=host.clearSaveModelDestination(for:context.authority.address,lease:context.authority.token)}.buttonStyle(FlatButtonStyle())
            }
            Button("Save model") {_=host.saveModel(address:context.authority.address)}.buttonStyle(FlatButtonStyle(prominent:true))
                .disabled(context.saveModelConfiguration == nil)
                .accessibilityIdentifier("workflow.save-model.save")
            if context.saveModelConfiguration == nil {Text("Choose a supported model format before saving.").font(.caption).foregroundStyle(MLTheme.rose)}
            Text(host.statusMessage).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
        }.disabled(!active).onDisappear{fineEditor?.close();fineEditor=nil}.onAppear{host.reloadSelfPlayRecords()}
    }
}
