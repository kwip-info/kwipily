import SwiftUI
import KwipMLCore

struct FineTuningInspectorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var store:ModelLoaderStore
    @State private var editor:ExpandedConfigurationWindowController?
    private var typed:Bool {if case .object(let c)=context.configuration["fineTuning"]{return c["task"] == .string("typedDecisions")};return false}
    var body:some View {
        VStack(alignment:.leading,spacing:10){
            FlatChoiceControl(title:"Training task",selection:typed ? "typed":"causal",options:[.init(id:"causal",title:"Causal text model"),.init(id:"typed",title:"Typed decisions · Laya")]){task in
                let old=context.configuration;var next=old;next["fineTuning"] = .object(task=="typed" ? TypedDecisionConfiguration.trainingDefaults:FineTuningConfiguration.defaults)
                if task=="typed"{var run:[String:JSONValue]=[:];if case .object(let existing)=next["runSetup"]{run=existing};run["device"] = .string("cpu");next["runSetup"] = .object(run)}
                _=context.authority.configure(next,expected:old)
            }.disabled(context.isBusy)
            Text(typed ? "Imported typed-decision model · matching tokenizer and head":"Imported causal model · model-matched tokenizer").font(.caption.weight(.semibold))
            if typed {Text("CPU · "+(store.isBusy ? "active operation":"head or encoder + head")+" · complete standard predictions").font(.caption).foregroundStyle(MLTheme.muted)}
            else if let fine=try? FineTuningConfiguration.decode(context.configuration) {Text("\(fine.values["mode"]?.stringValue ?? "full") · \(fine.values["exampleFormat"]?.stringValue == "promptResponse" ? "Prompt + response":"Full text") · \(fine.values["maximumTokens"]?.integerValue ?? 128) context tokens").font(.caption).foregroundStyle(MLTheme.muted)}
            Button("Configure fine-tuning…"){
                if let editor,!editor.didClose{editor.present()}else{let appearance=host.configurationAppearance;let next=ExpandedConfigurationWindowController(title:typed ? "Train · Typed decisions":"Train · Fine-tuning",identifier:"info.kwip.KwipML.fine-tuning-editor"){controller in
                    let close:()->Void = {[weak controller] in controller?.close()}
                    let register: (@escaping ()->Void,@escaping ()->Bool)->Void = {[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty}
                    if typed{return AnyView(TypedDecisionTrainEditor(context:context,host:host,store:store,requestClose:close,registerDiscard:register).windowActivatingControls().appAppearance(appearance))}
                    return AnyView(FineTuningEditorView(context:context,host:host,store:store,requestClose:close,registerDiscard:register).windowActivatingControls().appAppearance(appearance))
                };editor=next;next.present()}
            }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
            if store.isBusy,store.trainingAddress==context.authority.address {
                Button(store.trainingPauseRequested ? "Pause requested":"Pause after completed epoch"){store.pauseFineTuning()}.buttonStyle(FlatButtonStyle()).disabled(store.trainingPauseRequested)
                Button("Stop now"){store.cancel()}.buttonStyle(FlatButtonStyle())
            }
            if store.statusAddress==context.authority.address {Text(store.status).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
            Text("Apply saves settings. Run Flow starts training. Inference selection and the latest resume boundary stay separate.").font(.caption).foregroundStyle(MLTheme.muted)
        }.onAppear{store.reloadTraining()}.onDisappear{editor?.close();editor=nil}
    }
}
