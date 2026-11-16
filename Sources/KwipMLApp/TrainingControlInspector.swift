import SwiftUI
import KwipMLCore

struct TrainingControlInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @Environment(\.colorScheme) private var colorScheme
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary}
    @State private var draft=TrainingControlPolicy.factory
    @State private var baseline=TrainingControlPolicy.factory
    @State private var expected:[String:JSONValue]=[:]
    @State private var initialized=false
    @State private var rejection:String?
    @State private var numericDrafts:Set<String>=[]
    private var numericDraft:Bool {!numericDrafts.isEmpty}
    private let pendingKey="trainingControlDraft"
    private var dirty:Bool {draft != baseline || numericDraft}
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            Divider()
            Text("Model selection").font(.caption.weight(.semibold))
            FlatChoiceControl(title:"Model selection",selection:draft.modelSelection,options:[.init(id:"finalEpoch",title:"Final epoch"),.init(id:"bestValidation",title:"Best validation")],accent:accent) {draft.modelSelection=$0}
            Text("Early stopping").font(.caption.weight(.semibold)).padding(.top,4)
            Toggle("Enable early stopping",isOn:$draft.earlyStopping.enabled).toggleStyle(FileInputCheckboxStyle(tint:accent)).font(.caption)
            if draft.modelSelection=="bestValidation" || draft.earlyStopping.enabled {
                FlatChoiceControl(title:"Monitor",selection:draft.monitor,options:[.init(id:"validationLoss",title:"Validation loss")],accent:accent) {draft.monitor=$0}
            }
            if draft.earlyStopping.enabled {
                Text("Patience (epochs)").font(.caption)
                BoundedNumberField(title:"Early-stopping patience",value:.wholeDisplay(Double(draft.earlyStopping.patience)),domain:.integer(minimum:1,maximum:10000),origin:context.numericOrigin(key:pendingKey),currentOrigin:{context.currentNumericOrigin(key:pendingKey)},onDraftChange:{_,pending in if pending {numericDrafts.insert("patience")} else {numericDrafts.remove("patience")};report()}) {candidate in
                    guard case .integer(let number)=candidate else {return false};draft.earlyStopping.patience=Int(number);numericDrafts.remove("patience");return true
                }
                Text("Minimum improvement").font(.caption)
                BoundedNumberField(title:"Early-stopping minimum improvement",value:.decimal(draft.earlyStopping.minimumImprovement),domain:.decimal(minimum:0,maximum:1_000_000),origin:context.numericOrigin(key:pendingKey),currentOrigin:{context.currentNumericOrigin(key:pendingKey)},onDraftChange:{_,pending in if pending {numericDrafts.insert("delta")} else {numericDrafts.remove("delta")};report()}) {candidate in
                    guard case .decimal(let number)=candidate else {return false};draft.earlyStopping.minimumImprovement=number;numericDrafts.remove("delta");return true
                }
            }
            HStack {
                Button("Revert",action:reload).disabled(!dirty && rejection==nil)
                Spacer()
                Button("Apply",action:apply).disabled(!dirty || numericDraft)
            }.buttonStyle(FlatButtonStyle())
            if let rejection {Text(rejection).font(.caption).foregroundStyle(.orange)}
        }.onAppear {if !initialized {reload();initialized=true}}
            .onChange(of:draft) {_,_ in report()}
            .onChange(of:context.configuration) {_,_ in if !dirty {reload()}}
    }
    private func report() {guard initialized else {return};context.setNumericPending(origin:context.numericOrigin(key:pendingKey),key:pendingKey,pending:dirty)}
    private func reload() {draft=context.trainingControlPolicy ?? .factory;baseline=draft;expected=context.configuration;numericDrafts=[];rejection=nil;report()}
    private func apply() {
        guard context.authority.accepts(expected) else {rejection="Train changed. Revert this draft before editing its current settings.";return}
        guard context.setTrainingControl(draft,expected:expected) else {rejection=context.statusMessage;return}
        reload()
    }
}
