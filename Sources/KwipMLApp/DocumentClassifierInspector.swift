import SwiftUI
import KwipMLCore

struct DocumentClassifierInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @State private var draft:[String:JSONValue]=[:]
    @State private var original:[String:JSONValue]=[:]
    @State private var error:String?
    @Environment(\.colorScheme) private var scheme
    private var accent:Color {scheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary}
    private var settings:[String:JSONValue] {if case .object(let c)=draft["settings"] {return c};return DocumentClassificationConfiguration.estimatorDefaults}
    private var textColumn:String {if case .array(let values)=draft["featureColumns"] {return values.first?.stringValue ?? ""};return ""}
    private var pending:Bool {draft != original}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("scikit-learn · Logistic regression · CPU").font(.caption.weight(.semibold))
            Text("Single-label binary or multiclass · frozen TF-IDF preparation").font(.caption).foregroundStyle(MLTheme.muted)
            VStack(alignment:.leading,spacing:4) {Text("Operation");FlatChoiceControl(title:"Classifier operation",selection:draft["operation"]?.stringValue ?? "fit",options:[.init(id:"fit",title:"Fit and validate"),.init(id:"predict",title:"Predict unseen documents"),.init(id:"evaluate",title:"Evaluate labeled documents")],accent:accent){draft["operation"] = .string($0)}}
            InspectorSection("Data mapping",symbol:"tablecells",identifier:"document-classifier.mapping",initiallyExpanded:true,accent:accent) {
                field("Prepared text column",value:Binding(get:{textColumn},set:{draft["featureColumns"] = .array([.string($0)])}))
                if draft["operation"] != .string("predict") {field("Category column",value:Binding(get:{draft["targetColumn"]?.stringValue ?? ""},set:{draft["targetColumn"] = .string($0)}))}
                Text((draft["operation"] == nil || draft["operation"] == .string("fit")) ? "Connect Training and Validation from Text encoding → Document features. Labels must be strings; join them by document identity before Partition.":"Connect Load model → Document classifier package and an unseen text table. Predict needs no labels; Evaluate requires string categories. The frozen processor is never refitted.").foregroundStyle(MLTheme.muted)
            }
            if draft["operation"] == nil || draft["operation"] == .string("fit") {
            InspectorSection("Solver",symbol:"slider.horizontal.3",identifier:"document-classifier.solver",initiallyExpanded:true,accent:accent) {
                Text("L2 regularization · lbfgs").foregroundStyle(MLTheme.muted)
                VStack(alignment:.leading,spacing:4) {Text("Class weights");FlatChoiceControl(title:"Class weights",selection:settings["classWeight"]?.stringValue ?? "none",options:[.init(id:"none",title:"None"),.init(id:"balanced",title:"Balanced from Training")],accent:accent){set("classWeight",.string($0))}}
                numeric("Inverse regularization C",key:"C",minimum:1e-8,maximum:1e8)
                numeric("Maximum solver iterations",key:"maxIterations",minimum:1,maximum:10000,integer:true)
                numeric("Convergence tolerance",key:"tolerance",minimum:1e-12,maximum:1)
                Toggle("Fit intercept",isOn:Binding(get:{settings["fitIntercept"] == true},set:{set("fitIntercept",.bool($0))})).toggleStyle(.checkbox)
                Text("Training fits coefficients; Validation evaluates. Test stays held out. This full-batch solver does not report neural epochs or continue optimizer state.").foregroundStyle(MLTheme.muted)
            }
            }
            if let operation=draft["operation"]?.stringValue,operation != "fit" {
                InspectorSection("Review",symbol:"flag",identifier:"document-classifier.review",initiallyExpanded:true,accent:accent) {
                    Toggle("Flag low top probability",isOn:Binding(get:{draft["reviewThreshold"] != nil},set:{if $0 {draft["reviewThreshold"] = .number(0.6)} else {draft.removeValue(forKey:"reviewThreshold")}})).toggleStyle(.checkbox)
                    if draft["reviewThreshold"] != nil {
                        VStack(alignment:.leading,spacing:4) {Text("Review threshold (0–1)");BoundedNumberField(title:"Review threshold",value:.decimal(draft["reviewThreshold"]?.doubleValue ?? 0.6),domain:.decimal(minimum:0.000001,maximum:1),origin:context.numericOrigin(key:"reviewThreshold"),currentOrigin:{context.currentNumericOrigin(key:"reviewThreshold")},onDraftChange:{o,p in context.setNumericPending(origin:o,key:"reviewThreshold",pending:p)}) {v in draft["reviewThreshold"] = .number(v.double);return true}}
                    }
                    Text("Flags review without changing the predicted label. Class probabilities are not calibrated certainty or unknown-category detection.").foregroundStyle(MLTheme.muted)
                }
            }
            HStack {Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending);Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)}
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).onAppear(perform:reload).onChange(of:context.contextIdentity){_,_ in reload()}
            .onChange(of:draft){_,_ in report()}.onDisappear {context.setNumericPending(origin:context.numericOrigin(key:"documentClassifier"),key:"documentClassifier",pending:false)}
    }
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"documentClassifier"),key:"documentClassifier",pending:pending)}
    private func reload(){original=context.configuration;draft=original;error=nil;report()}
    private func apply(){do {_=try WorkflowEstimatorConfiguration.decode(draft);if context.authority.configure(draft,expected:original) {original=draft;error=nil;report()} else {error=context.statusMessage}}catch {self.error=error.localizedDescription}}
    private func set(_ key:String,_ value:JSONValue){var c=settings;c[key]=value;draft["settings"] = .object(c)}
    private func field(_ title:String,value:Binding<String>)->some View {VStack(alignment:.leading,spacing:4){Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:value).flatField()}}
    private func numeric(_ title:String,key:String,minimum:Double,maximum:Double,integer:Bool=false)->some View {
        VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:integer ? .wholeDisplay(settings[key]?.doubleValue ?? minimum):.decimal(settings[key]?.doubleValue ?? minimum),domain:integer ? .integer(minimum:minimum,maximum:maximum):.decimal(minimum:minimum,maximum:maximum),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{o,p in context.setNumericPending(origin:o,key:key,pending:p)}) {value in
                switch value {case .integer(let n):set(key,.integer(n));case .decimal(let n):set(key,.number(n))};return true
            }
        }
    }
}
