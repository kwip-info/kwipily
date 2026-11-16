import SwiftUI
import KwipMLCore

struct PrebuiltModelInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    let host:WorkflowWorkspaceSession?
    @State private var editor:ExpandedConfigurationWindowController?
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Prebuilt models").font(.caption.weight(.semibold))
            Text(summary).font(.caption).foregroundStyle(MLTheme.muted)
            Button("Configure model…") {
                if let editor,!editor.didClose {editor.present();return}
                let next=ExpandedConfigurationWindowController(title:"Prebuilt models",identifier:"info.kwip.KwipML.prebuilt-editor") {controller in
                    let content=PrebuiltModelEditor(context:context,host:host,requestClose:{[weak controller] in controller?.close()},registerDiscard:{[weak controller] discard,dirty in controller?.discardDraft=discard;controller?.hasUnappliedDraft=dirty}).windowActivatingControls()
                    if let host {return AnyView(content.appAppearance(host.configurationAppearance))}
                    return AnyView(content)
                };editor=next;next.present()
            }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
            Text("Apply saves configuration. Run Flow performs the selected operation.").font(.caption).foregroundStyle(MLTheme.muted)
        }.onDisappear{editor?.close();editor=nil}
    }
    private var summary:String {
        if let a=context.configuration["algorithm"]?.stringValue,let entry=PrebuiltModelConfiguration.algorithms[a] {return entry.title+" · "+(context.configuration["operation"]?.stringValue ?? "fit")+" · CPU"}
        return DocumentClassificationConfiguration.isDocument(context.configuration) ? "Document classification · frozen TF-IDF · CPU":context.configuration["backendID"] == .string("tensorflow") ? "Existing Keras classifier · CPU":"Existing logistic regression · CPU"
    }
}

struct PrebuiltModelEditor:View {
    @ObservedObject var context:WorkflowControlNodeContext
    let host:WorkflowWorkspaceSession?
    let requestClose:()->Void
    let registerDiscard:(@escaping ()->Void,@escaping ()->Bool)->Void
    @State private var original:[String:JSONValue]=[:]
    @State private var draft:[String:JSONValue]=[:]
    @State private var section="Task"
    @State private var message=""
    @State private var numeric=Set<String>()
    @State private var mappingColumns:[DataColumnSchema]=[]
    private var sections:[String] {["Task","Data mapping","Algorithm"]+(document ? ["Review"]:[])}
    private var document:Bool {DocumentClassificationConfiguration.isDocument(draft)}
    private var keras:Bool {draft["backendID"] == .string("tensorflow")}
    private var mode:String {prebuilt ? entry?.task ?? "classification":document ? "document":keras ? "keras":"legacy"}
    private var taskOptions:[(String,String)] {[("classification","Classification"),("regression","Regression"),("clustering","Clustering"),("document","Document classification"),("keras","Keras dense classifier")]+(!PrebuiltModelConfiguration.hasTables(original) && original["backendID"] != .string("tensorflow") ? [("legacy","Existing logistic project")]:[])}
    private var prebuilt:Bool {PrebuiltModelConfiguration.isPrebuilt(draft)}
    private var entry:PrebuiltModelConfiguration.Algorithm? {PrebuiltModelConfiguration.algorithms[draft["algorithm"]?.stringValue ?? ""]}
    private var features:[String] {if case .array(let raw)=draft["featureColumns"] {return raw.compactMap(\.stringValue)};return []}
    private var settings:[String:JSONValue] {if case .object(let raw)=draft["settings"] {return raw};return [:]}
    private var pending:Bool {draft != original || !numeric.isEmpty}
    private var columns:[DataColumnSchema] {mappingColumns}
    private func refreshColumns(){mappingColumns=host?.sourceProvider.recordedSchema(for:context.authority.address,in:context.authority.owner.document,inputPort:(draft["operation"]?.stringValue ?? "fit") == "fit" ? "training":"table") ?? []}
    var body:some View {
        VStack(spacing:0) {
            HStack {VStack(alignment:.leading,spacing:5){Text("Prebuilt models").font(.title2.weight(.semibold));Text(context.authoredName).font(.caption).foregroundStyle(MLTheme.muted)};Spacer();Button("Close",action:close).buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}.padding(20)
            if !context.isCurrent {Text("The original selection changed. This draft is retained for reference; it cannot be applied. Close and reopen the original component to edit it.").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,20).padding(.bottom,10)}
            Divider()
            HStack(spacing:0) {
                VStack(alignment:.leading,spacing:5) {ForEach(sections,id:\.self) {name in Button{section=name}label:{Text(name).frame(maxWidth:.infinity,alignment:.leading).padding(10).background(section==name ? MLTheme.rose.opacity(0.16):Color.clear).contentShape(Rectangle())}.buttonStyle(.plain)};Spacer()}.padding(12).frame(width:200)
                Divider()
                ScrollView {VStack(alignment:.leading,spacing:16){Text(section).font(.title3.weight(.semibold));content}.padding(24).frame(maxWidth:.infinity,alignment:.leading)}
            }
            Divider()
            VStack(alignment:.leading,spacing:8){if !message.isEmpty{Text(message).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
                HStack {Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending);Spacer();Text("Apply saves settings. Run Flow executes.").font(.caption).foregroundStyle(MLTheme.muted);Button("Apply",action:apply).buttonStyle(FlatButtonStyle(prominent:true)).disabled(!pending || !numeric.isEmpty || context.isBusy)}
            }.padding(16)
        }.frame(minWidth:800,minHeight:640).background(MLTheme.background).onAppear{reload();registerDiscard({discard()}, {pending})}.onDisappear{discard()}.onChange(of:draft){_,_ in report()}.onChange(of:section){_,name in if name=="Data mapping" {refreshColumns()}}.onChange(of:draft["operation"]){_,_ in refreshColumns()}.onChange(of:context.contextIdentity){_,_ in if !pending {reload()}}
    }
    @ViewBuilder private var content:some View {
        if section=="Task" {
            choice("Task",selection:mode,options:taskOptions) {task in
                numeric=[]
                if task=="document" {draft=DocumentClassificationConfiguration.defaults}
                else if task=="legacy" {draft=WorkflowEstimatorConfiguration.sklearnDefaults;draft["featureColumns"] = .array([.string("feature"),.string("second")]);draft["targetColumn"] = .string("group")}
                else if task=="keras" {draft=["backendID":.string("tensorflow"),"featureColumns":.array([.string("feature"),.string("second")]),"targetColumn":.string("group"),"settings":.object(TensorFlowEstimatorConfiguration.defaultSettings)]}
                else {draft=PrebuiltModelConfiguration.defaults(task=="regression" ? "ridge":task=="clustering" ? "kmeans":"logistic")}
            }
            if prebuilt {
                if let model=loadedModel,case .object = model.metadata["prebuiltConfiguration"] {Button("Use loaded model settings",action:useLoadedModel).buttonStyle(FlatButtonStyle())}
                choice("Operation",selection:draft["operation"]?.stringValue ?? "fit",options:entry?.predict == false ? [("fit","Fit population")]:entry?.task=="clustering" ? [("fit","Fit population"),("predict","Assign new rows")]:[("fit","Fit and validate"),("predict","Predict new rows"),("evaluate","Evaluate labeled rows")]) {draft["operation"] = .string($0)}
                note(entry?.task=="clustering" ? "Cluster identities describe similarity. Memberships and distances are not churn probabilities. DBSCAN provides fit-only analysis.":"Fit learns only Training; Validation stays held out. Predict/Evaluate reuse the frozen package, never refit.")
            } else if document {
                choice("Operation",selection:draft["operation"]?.stringValue ?? "fit",options:[("fit","Fit and validate"),("predict","Predict unseen documents"),("evaluate","Evaluate labeled documents")]){draft["operation"] = .string($0)}
                note("Training-only TF-IDF and single-label document classification. Frozen Predict/Evaluate use the saved processor without refitting.")
            } else {note("Original two-feature project contract · CPU fit and validation. Saved defaults are preserved; choosing a different task is an explicit configuration change.") }
        } else if section=="Data mapping" {
            Text("Ordered features").font(.headline)
            if columns.isEmpty {note("Connect and refresh a source to populate column choices. Explicit feature names remain editable; no external file is read while editing.")}
            ForEach(features.indices,id:\.self) {i in
                HStack {Text(String(i+1)).font(.caption.monospaced());if columns.isEmpty {TextField("Feature column",text:Binding(get:{features.indices.contains(i) ? features[i]:""},set:{v in var names=features;names[i]=v;setFeatures(names)})).flatField()} else {FlatChoiceControl(title:"Feature \(i+1)",selection:features[i],options:columns.filter{document ? $0.dtype=="string":["float64","int64"].contains($0.dtype)}.map{.init(id:$0.name,title:$0.name)}){v in var names=features;names[i]=v;setFeatures(names)}};Button("Remove"){var names=features;names.remove(at:i);setFeatures(names)}.buttonStyle(FlatButtonStyle()).disabled(!prebuilt || features.count<=1)}
            }
            if prebuilt {Button("Add feature"){setFeatures(features+[columns.first(where:{!features.contains($0.name) && ["float64","int64"].contains($0.dtype)})?.name ?? ""])}.buttonStyle(FlatButtonStyle()).disabled(features.count>=128)}
            if entry?.task=="clustering" {note("Target · Not used")}else if draft["operation"] != .string("predict") {if columns.isEmpty {field("Target column",key:"targetColumn")}else{choice("Target column",selection:draft["targetColumn"]?.stringValue ?? "",options:columns.map{($0.name,$0.name)}){draft["targetColumn"] = .string($0)}}}
            note("Source row identities and scalar columns are preserved in Predictions/Assignments. Identifiers are not automatically features. Impute, encode and scale explicitly before this tool; learned preparation must fit Training only.")
        } else if section=="Algorithm" {
            if prebuilt {
            choice("Algorithm",selection:draft["algorithm"]?.stringValue ?? "",options:PrebuiltModelConfiguration.algorithms.filter{$0.value.task==entry?.task}.sorted{$0.value.title<$1.value.title}.map{($0.key,$0.value.title)}) {a in
                let previous=draft;numeric=[];draft=PrebuiltModelConfiguration.defaults(a);draft["featureColumns"]=previous["featureColumns"];draft["targetColumn"]=previous["targetColumn"]
            }
            if let entry {ForEach(entry.fields,id:\.key){field in
                if field.kind=="bool" {Toggle(field.title,isOn:Binding(get:{settings[field.key] == .bool(true)},set:{setOption(field.key,.bool($0))})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}else{number(field)}
            }}
            } else if keras {
                Text("Keras dense classifier · existing model definition connection").font(.headline)
                ForEach(TensorFlowEstimatorConfiguration.fields,id:\.key){field in
                    if field.key != "device" {switch field.kind {case .integer(let range):numberControl(field.title,key:field.key,minimum:Double(range.lowerBound),maximum:Double(range.upperBound),integer:true)
                    case .positiveDecimal(let maximum),.nonnegativeDecimal(let maximum):numberControl(field.title,key:field.key,minimum:0,maximum:maximum,integer:false)
                    case .choice:EmptyView()}}
                }
            } else {
                Text(document ? "Document TF-IDF · L2/lbfgs logistic regression":"Original binary L2/lbfgs logistic regression").font(.headline)
                if let logistic=PrebuiltModelConfiguration.algorithms["logistic"] {ForEach(logistic.fields,id:\.key){f in if f.kind=="bool" {Toggle(f.title,isOn:Binding(get:{settings[f.key] == true},set:{setOption(f.key,.bool($0))})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}else{number(f)}}}
                if document {choice("Class weights",selection:settings["classWeight"]?.stringValue ?? "none",options:[("none","None"),("balanced","Balanced from Training")]){setOption("classWeight",.string($0))}}
            }
            if draft["algorithm"] == .string("dbscan") {note("Noise is −1. Maximum 2,000 fit rows bounds worst-case neighborhood memory. No Model output or new-row prediction.")}
        } else if section=="Review" {
            if document {
                Toggle("Flag low top probability",isOn:Binding(get:{draft["reviewThreshold"] != nil},set:{if $0 {draft["reviewThreshold"] = .number(0.6)}else{draft.removeValue(forKey:"reviewThreshold")}})).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))
                if draft["reviewThreshold"] != nil {reviewThreshold}
                note("Held-out document metrics and complete predictions use the frozen Training vocabulary. Review flags do not change the predicted category or establish calibrated certainty.")
            } else if !prebuilt {note("Existing held-out binary classifier metrics retain their original semantics. No neural gradient crosses preprocessing.")}
            else {note(entry?.task=="classification" ? "Held-out accuracy, log loss, macro precision/recall/F1 and confusion matrix. Unknown Validation categories are refused; probabilities are not calibrated certainty.":entry?.task=="regression" ? "Held-out MAE, RMSE and R². R² is unavailable for fewer than two rows or a constant target.":"Complete populations and noise counts. Silhouette uses at most 512 seeded observations, excludes noise and reports coverage; undefined cases remain unavailable. Profiles show means in the actual prepared feature coordinates. Fitting never silently samples.")}


        }
    }
    private var reviewThreshold:some View {BoundedNumberField(title:"Review threshold",value:.decimal(draft["reviewThreshold"]?.doubleValue ?? 0.6),domain:.decimal(minimum:0.000001,maximum:1),origin:context.numericOrigin(key:"reviewThreshold"),currentOrigin:{context.currentNumericOrigin(key:"reviewThreshold")},onDraftChange:{_,pending in if pending {numeric.insert("reviewThreshold")}else{numeric.remove("reviewThreshold")};report()}){value in draft["reviewThreshold"] = .number(value.double);return true}}
    private func note(_ text:String)->some View {Text(text).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true).padding(12).frame(maxWidth:.infinity,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))}
    private func field(_ title:String,key:String)->some View {VStack(alignment:.leading,spacing:5){Text(title).font(.caption);TextField(title,text:Binding(get:{draft[key]?.stringValue ?? ""},set:{draft[key] = .string($0)})).flatField()}}
    private func choice(_ title:String,selection:String,options:[(String,String)],action:@escaping(String)->Void)->some View {VStack(alignment:.leading,spacing:5){Text(title).font(.caption);FlatChoiceControl(title:title,selection:selection,options:options.map{.init(id:$0.0,title:$0.1)}){action($0)}}}
    private func number(_ field:PrebuiltModelConfiguration.Field)->some View {numberControl(field.title,key:field.key,minimum:field.minimum ?? 0,maximum:field.maximum ?? 1,integer:field.kind=="integer")}
    private func numberControl(_ title:String,key:String,minimum:Double,maximum:Double,integer:Bool)->some View {
        VStack(alignment:.leading,spacing:5){Text(title).font(.caption);BoundedNumberField(title:title,value:integer ? .wholeDisplay(settings[key]?.doubleValue ?? 0):.decimal(settings[key]?.doubleValue ?? 0),domain:integer ? .integer(minimum:minimum,maximum:maximum):.decimal(minimum:minimum,maximum:maximum),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{_,pending in if pending {numeric.insert(key)}else{numeric.remove(key)};report()}){value in setOption(key,integer ? .integer(Int64(value.double)):.number(value.double));return true}}
    }
    private var loadedModel:ModelAssetReference? {guard let host,let link=host.owner.document.edges.first(where:{$0.to.address==context.authority.address && $0.to.port=="model"}) else{return nil};return host.modelLoader.reference(link.from.address)}
    private func useLoadedModel(){guard let model=loadedModel,case .object(let c)=model.metadata["prebuiltConfiguration"],let a=c["algorithm"]?.stringValue else{return};let target=draft["targetColumn"];draft=PrebuiltModelConfiguration.defaults(a);draft["settings"]=c["options"];draft["featureColumns"]=model.metadata["featureColumns"];draft["operation"] = .string("predict");if !PrebuiltModelConfiguration.isCluster(draft){draft["targetColumn"]=target ?? .string("target")};numeric=[]}
    private func setFeatures(_ names:[String]){draft["featureColumns"] = .array(names.map(JSONValue.string))}
    private func setOption(_ key:String,_ value:JSONValue){var next=settings;next[key]=value;draft["settings"] = .object(next)}
    private func report(){context.setNumericPending(origin:context.numericOrigin(key:"prebuiltEditor"),key:"prebuiltEditor",pending:pending)}
    private func discard(){context.setNumericPending(origin:context.numericOrigin(key:"prebuiltEditor"),key:"prebuiltEditor",pending:false)}
    private func reload(){guard context.isCurrent else{return};original=context.configuration;draft=original;numeric=[];message="";refreshColumns();report()}
    private func close(){requestClose()}
    private func apply(){do{let decoded = try WorkflowEstimatorConfiguration.decode(draft);if decoded.backendID=="tensorflow" {try TensorFlowEstimatorConfiguration.validate(decoded.settings)};if context.authority.configure(draft,expected:original){original=draft;message="Applied";report()}else{message=context.statusMessage}}catch{message=error.localizedDescription}}
}
