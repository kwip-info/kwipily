import SwiftUI
import Combine
import KwipMLCore

/// Atomic recipe editing keeps incomplete controls separate from applied source.
@MainActor final class DataProgramLearningControls:ObservableObject {
    let componentID:String
    let fields:[ComponentField]
    @Published private(set) var values:[String:JSONValue]
    @Published private(set) var rejection:String?
    @Published private(set) var resetGeneration=0
    private let canvas:TensorLearningCanvasAdapter?
    private let lesson:TensorLearningStore?
    private let nodeID:String?
    private let session:TensorLearningDraftSession
    private let identity:UUID
    private let mode:LearningMode?
    private var baseline:[String:JSONValue]
    private var observation:AnyCancellable?
    private var prefix:String {nodeID.map {"\($0.utf8.count):\($0):"} ?? "program."}
    init(canvas:TensorLearningCanvasAdapter,nodeID:String) {
        self.canvas=canvas;lesson=nil;self.nodeID=nodeID;let owner=canvas.session;session=owner
        identity=owner.identity;mode=owner.draft?.mode
        let node=canvas.graph.nodes.first {$0.id==nodeID}
        let id=node?.type ?? "",initial=node?.config ?? [:]
        componentID=id;fields=canvas.registry.component(type:id)?.fields ?? []
        baseline=initial;values=initial
        restorePending()
        observation=session.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    init(lesson:TensorLearningStore) {
        canvas=nil;self.lesson=lesson;nodeID=nil;componentID="program_export";let owner=lesson.drafts;session=owner
        identity=owner.identity;mode=owner.draft?.mode
        fields=BundledProgramExportCapabilities.entry.configurationFields
        let initial=lesson.draft?.programExportRecipe.flatMap {try? ProgramExportConfiguration.decodeRecipe($0).values} ?? [:]
        baseline=initial;values=initial
        restorePending()
        observation=session.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    private func restorePending() {
        for field in fields {
            if let text=session.draft?.pendingFields[prefix+field.key],let data=text.data(using:.utf8),let value=try? JSONDecoder().decode(JSONValue.self,from:data),field.accepts(value) {values[field.key]=value}
        }
    }
    private var current:Bool {
        guard session.identity==identity,session.draft?.mode==mode else {return false}
        if let canvas,let nodeID {return canvas.historyContextIsActive && canvas.graph.nodes.first {$0.id==nodeID && $0.type==componentID}?.config==baseline}
        return lesson?.isActive==true && lesson?.isBusy==false && session.draft?.programExportRecipe==(try? ProgramExportConfiguration(values:baseline).recipeValue)
    }
    var origin:String {identity.uuidString+prefix+String(resetGeneration)}
    var currentOrigin:String? {current ? origin:nil}
    private var numericPending:Bool {(session.draft?.pendingFields.keys.contains {$0.hasPrefix(prefix+"_numeric.")}) ?? false}
    var hasChanges:Bool {values != baseline || numericPending}
    var canApply:Bool {current && hasChanges && !numericPending && validationMessage==nil}
    var validationMessage:String? {
        do {_=try canonical(values);return nil} catch {return error.localizedDescription}
    }
    private func canonical(_ proposed:[String:JSONValue])throws->[String:JSONValue] {
        switch componentID {
        case "dataset_source":return try DatasetConfiguration(values:proposed).values
        case "data_loader":return try DataLoaderConfiguration(values:proposed).values
        case "program_export":return try ProgramExportConfiguration(values:proposed).values
        default:throw LearningContractError("Choose a lesson recipe or program action.")
        }
    }
    @discardableResult func setField(_ key:String,value:JSONValue)->Bool {
        guard current,let field=fields.first(where:{$0.key==key}),field.accepts(value),let data=try? JSONEncoder().encode(value),let text=String(data:data,encoding:.utf8) else {return false}
        guard session.update({draft in
            if value==baseline[key] {draft.pendingFields.removeValue(forKey:prefix+key)}
            else {draft.pendingFields[prefix+key]=text}
            draft.pendingFields.removeValue(forKey:prefix+"_numeric."+key)
        }) else {rejection=session.errorMessage;return false}
        values[key]=value;rejection=nil;return true
    }
    func setNumericPending(_ key:String,pending:Bool) {
        guard current,fields.contains(where:{$0.key==key}) else {return}
        _=session.update {draft in
            if pending {draft.pendingFields[prefix+"_numeric."+key]="unfinished"}
            else {draft.pendingFields.removeValue(forKey:prefix+"_numeric."+key)}
        }
    }
    @discardableResult func apply()->Bool {
        guard current,!numericPending else {return false}
        do {
            let canonical=try canonical(values)
            if let canvas,let nodeID {
                guard canvas.configure(nodeID:nodeID,values:canonical) else {return false}
            } else {
                let recipe=try ProgramExportConfiguration(values:canonical).recipeValue
                guard session.update({$0.programExportRecipe=recipe}) else {return false}
            }
            _=session.update {draft in for key in Array(draft.pendingFields.keys) where key.hasPrefix(prefix) {draft.pendingFields.removeValue(forKey:key)}}
            baseline=canonical;values=canonical;rejection=nil;resetGeneration+=1;return true
        } catch {rejection=error.localizedDescription;return false}
    }
    @discardableResult func discard()->Bool {
        guard current else {return false}
        guard session.update({draft in for key in Array(draft.pendingFields.keys) where key.hasPrefix(prefix) {draft.pendingFields.removeValue(forKey:key)}}) else {return false}
        values=baseline;rejection=nil;resetGeneration+=1;return true
    }
}

struct DataProgramLearningControlsView:View {
    @StateObject private var context:DataProgramLearningControls
    init(canvas:TensorLearningCanvasAdapter,nodeID:String) {_context=StateObject(wrappedValue:DataProgramLearningControls(canvas:canvas,nodeID:nodeID))}
    init(lesson:TensorLearningStore) {_context=StateObject(wrappedValue:DataProgramLearningControls(lesson:lesson))}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorConfigurationSection(isEnabled:context.currentOrigin != nil) {
                ForEach(context.fields,id:\.key) {field in
                    let generation=context.resetGeneration
                    RegistryFieldControl(field:field,value:context.values[field.key] ?? field.defaultValue,origin:context.origin,currentOrigin:{context.resetGeneration==generation ? context.currentOrigin:nil},onDraftChange:{_,pending in
                        guard context.resetGeneration==generation else {return};context.setNumericPending(field.key,pending:pending)
                    },componentID:context.componentID,configuration:context.values) {value in
                        context.resetGeneration==generation && context.setField(field.key,value:value)
                    }
                }.id(context.resetGeneration)
                if let error=context.rejection ?? context.validationMessage {Label(error,systemImage:"exclamationmark.circle").font(.caption).foregroundStyle(MLTheme.rose)}
                HStack {
                    Button("Apply") {_=context.apply()}.disabled(!context.canApply)
                    Button("Discard") {_=context.discard()}.disabled(!context.hasChanges)
                    Spacer(minLength:0)
                    if context.hasChanges {Text("Unapplied changes").font(.caption2).foregroundStyle(MLTheme.muted)}
                }.buttonStyle(FlatButtonStyle())
            }
            InspectorInformationSection {
                Text(context.componentID=="program_export" ? "This action saves executable computation with its learned settings and input-size rules. It does not add a forward layer. Only reload programs from trusted sources.":"These controls describe how the supplied training observations are selected and grouped. The source, desired outcomes and model stay owned by Train. Apply or discard all dependent controls together before checking. The visual editor accepts up to 4,096 selected positions and exactly editable mixing seeds up to 9,007,199,254,740,991; the versioned recipe API retains its broader bounds.").font(.caption)
            }
        }
    }
}
