import SwiftUI
import Combine
import KwipMLCore

/// Ordered source bindings. Item and weight names move as one pair.
struct BagInputMappingDraft:Equatable {
    struct Row:Identifiable,Equatable {let id:UUID;var indexColumn:String;var weightColumn:String}
    var rows:[Row]
    var lengthColumn:String
    var target:String
    init(source:BagSourceConfiguration,mapping:BagInputMapping?=nil,target:String="") {
        let names=mapping?.indicesColumns ?? Array(repeating:"",count:source.maximumItemsPerBag)
        rows=names.enumerated().map {index,name in .init(id:UUID(),indexColumn:name,weightColumn:mapping?.weightColumns?[index] ?? "")}
        lengthColumn=mapping?.lengthColumn ?? "";self.target=target
    }
    mutating func move(_ id:UUID,by offset:Int) {
        guard abs(offset)==1,let index=rows.firstIndex(where:{$0.id==id}),rows.indices.contains(index+offset) else {return}
        rows.swapAt(index,index+offset)
    }
    func mapping(source:BagSourceConfiguration)throws->BagInputMapping {
        var value:[String:JSONValue]=["schemaVersion":1,"indicesColumns":.array(rows.map {.string($0.indexColumn)})]
        if source.usesPerSampleWeights {value["weightColumns"] = .array(rows.map {.string($0.weightColumn)})}
        if source.layout == "packed" {value["lengthColumn"] = .string(lengthColumn)}
        return try .init(value:.object(value))
    }
}

extension WorkflowControlNodeContext {
    /// Requirements are declarations from the actual connected models, never observed source schema.
    func bagMappingRequirements()throws->[WorkflowMappingRequirement] {
        let requirements=try mappingRequirements()
        guard !requirements.isEmpty else {throw DataFlowContractError("Connect Training data to Train, then connect its Model definition to inspect grouped input requirements.")}
        guard requirements.allSatisfy({$0.problem == nil && $0.bagRequirement != nil}),let first=requirements.first?.bagRequirement else {
            throw DataFlowContractError(requirements.compactMap(\.problem).first ?? "Every connected Train must use a grouped input model. Use separate mappings for grouped and ordinary inputs.")
        }
        guard requirements.allSatisfy({$0.bagRequirement?.configuration == first.configuration}) else {
            throw DataFlowContractError("Connected models use different grouped input controls. Give each source layout, capacity and weight arrangement its own Training data mapping.")
        }
        return requirements
    }
}

@MainActor final class BagTrainingMappingInspectorContext:ObservableObject {
    let context:WorkflowControlNodeContext
    let requirements:[WorkflowMappingRequirement]
    let source:BagSourceConfiguration
    @Published private(set) var draft:BagInputMappingDraft
    @Published private(set) var rejection:String?
    private var expected:[String:JSONValue]
    private var baseline:BagInputMappingDraft
    private var observation:AnyCancellable?
    private let pendingKey="bagInputMappingDraft"
    var validationMessage:String? {do {try validate(draft);return nil} catch {return error.localizedDescription}}
    var hasChanges:Bool {context.mappingConfiguration["bagInput"] == nil || draft != baseline}
    var canApply:Bool {hasChanges && !context.isBusy && validationMessage == nil}
    init(context:WorkflowControlNodeContext)throws {
        guard context.isMapping,context.isCurrent,!context.hasObjectiveMapping else {throw DataFlowContractError("Select current grouped Training data with one binary target; use the objective editor for explicit desired-outcome roles.")}
        self.context=context;expected=context.configuration
        requirements=try context.bagMappingRequirements()
        guard let requirement=requirements.first?.bagRequirement else {throw DataFlowContractError("Grouped source requirements are unavailable.")}
        source=requirement.configuration
        let stored=try context.mappingConfiguration["bagInput"].map {try BagInputMapping(value:$0)}
        let initialDraft=BagInputMappingDraft(source:source,mapping:stored,target:context.mappingConfiguration["targetColumn"]?.stringValue ?? "")
        draft=initialDraft;baseline=initialDraft
        observation=context.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    private func validate(_ draft:BagInputMappingDraft)throws {
        let mapping=try draft.mapping(source:source)
        for requirement in try context.bagMappingRequirements() {
            guard let bag=requirement.bagRequirement else {throw DataFlowContractError("Grouped source is unavailable.")}
            try mapping.validate(sourceConfiguration:bag.configuration,targetColumn:draft.target)
        }
    }
    @discardableResult func update(_ edit:(inout BagInputMappingDraft)->Void)->Bool {
        guard context.authority.accepts(expected) else {return false}
        var next=draft;edit(&next);draft=next;rejection=nil;reportPending();return true
    }
    @discardableResult func apply()->Bool {
        guard context.authority.accepts(expected) else {return false}
        do {
            try validate(draft)
            guard context.applyBagMapping(try draft.mapping(source:source),target:draft.target,expected:expected) else {return false}
            expected=context.configuration;baseline=draft;rejection=nil;reportPending();return true
        } catch {rejection=error.localizedDescription;reportPending();return false}
    }
    @discardableResult func discard()->Bool {
        guard context.authority.accepts(expected) else {return false}
        draft=baseline;rejection=nil;reportPending();return true
    }
    func releasePending() {context.setNumericPending(origin:context.numericOrigin(key:pendingKey),key:pendingKey,pending:false)}
    private func reportPending() {context.setNumericPending(origin:context.numericOrigin(key:pendingKey),key:pendingKey,pending:hasChanges)}
}

struct BagInputMappingFields:View {
    let source:BagSourceConfiguration
    @Binding var draft:BagInputMappingDraft
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text(source.layout == "packed" ? "Packed groups · up to \(source.maximumItemsPerBag) items per observation":"Fixed rows · \(source.maximumItemsPerBag) items per observation").font(.caption.bold())
            ForEach(Array(draft.rows.enumerated()),id:\.element.id) {index,row in
                VStack(alignment:.leading,spacing:5) {
                    HStack(spacing:5) {
                        Text("\(index+1)").font(.caption.monospacedDigit())
                        TextField("Category ID column",text:binding(row.id,weight:false)).flatField().accessibilityLabel("Category ID column \(index+1)")
                        Button {draft.move(row.id,by:-1)} label:{Image(systemName:"chevron.up")}.disabled(index==0).accessibilityLabel("Move category column \(index+1) earlier")
                        Button {draft.move(row.id,by:1)} label:{Image(systemName:"chevron.down")}.disabled(index==draft.rows.count-1).accessibilityLabel("Move category column \(index+1) later")
                        Button {draft.rows.removeAll {$0.id==row.id}} label:{Image(systemName:"minus")}.disabled(draft.rows.count<=1).accessibilityLabel("Remove category column \(index+1)")
                    }.buttonStyle(.borderless)
                    if source.usesPerSampleWeights {
                        TextField("Matching item-weight column",text:binding(row.id,weight:true)).flatField().accessibilityLabel("Item-weight column \(index+1)")
                    }
                }.padding(8).overlay(Rectangle().stroke(MLTheme.rule,lineWidth:1))
            }
            Button("Add item column") {draft.rows.append(.init(id:UUID(),indexColumn:"",weightColumn:""))}.buttonStyle(FlatButtonStyle()).disabled(draft.rows.count>=128)
            if source.layout == "packed" {
                TextField("Used-item count column",text:$draft.lengthColumn).flatField().accessibilityLabel("Used-item count column")
            }
        }
    }
    private func binding(_ id:UUID,weight:Bool)->Binding<String> {
        .init(get:{guard let row=draft.rows.first(where:{$0.id==id}) else {return ""};return weight ? row.weightColumn:row.indexColumn},set:{value in guard let index=draft.rows.firstIndex(where:{$0.id==id}) else {return};if weight {draft.rows[index].weightColumn=value} else {draft.rows[index].indexColumn=value}})
    }
}
struct BagTrainingMappingInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @State private var editor:BagTrainingMappingInspectorContext?
    @State private var problem:String?
    var body:some View {
        Group {
            if let editor {BagTrainingMappingEditor(editor:editor)}
            else {Text(problem ?? "Inspecting grouped source requirements…").font(.caption).foregroundStyle(MLTheme.rose)}
        }.onAppear {guard editor == nil else {return};do {editor=try .init(context:context)} catch {problem=error.localizedDescription}}
    }
}
private struct BagTrainingMappingEditor:View {
    @ObservedObject var editor:BagTrainingMappingInspectorContext
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            BagInputMappingFields(source:editor.source,draft:.init(get:{editor.draft},set:{next in _=editor.update {$0=next}}))
            VStack(alignment:.leading,spacing:6) {
                Text("Desired binary outcome").font(.caption.bold())
                TextField("Target column",text:.init(get:{editor.draft.target},set:{value in _=editor.update {$0.target=value}})).flatField()
            }.padding(10).overlay(Rectangle().stroke(MLTheme.rule,lineWidth:1))
            if let message=editor.validationMessage ?? editor.rejection {Label(message,systemImage:"exclamationmark.circle").font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {
                Button("Apply column mapping") {_=editor.apply()}.disabled(!editor.canApply)
                Button("Discard") {_=editor.discard()}.disabled(!editor.hasChanges || editor.context.isBusy)
                if editor.hasChanges {Text("Unapplied changes").font(.caption2).foregroundStyle(MLTheme.muted)}
            }.buttonStyle(FlatButtonStyle())
        }.disabled(editor.context.isBusy).onDisappear {editor.releasePending()}
    }
}
