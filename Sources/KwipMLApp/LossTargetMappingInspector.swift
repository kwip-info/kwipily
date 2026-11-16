import SwiftUI
import Combine
import KwipMLCore

extension WorkflowControlNodeContext {
    var hasObjectiveMapping:Bool {
        guard isMapping,isCurrent else {return false}
        if mappingConfiguration["lossTargetMapping"] != nil {return true}
        let trains=isTrain ? Set([authority.address]):Set(authority.owner.document.edges.filter {$0.from.address == authority.address && $0.to.port == "data"}.map(\.to.address))
        return authority.owner.document.nodes.contains {node in guard trains.contains(node.address),case .train(let config)=node.payload else {return false};return config["objective"] != nil}
            || authority.owner.document.edges.contains {trains.contains($0.to.address) && $0.to.port == "objective"}
    }
}

@MainActor final class LossTargetMappingInspectorContext:ObservableObject {
    struct Requirement {
        let label:String
        let input:TypedTensorContract?
        let prediction:TypedTensorContract?
        let problem:String?
        var bag:BagWorkflowInputRequirement? = nil
    }
    let context:WorkflowControlNodeContext
    let componentID:String
    let configuration:LossConfiguration
    let requirements:[Requirement]
    @Published private(set) var draft:LossTargetMappingDraft
    @Published private(set) var rejection:String?
    private var expected:[String:JSONValue]
    private var baseline:LossTargetMapping
    private var observation:AnyCancellable?
    private let pendingKey="lossTargetMappingDraft"
    @Published private(set) var bagDraft:BagInputMappingDraft?
    private var baselineBag:BagInputMappingDraft?
    var bagSource:BagSourceConfiguration? {requirements.first?.bag?.configuration}
    private var candidateMapping:LossTargetMapping {
        var mapping=draft.mapping
        if let bagDraft {mapping.inputColumns=bagDraft.rows.map(\.indexColumn)}
        return mapping
    }
    @discardableResult func updateBag(_ edit:(inout BagInputMappingDraft)->Void)->Bool {
        guard context.authority.accepts(expected),var next=bagDraft else {return false}
        edit(&next);bagDraft=next;rejection=nil;reportPending();return true
    }
    var roleChangeMessage:String? {
        let active=Self.editableMapping(configuration:configuration,stored:baseline,prediction:predictionContract?.layout.sample.dimensions ?? [1])
        guard active.roles.map(\.role) != baseline.roles.map(\.role) || active.roles.map(\.dtype) != baseline.roles.map(\.dtype) else {return nil}
        return "Train’s objective now requires different outcome roles or number types. Matching column names and directions are retained for repair. Apply explicitly saves these active roles; inactive columns remain in the saved source until then. Discard leaves the saved source unchanged."
    }
    var inputContract:TypedTensorContract? {requirements.first?.input}
    var predictionContract:TypedTensorContract? {requirements.first?.prediction}
    var validationMessage:String? {
        do {try validate(candidateMapping);return nil} catch {return error.localizedDescription}
    }
    var hasChanges:Bool {context.mappingConfiguration["lossTargetMapping"] == nil || candidateMapping != baseline || bagDraft != baselineBag}
    var canApply:Bool {hasChanges && !context.isBusy && validationMessage == nil}
    init(context:WorkflowControlNodeContext)throws {
        guard context.isMapping,context.isCurrent else {throw DataFlowContractError("Select a current Training data component.")}
        self.context=context;expected=context.configuration
        let owner=context.authority.owner,document=owner.document,address=context.authority.address
        let outgoing=document.edges.filter {$0.from.address == address && $0.to.port == "data"}
        let consumers=context.isTrain ? document.nodes.filter {$0.address==address}:document.nodes.filter {node in outgoing.contains {$0.to.address == node.address} && {if case .train=node.payload {return true};return false}()}
        guard !consumers.isEmpty else {throw DataFlowContractError("Connect Training data’s data output to Train’s data input before choosing objective roles.")}
        var configs:[LossConfiguration]=[],contracts:[Requirement]=[]
        for train in consumers {
            let data=document.edges.filter {$0.to.address == train.address && $0.to.port == "data"}
            guard context.isTrain || (data.count == 1 && data.first?.from.address == address && data.first?.from.port == "data" && data.first?.kind == .trainingData) else {throw DataFlowContractError("Use one training-data connection from this mapping’s data output to Train’s data input.")}
            let objective=document.edges.filter {$0.to.address == train.address && $0.to.port == "objective"}
            if case .train(let trainConfig)=train.payload,let raw=trainConfig["objective"] {
                guard objective.isEmpty else {throw DataFlowContractError("Choose one objective authority per Train.")}
                configs.append(try LossConfiguration.decodeRecipe(raw))
            } else {
                guard objective.count == 1,let link=objective.first,link.kind == .objectiveRecipe,link.from.port == "out",
                      let node=document.nodes.first(where:{$0.address == link.from.address}),case .lossRecipe(let id,1,let values)=node.payload else {throw DataFlowContractError("Choose an Objective in every connected Train. Mixed implicit and explicit objectives need separate mappings.")}
                configs.append(try LossConfiguration(componentID:id,values:values))
            }
            do {
                let models=document.edges.filter {$0.to.address == train.address && $0.to.port == "model"}
                guard models.count == 1,let model=models.first,model.kind == .modelDefinition,["model","definition"].contains(model.from.port) else {throw DataFlowContractError("Connect one Model definition to Train to inspect the required array directions.")}
                let snapshot=try WorkflowCompiler.modelSnapshot(document,definition:model.from.address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,context:.explicitObjective)
                let bag=snapshot.graph.nodes.contains {$0.type == "bag_input"} ? try WorkflowMappingRequirements.bagInputRequirement(snapshot.graph,registry:owner.modelRegistry,context:.explicitObjective):nil
                contracts.append(.init(label:train.name.isEmpty ? "Train":train.name,
                    input:bag == nil ? try WorkflowMappingRequirements.inputContract(snapshot.graph,registry:owner.modelRegistry,context:.explicitObjective):nil,
                    prediction:try WorkflowMappingRequirements.outputContract(snapshot.graph,registry:owner.modelRegistry),problem:nil,bag:bag))
            } catch {contracts.append(.init(label:train.name.isEmpty ? "Train":train.name,input:nil,prediction:nil,problem:error.localizedDescription))}
        }
        guard let selected=configs.first,configs.allSatisfy({$0 == selected}) else {throw DataFlowContractError("Connected Trains use different objectives or controls. Give each objective its own Training data mapping.")}
        configuration=selected;componentID=selected.componentID;requirements=contracts
        var initialBagDraft:BagInputMappingDraft?
        if context.requiresBagMapping {
            let resolved=try context.bagMappingRequirements()
            guard let source=resolved.first?.bagRequirement?.configuration else {throw DataFlowContractError("Grouped source requirements are unavailable.")}
            let stored=try context.mappingConfiguration["bagInput"].map {try BagInputMapping(value:$0)}
            initialBagDraft = .init(source:source,mapping:stored)
        }
        let initialBaseline:LossTargetMapping
        if context.mappingConfiguration["lossTargetMapping"] != nil {
            initialBaseline=try WorkflowControlContracts.lossTargetMapping(context.mappingConfiguration.filter {$0.key != "bagInput"},address:address)
            guard initialBaseline.componentID == selected.componentID else {throw DataFlowContractError("The stored roles belong to another objective. Select that objective on Train or start a separate Training data mapping.")}
        } else {
            let inputs: [String]
            if let initialBagDraft {inputs=initialBagDraft.rows.map(\.indexColumn)}
            else if case .array(let values)=context.mappingConfiguration["featureColumns"] {inputs=values.compactMap(\.stringValue)}
            else {inputs=Array(repeating:"",count:min(TrainingColumnMapping.maximumColumns,contracts.first?.input?.layout.sample.elementCount ?? 1))}
            initialBaseline=Self.initialMapping(configuration:selected,inputs:inputs,prediction:contracts.first?.prediction?.layout.sample.dimensions ?? [1],legacyTarget:context.mappingConfiguration["targetColumn"]?.stringValue)
        }
        baseline=initialBaseline
        draft = .init(mapping:Self.editableMapping(configuration:selected,stored:initialBaseline,prediction:contracts.first?.prediction?.layout.sample.dimensions ?? [1]))
        bagDraft=initialBagDraft;baselineBag=initialBagDraft
        observation=context.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    private static func initialMapping(configuration:LossConfiguration,inputs:[String],prediction:[Int],legacyTarget:String?)->LossTargetMapping {
        let id=configuration.componentID,c=configuration.values
        if id == "loss_triplet_margin" {
            let width=max(1,inputs.count/3)
            let groups=(0..<3).map {group in (0..<width).map {index in let position=group*width+index;return inputs.indices.contains(position) ? inputs[position]:""}}
            return .init(componentID:id,inputColumns:groups.flatMap {$0},roles:[],tripletSourceGroups:groups)
        }
        func role(_ name:String,_ shape:[Int],_ dtype:ContractDType)->LossTargetRoleMapping {
            let count=min(TrainingColumnMapping.maximumColumns,shape.reduce(1,*))
            return .init(role:name,columns:count == 1 && name == "target" ? [legacyTarget ?? ""]:Array(repeating:"",count:count),sampleShape:shape,dtype:dtype)
        }
        var shape=prediction,dtype=ContractDType.float32
        if id == "loss_ctc" {return .init(componentID:id,inputColumns:inputs,roles:[role("target",[1],.int64),role("input_lengths",[],.int64),role("target_lengths",[],.int64)])}
        if id == "loss_nll" || (id == "loss_cross_entropy" && c["target_mode"] != .string("probabilities")) {
            let rank=prediction.count+1,axis=c["class_axis"]?.integerValue ?? 1,full=(axis%rank+rank)%rank
            if full>0 && shape.indices.contains(full-1) {shape.remove(at:full-1)}
            dtype = .int64
        }
        var roles=[role("target",shape,dtype)]
        if id == "loss_bce_with_logits" {for name in ["weight","pos_weight"] where c[name+"_mode"] == .string("mapped") {roles.append(role(name,prediction,.float32))}}
        return .init(componentID:id,inputColumns:inputs,roles:roles)
    }
    private static func editableMapping(configuration:LossConfiguration,stored:LossTargetMapping,prediction:[Int])->LossTargetMapping {
        guard configuration.componentID != "loss_triplet_margin" else {return stored}
        let required=initialMapping(configuration:configuration,inputs:stored.inputColumns,prediction:prediction,legacyTarget:nil)
        var active=stored
        active.roles=required.roles.map {role in
            guard var prior=stored.roles.first(where:{$0.role == role.role}) else {return role}
            prior.dtype=role.dtype;return prior
        }
        return active
    }
    func validate(_ mapping:LossTargetMapping,bagMapping:BagInputMapping?=nil)throws {
        let packet:BagInputMapping?
        if let bagSource {
            packet=try bagMapping ?? bagDraft?.mapping(source:bagSource)
            guard let packet else {throw DataFlowContractError("Preserve the grouped source column bindings.")}
            for requirement in try context.bagMappingRequirements() {
                guard let bag=requirement.bagRequirement else {throw DataFlowContractError("Grouped source is unavailable.")}
                try packet.validate(sourceConfiguration:bag.configuration,objectiveMapping:mapping)
            }
        } else {
            guard bagMapping == nil else {throw DataFlowContractError("This objective uses ordinary model inputs.")}
            packet=nil
        }
        let recipe=try LossRecipe(configuration:configuration,source:.init(domain:.control,localID:"authoring-requirement"),targetMapping:mapping)
        for requirement in requirements {
            guard let prediction=requirement.prediction else {throw DataFlowContractError(requirement.problem ?? "Connect a valid Model definition to inspect the required array directions.")}
            if requirement.bag == nil {
                guard let input=requirement.input,mapping.inputColumns.count == input.layout.sample.elementCount else {throw DataFlowContractError("Ordered input columns must fill the ordinary model’s values per observation.")}
            } else if packet == nil {throw DataFlowContractError("Grouped models require category ID bindings, not a dense input shape.")}
            try WorkflowMappingRequirements.validateObjective(prediction:prediction,recipe:recipe)
        }
    }
    @discardableResult func update(_ edit:(inout LossTargetMappingDraft)->Void)->Bool {
        guard context.authority.accepts(expected) else {return false}
        var next=draft;edit(&next);draft=next;rejection=nil;reportPending();return true
    }
    @discardableResult func apply()->Bool {
        guard context.authority.accepts(expected) else {return false}
        do {
            let candidate=candidateMapping
            let bag=try bagSource.flatMap {try bagDraft?.mapping(source:$0)}
            try validate(candidate,bagMapping:bag)
            guard context.applyObjectiveMapping(candidate,bagMapping:bag,expected:expected) else {return false}
            expected=context.configuration;baseline=candidate;baselineBag=bagDraft;draft.inputs=TrainingMappingDraft(features:candidate.inputColumns,target:"");rejection=nil;reportPending();return true
        } catch {rejection=error.localizedDescription;reportPending();return false}
    }
    @discardableResult func discard()->Bool {
        guard context.authority.accepts(expected) else {return false}
        draft = .init(mapping:Self.editableMapping(configuration:configuration,stored:baseline,prediction:predictionContract?.layout.sample.dimensions ?? [1]));bagDraft=baselineBag;rejection=nil;reportPending();return true
    }
    func releasePending() {context.setNumericPending(origin:context.numericOrigin(key:pendingKey),key:pendingKey,pending:false)}
    private func reportPending() {context.setNumericPending(origin:context.numericOrigin(key:pendingKey),key:pendingKey,pending:hasChanges)}
}

struct LossTargetMappingInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @State private var editor:LossTargetMappingInspectorContext?
    @State private var problem:String?
    var body:some View {
        Group {
            if let editor {LossTargetMappingEditor(editor:editor)}
            else {Text(problem ?? "Inspecting connected objective requirements…").font(.caption).foregroundStyle(MLTheme.rose)}
        }.onAppear {guard editor == nil else {return};do {editor=try .init(context:context)} catch {problem=error.localizedDescription}}
    }
}

private struct LossTargetMappingEditor:View {
    @ObservedObject var editor:LossTargetMappingInspectorContext
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            ForEach(Array(editor.requirements.enumerated()),id:\.offset) {_,requirement in
                VStack(alignment:.leading,spacing:4) {
                    Text(requirement.label).font(.caption.bold())
                    if let input=requirement.input {Text("Required input: \(input.layout.sample.dimensions.description) · \(input.dtype.rawValue) · \(input.layout.sample.elementCount) ordered columns").font(.caption)}
                    if let prediction=requirement.prediction {Text("Prediction array: \(prediction.layout.sample.dimensions.description) · \(prediction.dtype.rawValue)").font(.caption)}
                    if let problem=requirement.problem {Text(problem).font(.caption).foregroundStyle(MLTheme.rose)}
                }
            }
            if let source=editor.bagSource,let draft=editor.bagDraft {
                BagInputMappingFields(source:source,draft:Binding(get:{editor.bagDraft ?? draft},set:{next in _=editor.updateBag {$0=next}}))
            } else if editor.draft.groups.isEmpty {
                columnGroup(title:"Model input columns",binding:Binding(get:{editor.draft.inputs},set:{next in _=editor.update {$0.inputs=next}}))
            } else {
                ForEach(0..<editor.draft.groups.count,id:\.self) {index in
                    columnGroup(title:["Anchor columns","Related-example columns","Unrelated-example columns"][index],binding:Binding(get:{editor.draft.groups[index]},set:{next in _=editor.update {$0.groups[index]=next}}))
                }
            }
            ForEach(Array(editor.draft.roles.enumerated()),id:\.element.id) {index,role in
                VStack(alignment:.leading,spacing:8) {
                    columnGroup(title:roleTitle(role.name),binding:Binding(get:{editor.draft.roles[index].columns},set:{next in _=editor.update {$0.roles[index].columns=next}}))
                    Text(role.dtype == .int64 ? "Whole numbers (int64)":"Decimal values (float32)").font(.caption).foregroundStyle(MLTheme.muted)
                    if role.sampleShape.isEmpty {Text("Scalar: one value per observation.").font(.caption)}
                    else {ForEach(Array(role.sampleShape.enumerated()),id:\.offset) {dimension,size in
                        Stepper("Direction \(dimension+1): \(size)",value:Binding(get:{editor.draft.roles[index].sampleShape[dimension]},set:{value in _=editor.update {$0.roles[index].sampleShape[dimension]=value}}),in:1...TrainingColumnMapping.maximumColumns).font(.caption)
                    }}
                    if editor.componentID != "loss_ctc" || role.name == "target" {
                        HStack {
                            Button("Add direction") {_=editor.update {$0.roles[index].sampleShape.append(1)}}.disabled(role.sampleShape.count>=7 || editor.componentID == "loss_ctc")
                            Button("Remove direction") {_=editor.update {if !$0.roles[index].sampleShape.isEmpty {$0.roles[index].sampleShape.removeLast()}}}.disabled(role.sampleShape.isEmpty || (editor.componentID == "loss_ctc" && role.sampleShape.count==1))
                        }.buttonStyle(FlatButtonStyle())
                    }
                }
                Divider()
            }
            if let change=editor.roleChangeMessage {Text(change).font(.caption).foregroundStyle(MLTheme.muted)}
            if let problem=editor.validationMessage ?? editor.rejection {Label(problem,systemImage:"exclamationmark.circle").font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {
                Button("Apply column mapping") {_=editor.apply()}.disabled(!editor.canApply)
                Button("Discard") {_=editor.discard()}.disabled(!editor.hasChanges || editor.context.isBusy)
                if editor.hasChanges {Text("Unapplied changes").font(.caption2).foregroundStyle(MLTheme.muted)}
            }.buttonStyle(FlatButtonStyle())
        }.disabled(editor.context.isBusy).onDisappear {editor.releasePending()}
    }
    private func roleTitle(_ name:String)->String {
        switch name {case "target":return "Desired-outcome columns";case "input_lengths":return "Source-length column";case "target_lengths":return "Desired-sequence-length column";case "weight":return "Importance columns";case "pos_weight":return "Positive-outcome importance columns";default:return name}
    }
    private func columnGroup(title:String,binding:Binding<TrainingMappingDraft>)->some View {
        VStack(alignment:.leading,spacing:6) {
            Text(title).font(.caption.bold())
            ForEach(Array(binding.wrappedValue.rows.enumerated()),id:\.element.id) {index,row in
                HStack(spacing:4) {
                    Text("\(index+1)").font(.caption.monospacedDigit())
                    TextField("Column name",text:Binding(get:{binding.wrappedValue.rows.first {$0.id == row.id}?.name ?? ""},set:{value in var next=binding.wrappedValue;next.setName(value,for:row.id);binding.wrappedValue=next})).flatField().accessibilityLabel(title+" column \(index+1)")
                    Button {var next=binding.wrappedValue;next.move(row.id,by:-1);binding.wrappedValue=next} label:{Image(systemName:"chevron.up")}.disabled(index==0).accessibilityLabel("Move \(title) column \(index+1) earlier")
                    Button {var next=binding.wrappedValue;next.move(row.id,by:1);binding.wrappedValue=next} label:{Image(systemName:"chevron.down")}.disabled(index==binding.wrappedValue.rows.count-1).accessibilityLabel("Move \(title) column \(index+1) later")
                    Button {var next=binding.wrappedValue;next.remove(row.id);binding.wrappedValue=next} label:{Image(systemName:"minus")}.disabled(binding.wrappedValue.rows.count<=1).accessibilityLabel("Remove \(title) column \(index+1)")
                }.buttonStyle(.borderless)
            }
            Button("Add column") {var next=binding.wrappedValue;next.add();binding.wrappedValue=next}.buttonStyle(FlatButtonStyle()).disabled(binding.wrappedValue.rows.count>=TrainingColumnMapping.maximumColumns)
        }.padding(10).overlay(Rectangle().stroke(MLTheme.rule,lineWidth:1))
    }
}
