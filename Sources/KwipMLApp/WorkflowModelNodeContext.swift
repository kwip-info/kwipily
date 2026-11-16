import Foundation
import Combine
import KwipMLCore

@MainActor final class WorkflowModelNodeContext:ObservableObject {
    let authority:WorkflowNodeEditingAuthority
    private let onEditPrivateBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)?
    private let onOpenBlock:((WorkflowNodeAddress,WorkflowEditorLease)->Void)?
    private var observation:AnyCancellable?
    init(owner:WorkflowAuthoringStore,address:WorkflowNodeAddress,onOpenBlock:((WorkflowNodeAddress,WorkflowEditorLease)->Void)?=nil, onEditPrivateBlock:((WorkflowNodeAddress,WorkflowEditorLease)->Void)?=nil)throws {
        guard address.domain == .model else {throw DataFlowContractError("Select a model node.")}
        authority=try .init(owner:owner,address:address);self.onOpenBlock=onOpenBlock;self.onEditPrivateBlock=onEditPrivateBlock
        observation=authority.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    /// Combined boundary controls share one editing lease; one section's commits must
    /// not retire another section on the same selected component.
    init(authority:WorkflowNodeEditingAuthority)throws {
        guard authority.address.domain == .model,authority.isCurrent else {throw DataFlowContractError("Select a current model node.")}
        self.authority=authority;onOpenBlock=nil;onEditPrivateBlock=nil
        observation=authority.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    var primitiveComponentID:String? {if case .modelPrimitive(let type,_,_)=authority.node?.payload {return type};return nil}
    var contextIdentity:String {authority.contextIdentity}
    var isCurrent:Bool {authority.isCurrent}
    var isBusy:Bool {authority.isBusy}
    var configuration:[String:JSONValue] {authority.configuration}
    var authoredName:String {authority.node?.name ?? ""}
    var statusMessage:String {authority.message ?? ""}
    var isLegacyBiasEnabled:Bool {if case .modelPrimitive("linear",1,_)=authority.node?.payload {return true};return false}
    var definition:CompositeDefinition? {
        guard case .modelBlock(let reference,_)=authority.node?.payload else {return nil}
        return authority.owner.document.definitions.first {$0.reference == reference}
    }
    var inlineBoundFields:Set<String> {
        guard let child=try? WorkflowInlineContainer.parse(in:authority.owner.document,address:authority.address),
              let definition=try? WorkflowInlineContainer.definition(in:authority.owner.document,root:child.root) else {return []}
        return Set((definition.parameters.flatMap(\.targets)+(definition.integerProducts ?? []).flatMap(\.targets)).filter {$0.componentID==child.childID}.map(\.field))
    }
    var isInlineContainer:Bool {
        guard let definition=try? WorkflowInlineContainer.definition(in:authority.owner.document,root:authority.address) else {return false}
        return definition.body.instances.isEmpty && !definition.body.nodes.isEmpty
    }
    @discardableResult func setContainerInterface(inputs:[CompositeInput],outputs:[CompositeOutput],expected:CompositeDefinition)->Bool {
        guard isInlineContainer,definition==expected else {authority.message="Container changed. Revert to load its current interface.";return false}
        return authority.apply(.inlineContainer(root:authority.address,command:.setInterface(inputs:inputs,outputs:outputs)))
    }
    var containerPortChoices:(inputs:[(CompositePortReference,String)],outputs:[(CompositePortReference,String)]) {
        guard let definition,let body=try? CompositeConfigurationResolver.resolve(definition,overrides:configuration,definitions:authority.owner.document.definitions,registry:authority.owner.modelRegistry) else {return ([],[])}
        func choices(_ output:Bool)->[(CompositePortReference,String)] {
            body.nodes.flatMap {node -> [(CompositePortReference,String)] in
                guard let component=authority.owner.modelRegistry.component(type:node.type) else {return []}
                return (output ? component.configuredOutputs(node.config):component.configuredInputs(node.config)).map {port in
                    (CompositePortReference(node.id,port),(node.name.isEmpty ? component.title:node.name)+" · "+port)
                }
            }
        }
        return (choices(false),choices(true))
    }
    @discardableResult func setAttention(configuration:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard primitiveComponentID.map(AttentionConfiguration.componentIDs.contains)==true,authority.accepts(expected) else {return false}
        return authority.apply(.setAttention(address:authority.address,configuration:configuration))
    }
    @discardableResult func setEmbeddingBag(configuration:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard primitiveComponentID=="embedding_bag",authority.accepts(expected) else {return false}
        return authority.apply(.setEmbeddingBag(address:authority.address,configuration:configuration))
    }
    @discardableResult func setEmbedding(configuration:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard primitiveComponentID=="embedding",authority.accepts(expected) else {return false}
        return authority.apply(.setEmbedding(address:authority.address,configuration:configuration))
    }
    @discardableResult func setBilinear(configuration:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard primitiveComponentID=="bilinear",authority.accepts(expected) else {return false}
        return authority.apply(.setBilinear(address:authority.address,configuration:configuration))
    }
    @discardableResult func setRecurrent(_ method:RecurrentSequenceConfiguration.Method,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setRecurrent(address:authority.address,method:method,configuration:configuration))
    }
    var tensorOperationConnections:(inputs:Set<String>,outputs:Set<String>) {
        if let child=try? WorkflowInlineContainer.parse(in:authority.owner.document,address:authority.address),
           let definition=try? WorkflowInlineContainer.definition(in:authority.owner.document,root:child.root) {
            return (Set(definition.body.edges.filter {$0.target==child.childID}.map(\.targetPort)+definition.inputs.flatMap(\.targets).filter {$0.componentID==child.childID}.map(\.portID)),
                    Set(definition.body.edges.filter {$0.source==child.childID}.map(\.sourcePort)+definition.outputs.filter {$0.source.componentID==child.childID}.map {$0.source.portID}))
        }
        return (Set(authority.owner.document.edges.filter {$0.to.address==authority.address}.map {$0.to.port}),Set(authority.owner.document.edges.filter {$0.from.address==authority.address}.map {$0.from.port}))
    }
    @discardableResult func setTensorOperation(_ type:String,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setTensorOperation(address:authority.address,type:type,configuration:configuration))
    }
    @discardableResult func setReshape(_ variant:ReshapeAuthoringConfiguration.Variant,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setReshape(address:authority.address,variant:variant,configuration:configuration))
    }
    @discardableResult func setLinear(_ variant:LinearConfiguration.Variant,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setLinear(address:authority.address,variant:variant,configuration:configuration))
    }
    @discardableResult func setNormalization(_ variant:NormalizationConfiguration.Variant,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setNormalization(address:authority.address,variant:variant,configuration:configuration))
    }
    @discardableResult func setConvolution(_ variant:ConvolutionConfiguration.Variant,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setConvolution(address:authority.address,variant:variant,configuration:configuration))
    }
    @discardableResult func setPooling(_ variant:PoolingConfiguration.Variant,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setPooling(address:authority.address,variant:variant,configuration:configuration))
    }
    @discardableResult func setActivation(_ function:ActivationConfiguration.Function,configuration:[String:JSONValue],expectedType:String,expected:[String:JSONValue])->Bool {
        guard primitiveComponentID==expectedType,authority.accepts(expected) else {return false}
        return authority.apply(.setActivation(address:authority.address,function:function,configuration:configuration))
    }
    var fields:[ComponentField] {
        switch authority.node?.payload {
        case .modelPrimitive(let type,let version,_):return (authority.owner.modelRegistry.component(type:type)?.fields(configuration:configuration) ?? []).filter {!(version == 1 && type == "linear" && $0.key == "bias")}
        case .modelBlock:return definition?.parameters.map(\.field) ?? []
        default:return []
        }
    }
    func value(for key:String)->JSONValue? {configuration[key] ?? (definition != nil ? fields.first {$0.key == key}?.defaultValue : nil)}
    @discardableResult func setField(_ key:String,value:JSONValue,expected:[String:JSONValue])->Bool {
        guard authority.accepts(expected),let field=fields.first(where:{$0.key == key}),field.accepts(value) else {authority.message="This field value was not applied.";return false}
        var next=expected;next[key]=value
        return authority.configure(next,expected:expected)
    }
    @discardableResult func removeOverride(_ key:String,expected:[String:JSONValue])->Bool {
        guard definition != nil,fields.contains(where:{$0.key == key}),authority.accepts(expected) else {return false}
        var next=expected;next.removeValue(forKey:key);return authority.configure(next,expected:expected)
    }
    var canOpenBlock:Bool {
        guard isCurrent,!authority.owner.isBusy,(definition != nil || primitiveComponentID=="mixture_experts"),onOpenBlock != nil else {return false}
        let drafts=authority.owner.document.drafts
        return authority.owner.flowRootURL != nil || drafts.isEmpty || drafts.first?.instancePath.first == authority.address.localID
    }
    @discardableResult func openBlock()->Bool {
        guard canOpenBlock,let onOpenBlock else {return false};onOpenBlock(authority.address,authority.token);return true
    }
    var hasPrivateDraft: Bool { authority.owner.document.drafts.first?.instancePath.first == authority.address.localID }
    var canEditPrivateCopy: Bool { canOpenBlock && isFileBackedBlock && onEditPrivateBlock != nil && (authority.owner.document.drafts.isEmpty || hasPrivateDraft) }
    @discardableResult func editPrivateCopy() -> Bool {
        guard canEditPrivateCopy, let onEditPrivateBlock else { return false }
        onEditPrivateBlock(authority.address, authority.token); return true
    }
    @Published private(set) var flowAdoption: WorkflowSourcePersistence.Adoption?
    var isFileBackedBlock: Bool { definition != nil && authority.owner.flowRootURL != nil }
    func reviewFlowUpdate() {
        guard !isBusy, let definition else { return }
        do {
            let head = try authority.owner.flowAdoptionHead(definition: definition.reference, lease: authority.token)
            if authority.owner.pinnedFlowDocument?.dependencies.first(where: { $0.definition == definition.reference })?.revision == head {
                flowAdoption = nil; authority.message = "This block uses the latest saved revision."; return
            }
            let proposal = try authority.owner.prepareFlowAdoption(definition: definition.reference, lease: authority.token)
            flowAdoption = proposal
            authority.message = nil
        } catch { flowAdoption = nil; authority.message = error.localizedDescription }
    }
    @discardableResult func adoptReviewedFlowUpdate() -> Bool {
        guard let proposal = flowAdoption else { return false }
        let accepted = authority.applyTransaction { authority.owner.adoptFlowRevision(proposal, lease: $0) }
        flowAdoption = nil
        return accepted
    }
    func dismissFlowUpdate() { flowAdoption = nil }
    @discardableResult func rename(_ name:String)->Bool {authority.rename(name)}
    func numericOrigin(key:String)->String {authority.numericOrigin(key:key)}
    func currentNumericOrigin(key:String)->String? {authority.currentNumericOrigin(key:key)}
    func setNumericPending(origin:String,key:String,pending:Bool) {authority.setNumericPending(origin:origin,key:key,pending:pending)}
    func retire() {authority.retire()}
}
