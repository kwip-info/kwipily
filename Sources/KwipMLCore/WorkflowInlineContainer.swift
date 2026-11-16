import Foundation
import CryptoKit

/// Authoring transactions over the actual immediate children of one graph Block.
/// Synthetic selections and layout never become executable source identities.
public enum WorkflowInlineContainer {
    public struct Selection:Equatable {public let root:WorkflowNodeAddress;public let childID:String}
    public indirect enum Command {
        case setInterface(inputs:[CompositeInput],outputs:[CompositeOutput])
        case configure(childID:String,config:[String:JSONValue])
        case rename(childID:String,name:String)
        case setLayerParameters(childID:String,settings:[String:JSONValue]?)
        case setActivation(childID:String,function:ActivationConfiguration.Function,configuration:[String:JSONValue]? = nil)
        case setPooling(childID:String,variant:PoolingConfiguration.Variant,configuration:[String:JSONValue]? = nil)
        case setConvolution(childID:String,variant:ConvolutionConfiguration.Variant,configuration:[String:JSONValue]? = nil)
        case setBilinear(childID:String,configuration:[String:JSONValue])
        case setEmbedding(childID:String,configuration:[String:JSONValue])
        case setAttention(childID:String,configuration:[String:JSONValue])
        case setEmbeddingBag(childID:String,configuration:[String:JSONValue])
        case setRecurrent(childID:String,method:RecurrentSequenceConfiguration.Method,configuration:[String:JSONValue]? = nil)
        case setTensorOperation(childID:String,type:String,configuration:[String:JSONValue])
        case setReshape(childID:String,variant:ReshapeAuthoringConfiguration.Variant,configuration:[String:JSONValue]? = nil)
        case setLinear(childID:String,variant:LinearConfiguration.Variant,configuration:[String:JSONValue]? = nil)
        case setNormalization(childID:String,variant:NormalizationConfiguration.Variant,configuration:[String:JSONValue]? = nil)
        case insertNode(GraphNode)
        case connect(from:CompositePortReference,to:CompositePortReference)
        case connectWorkflow(from:WorkflowEndpoint,to:WorkflowEndpoint)
        case disconnect(edgeID:String)
        case delete(childID:String)
        case moveIn(address:WorkflowNodeAddress,position:GraphOffset? = nil)
        case moveOut(childID:String,address:WorkflowNodeAddress,position:GraphOffset? = nil)
        case batch([Command])
    }
    public static func address(root:WorkflowNodeAddress,childID:String)->WorkflowNodeAddress {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let bytes=(try? encoder.encode([root.domain.rawValue,root.localID,childID])) ?? Data()
        return .init(domain:.model,localID:"inline-"+SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined())
    }
    /// Camera/selection callbacks resolve an immutable source index; they must not decode
    /// every expert and SHA every possible child address for each visible port or node.
    private final class SelectionIndex {
        let lock=NSLock()
        var source:WorkflowDocument?
        var members:[WorkflowNodeAddress:Selection]=[:]
        func parse(_ document:WorkflowDocument,_ candidate:WorkflowNodeAddress)->Selection? {
            lock.lock();defer{lock.unlock()}
            if source != document {
                var next:[WorkflowNodeAddress:Selection]=[:]
                for root in document.nodes {
                    guard let definition=try? WorkflowInlineContainer.definition(in:document,root:root.address) else{continue}
                    for child in definition.body.nodes {
                        next[WorkflowInlineContainer.address(root:root.address,childID:child.id)] = .init(root:root.address,childID:child.id)
                    }
                }
                members=next;source=document
            }
            return members[candidate]
        }
    }
    private static let selectionIndex=SelectionIndex()
    public static func parse(in document:WorkflowDocument,address candidate:WorkflowNodeAddress)throws->Selection? {
        guard candidate.domain == .model,!document.nodes.contains(where:{$0.address==candidate}) else{return nil}
        return selectionIndex.parse(document,candidate)
    }
    public static func definition(in document:WorkflowDocument,root:WorkflowNodeAddress)throws->CompositeDefinition {
        if WorkflowMixtureContainer.contains(document,root:root){return try WorkflowMixtureContainer.definition(in:document,root:root)}
        guard root.domain == .model,let node=document.nodes.first(where:{$0.address==root}),case .modelBlock(let pin,_)=node.payload,
              var definition=document.definitions.first(where:{$0.reference==pin}),isGraphContainer(definition) else {throw failure("Choose an actual editable graph container.")}
        if definition.composition == nil {
            // Historical Parallel features is displayed as its existing graph; this
            // transient policy is not persisted until an explicit semantic edit.
            definition.composition = .init(kind:.graph,children:(definition.body.nodes.map(\.id)+definition.body.instances.map(\.id)).map {.init(localID:$0)})
        }
        return definition
    }
    private static func isGraphContainer(_ definition:CompositeDefinition)->Bool {
        definition.composition?.kind == .graph || (definition.composition == nil && (definition.reference.id=="kwip.parallel-features" || definition.title=="Parallel features" || definition.reference==SeparableConv2dBlock.reference))
    }
    public static func node(in document:WorkflowDocument,root:WorkflowNodeAddress,childID:String,registry:ComponentRegistry)throws->WorkflowNode {
        let definition=try definition(in:document,root:root)
        let body:CompositeGraph
        if WorkflowMixtureContainer.contains(document,root:root){body=definition.body}
        else {
            guard let owner=document.nodes.first(where:{$0.address==root}),case .modelBlock(_,let overrides)=owner.payload else {throw failure("Container no longer exists.")}
            body=try CompositeConfigurationResolver.resolve(definition,overrides:overrides,definitions:document.definitions,registry:registry)
        }
        var authored:GraphNode?
        if WorkflowMixtureContainer.contains(document,root:root),let selected=try? WorkflowMixtureContainer.split(childID),
           let chain=(try? WorkflowMixtureContainer.chains(in:document,root:root))?.first(where:{$0.id==selected.chain}) {
            authored=chain.graph.nodes.first(where:{$0.id==selected.local})
        }
        guard let child=authored ?? body.nodes.first(where:{$0.id==childID}) else {throw failure("Choose an immediate primitive child; nested Blocks remain compact.")}
        let synthetic=address(root:root,childID:childID)
        guard !document.nodes.contains(where:{$0.address==synthetic}) else {throw failure("Inline selection collides with an authored node identity.")}
        return .init(address:synthetic,name:child.name,x:child.x,y:child.y,payload:.modelPrimitive(type:child.type,configurationVersion:2,config:child.config),parameterSettings:child.parameterSettings)
    }
    public static func apply(_ command:Command,to document:WorkflowDocument,root:WorkflowNodeAddress,registry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry?=nil,retirePreviousPrivateReference:CompositeDefinitionReference? = nil,idProvider:()->String={UUID().uuidString})throws->WorkflowDocument {
        if WorkflowMixtureContainer.contains(document,root:root){return try WorkflowMixtureContainer.apply(command,to:document,root:root,registry:registry,makeID:idProvider)}
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty,document.revision<UInt64.max else {throw failure("Resolve existing drafts or retained-definition/revision bounds before editing this container.")}
        var proposed=try definition(in:document,root:root),result=document
        if result.schemaVersion<11 {
            // Preserve legacy model source semantics when enabling explicit native
            // composition. Reading/projecting above never performs this migration.
            WorkflowModelInput.migrateOwnershipMetadata(&result)
            result.schemaVersion=max(11,result.schemaVersion)
        }
        let previousReference=proposed.reference
        if let retirePreviousPrivateReference {
            guard retirePreviousPrivateReference==previousReference,previousReference.id.hasPrefix("inline-private-") else {throw failure("Only the exact previous helper-private reference may be retired with current-session ownership proof.")}
        }
        let ownerIndex=result.nodes.firstIndex(where:{$0.address==root})!
        guard case .modelBlock(_,var overrides)=result.nodes[ownerIndex].payload else {throw failure("Container changed.")}
        proposed.body=try CompositeMigration.toCurrentBody(proposed.body,registry:registry)
        var reserved=Set(document.nodes.map {$0.address.localID}).union(document.edges.map(\.id)).union(document.definitions.map {$0.reference.id}).union(proposed.body.nodes.map(\.id)).union(proposed.body.instances.map(\.id)).union(proposed.body.edges.map(\.id)).union(proposed.inputs.map(\.id)).union(proposed.outputs.map(\.id))
        func fresh()throws->String {let id=idProvider();guard !id.isEmpty,id.unicodeScalars.count<=128,reserved.insert(id).inserted else {throw failure("An inline edit needs a fresh bounded identity.")};return id}
        // Boundary ports are visible product labels as well as stable saved IDs.
        func freshPort(_ direction:String)->String {
            var ordinal=1
            while !reserved.insert("\(direction)-\(ordinal)").inserted {ordinal+=1}
            return "\(direction)-\(ordinal)"
        }
        func childIndex(_ id:String)throws->Int {guard let index=proposed.body.nodes.firstIndex(where:{$0.id==id}) else {throw failure("The selected immediate primitive no longer exists.")};return index}
        func resolved()throws->CompositeGraph {try CompositeConfigurationResolver.resolve(proposed,overrides:overrides,definitions:result.definitions,registry:registry)}
        func boundFields(_ id:String)->Set<String> {Set(proposed.parameters.flatMap(\.targets).filter {$0.componentID==id}.map(\.field)+((proposed.integerProducts ?? []).flatMap(\.targets).filter {$0.componentID==id}.map(\.field)))}
        func pruneBindings(_ id:String)throws {
            proposed.integerProducts=proposed.integerProducts?.compactMap {binding in var next=binding;next.targets.removeAll {$0.componentID==id};return next.targets.isEmpty ? nil:next}
            if proposed.integerProducts?.isEmpty == true {proposed.integerProducts=nil}
            let required=Set((proposed.integerProducts ?? []).flatMap {[$0.lhs,$0.rhs]})
            var kept:[CompositeParameter]=[]
            for var parameter in proposed.parameters {
                parameter.targets.removeAll {$0.componentID==id}
                if parameter.targets.isEmpty {
                    guard !required.contains(parameter.field.key) else {throw failure("This child owns the last direct target of a parameter still used by a derived binding. Release that binding explicitly before moving or deleting it.")}
                    overrides.removeValue(forKey:parameter.field.key)
                } else {kept.append(parameter)}
            }
            proposed.parameters=kept
        }
        func externalPort(_ endpoint:WorkflowEndpoint,output:Bool)throws {
            guard endpoint.address.domain == .model,let node=result.nodes.first(where:{$0.address==endpoint.address}) else {throw failure("Connect an actual model-domain parent component.")}
            let names:[String]
            switch node.payload {
            case .modelPrimitive(let type,let version,let config):
                guard let component=registry.component(type:type) else {throw failure("The parent component is unavailable.")}
                let effective=try GraphMigration.effectiveConfiguration(of:.init(id:node.address.localID,type:type,config:config),schemaVersion:version,registry:registry)
                names=output ? component.configuredOutputs(effective):component.configuredInputs(effective)
            case .modelBlock(let pin,_):
                guard let def=result.definitions.first(where:{$0.reference==pin}) else {throw failure("The parent Block definition is missing.")}
                names=output ? def.outputs.map(\.id):def.inputs.map(\.id)
            default:throw failure("Only tensor model components can connect across a container boundary.")
            }
            guard names.contains(endpoint.port) else {throw failure("The parent endpoint is not a declared tensor port.")}
        }
        func policy() {proposed.composition = .init(kind:.graph,children:(proposed.body.nodes.map(\.id)+proposed.body.instances.map(\.id)).map {.init(localID:$0)})}
        func perform(_ command:Command,depth:Int=0)throws {
            guard depth<=8 else {throw failure("Inline command nesting exceeds its bounds.")}
            switch command {
            case .batch(let commands):guard commands.count<=128 else {throw failure("Too many inline commands.")};for command in commands {try perform(command,depth:depth+1)}
            case .setInterface(let inputs,let outputs):
                let connectedInputs=Set(result.edges.filter {$0.to.address==root}.map {$0.to.port})
                let connectedOutputs=Set(result.edges.filter {$0.from.address==root}.map {$0.from.port})
                guard connectedInputs.isSubset(of:Set(inputs.map(\.id))),connectedOutputs.isSubset(of:Set(outputs.map(\.id))) else {throw failure("Disconnect a container port's external wires before removing or renaming it.")}
                guard inputs.count<=128,outputs.count<=128,Set(inputs.map(\.id)).count==inputs.count,Set(outputs.map(\.id)).count==outputs.count else {throw failure("Use unique bounded names for container input and output ports.")}
                guard !outputs.isEmpty else {throw failure("Keep at least one named output in this container.")}
                proposed.inputs=inputs;proposed.outputs=outputs
            case .rename(let id,let name):guard name.count<=500 else {throw failure("Child name exceeds its bounds.")};proposed.body.nodes[try childIndex(id)].name=name
            case .configure(let id,let config):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                for field in boundFields(id) where config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its child value is bound.")}
                var next=config;for field in boundFields(id) {next[field]=proposed.body.nodes[index].config[field]}
                var node=proposed.body.nodes[index];node.config=next
                _ = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry)
                proposed.body.nodes[index]=node
            case .setAttention(let id,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let inputs=Set(proposed.body.edges.filter {$0.target==id}.map(\.targetPort)+proposed.inputs.flatMap(\.targets).filter {$0.componentID==id}.map(\.portID))
                let outputs=Set(proposed.body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var changed=try AttentionConfiguration.applying(configuration,to:effective,connectedInputPorts:inputs,connectedOutputPorts:outputs,registry:registry)
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its Attention value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setEmbeddingBag(let id,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let inputs=Set(proposed.body.edges.filter {$0.target==id}.map(\.targetPort)+proposed.inputs.flatMap(\.targets).filter {$0.componentID==id}.map(\.portID))
                var changed=try EmbeddingBagConfiguration.applying(configuration,to:effective,connectedInputPorts:inputs,registry:registry)
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its Embedding bag value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setEmbedding(let id,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                var changed=try EmbeddingConfiguration.applying(configuration,to:effective,registry:registry)
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its Embedding value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setBilinear(let id,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                var changed=try BilinearConfiguration.applying(configuration,to:effective,registry:registry)
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its Bilinear value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setRecurrent(let id,let method,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let inputs=Set(proposed.body.edges.filter {$0.target==id}.map(\.targetPort)+proposed.inputs.flatMap(\.targets).filter {$0.componentID==id}.map(\.portID))
                let outputs=Set(proposed.body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var changed=try RecurrentAuthoringConfiguration.switching(effective,configurationVersion:2,to:method,configuration:configuration,connectedInputPorts:inputs,connectedOutputPorts:outputs,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this recurrent's exposed container bindings before changing its method.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its recurrent value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setTensorOperation(let id,let type,let configuration):
                let index=try childIndex(id),body=try resolved(),effective=body.nodes.first {$0.id==id}!
                let inputs=Set(body.edges.filter {$0.target==id}.map(\.targetPort)+proposed.inputs.flatMap(\.targets).filter {$0.componentID==id}.map(\.portID))
                let outputs=Set(body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var planning=result
                if let position=planning.definitions.firstIndex(where:{$0.reference==proposed.reference}) {planning.definitions[position]=proposed}
                planning.nodes[ownerIndex].payload = .modelBlock(reference:proposed.reference,overrides:overrides)
                let graph=GraphDocument(nodes:body.nodes,edges:body.edges)
                let contracts=dataRegistry.map {TensorOperationAuthoring.incomingContracts(address:address(root:root,childID:id),document:planning,registry:registry,dataRegistry:$0)} ?? TensorOperationAuthoring.incomingContracts(nodeID:id,graph:graph,validation:GraphValidator.validate(graph,registry:registry))
                var changed=try TensorOperationAuthoring.switching(effective,configurationVersion:2,to:type,configuration:configuration,connectedInputs:inputs,connectedOutputs:outputs,inputs:contracts.mapValues(\.layout),registry:registry,inputContracts:contracts)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this tensor operation's exposed container bindings before changing method.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setReshape(let id,let variant,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                var changed=try ReshapeAuthoringConfiguration.switching(effective,configurationVersion:2,to:variant,configuration:configuration,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this transform's exposed container bindings before changing its variant.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its transform value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setLinear(let id,let variant,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let outputs=Set(proposed.body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var changed=try LinearConfiguration.switching(effective,configurationVersion:2,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this linear's exposed container bindings before changing its variant.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its linear value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setNormalization(let id,let variant,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let outputs=Set(proposed.body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var changed=try NormalizationConfiguration.switching(effective,configurationVersion:2,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this normalization's exposed container bindings before changing its variant.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its normalization value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setConvolution(let id,let variant,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let outputs=Set(proposed.body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var changed=try ConvolutionConfiguration.switching(effective,configurationVersion:2,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this convolution's exposed container bindings before changing its variant.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its convolution value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setPooling(let id,let variant,let configuration):
                let index=try childIndex(id),effective=try resolved().nodes.first {$0.id==id}!
                let outputs=Set(proposed.body.edges.filter {$0.source==id}.map(\.sourcePort)+proposed.outputs.filter {$0.source.componentID==id}.map {$0.source.portID})
                var changed=try PoolingConfiguration.switching(effective,configurationVersion:2,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this pooling component's exposed container bindings before changing its variant.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its pooling value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setActivation(let id,let function,let configuration):
                let index=try childIndex(id)
                let effective=try resolved().nodes.first {$0.id==id}!
                var changed=try ActivationConfiguration.switching(effective,configurationVersion:2,to:function,configuration:configuration,registry:registry)
                if changed.type != effective.type,!boundFields(id).isEmpty {throw failure("Release this activation's exposed container bindings before changing its function.")}
                if changed.type==effective.type {
                    for field in boundFields(id) where changed.config[field] != effective.config[field] {throw failure("Configure the exposed container parameter for \(field); its activation value is bound.")}
                    for field in boundFields(id) {changed.config[field]=proposed.body.nodes[index].config[field]}
                }
                proposed.body.nodes[index]=changed
            case .setLayerParameters(let id,let settings):
                if let settings {try LayerParameterSettings.validate(settings)}
                proposed.body.nodes[try childIndex(id)].parameterSettings=settings
                if settings != nil {WorkflowModelInput.migrateOwnershipMetadata(&result);result.schemaVersion=max(15,result.schemaVersion)}
            case .insertNode(let node):
                guard !proposed.body.nodes.contains(where:{$0.id==node.id}),!proposed.body.instances.contains(where:{$0.id==node.id}),!["input","index_input","bag_input","output"].contains(node.type) else {throw failure("Insert a unique ordinary primitive; model entry and output remain outside the container.")}
                _ = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry)
                proposed.body.nodes.append(node)
            case .connect(let from,let to):proposed.body.edges.append(.init(id:try fresh(),source:from.componentID,sourcePort:from.portID,target:to.componentID,targetPort:to.portID))
            case .connectWorkflow(let from,let to):
                let sourceChild=proposed.body.nodes.first {address(root:root,childID:$0.id)==from.address}?.id
                let targetChild=proposed.body.nodes.first {address(root:root,childID:$0.id)==to.address}?.id
                if let sourceChild,let targetChild {
                    proposed.body.edges.append(.init(id:try fresh(),source:sourceChild,sourcePort:from.port,target:targetChild,targetPort:to.port))
                } else if let targetChild {
                    if from.address==root {
                        guard let source=proposed.outputs.first(where:{$0.id==from.port})?.source else {throw failure("The container output no longer exists.")}
                        proposed.body.edges.append(.init(id:try fresh(),source:source.componentID,sourcePort:source.portID,target:targetChild,targetPort:to.port))
                    } else {
                        try externalPort(from,output:true)
                        let input=freshPort("input");proposed.inputs.append(.init(input,targets:[.init(targetChild,to.port)]))
                        result.edges.append(.init(id:try fresh(),from:from,to:.init(address:root,port:input),kind:.tensor))
                    }
                } else if let sourceChild {
                    guard to.address != root else {throw failure("Connect the actual target child rather than its container input boundary.")}
                    try externalPort(to,output:false)
                    guard !result.edges.contains(where:{$0.to==to}) else {throw failure("The parent input already has a connection.")}
                    let source=CompositePortReference(sourceChild,from.port),output:String
                    if let existing=proposed.outputs.first(where:{$0.source==source}) {output=existing.id}
                    else {output=freshPort("output");proposed.outputs.append(.init(output,source:source))}
                    result.edges.append(.init(id:try fresh(),from:.init(address:root,port:output),to:to,kind:.tensor))
                } else {throw failure("An inline connection must include an actual immediate child of this container.")}
            case .disconnect(let id):guard proposed.body.edges.contains(where:{$0.id==id}) else {throw failure("The internal connection no longer exists.")};proposed.body.edges.removeAll {$0.id==id}
            case .delete(let id):
                _ = try childIndex(id);try pruneBindings(id)
                let removedInputs=Set(proposed.inputs.filter {$0.targets.allSatisfy {$0.componentID==id}}.map(\.id)),removedOutputs=Set(proposed.outputs.filter {$0.source.componentID==id}.map(\.id))
                proposed.body.nodes.removeAll {$0.id==id};proposed.body.edges.removeAll {$0.source==id || $0.target==id}
                for index in proposed.inputs.indices {proposed.inputs[index].targets.removeAll {$0.componentID==id}}
                proposed.inputs.removeAll {$0.targets.isEmpty};proposed.outputs.removeAll {$0.source.componentID==id}
                result.edges.removeAll {($0.to.address==root && removedInputs.contains($0.to.port)) || ($0.from.address==root && removedOutputs.contains($0.from.port))}
            case .moveOut(let id,let address,let position):
                if let position {guard position.x.isFinite,position.y.isFinite else {throw failure("Container move coordinates must be finite.")}}
                guard address.domain == .model,!result.nodes.contains(where:{$0.address==address}) else {throw failure("Choose a fresh model-domain destination identity.")}
                let index=try childIndex(id),child=try resolved().nodes.first {$0.id==id}!
                try pruneBindings(id)
                let incident=proposed.body.edges.filter {$0.source==id || $0.target==id}
                for edge in incident where edge.target==id {
                    let source=CompositePortReference(edge.source,edge.sourcePort)
                    let output:String
                    if let existing=proposed.outputs.first(where:{$0.source==source}) {output=existing.id}
                    else {output=freshPort("output");proposed.outputs.append(.init(output,source:source))}
                    result.edges.append(.init(id:try fresh(),from:.init(address:root,port:output),to:.init(address:address,port:edge.targetPort),kind:.tensor))
                }
                let outgoing=Dictionary(grouping:incident.filter {$0.source==id},by:{$0.sourcePort})
                for port in outgoing.keys.sorted() {
                    let input=freshPort("input");proposed.inputs.append(.init(input,targets:outgoing[port]!.map {.init($0.target,$0.targetPort)}))
                    result.edges.append(.init(id:try fresh(),from:.init(address:address,port:port),to:.init(address:root,port:input),kind:.tensor))
                }
                let oldInputs=proposed.inputs
                for input in oldInputs {
                    let targets=input.targets.filter {$0.componentID==id};guard !targets.isEmpty else {continue}
                    let parents=result.edges.filter {$0.to.address==root && $0.to.port==input.id}
                    guard parents.count<=1,parents.allSatisfy({$0.kind == .tensor && $0.from.address.domain == .model}) else {throw failure("Repair the existing container input connection before transferring this child.")}
                    for parent in parents {try externalPort(parent.from,output:true)}
                    for target in targets {for parent in parents {result.edges.append(.init(id:try fresh(),from:parent.from,to:.init(address:address,port:target.portID),kind:parent.kind))}}
                }
                for output in proposed.outputs where output.source.componentID==id {
                    for index in result.edges.indices where result.edges[index].from.address==root && result.edges[index].from.port==output.id {result.edges[index].from = .init(address:address,port:output.source.portID)}
                }
                let emptyInputs=Set(proposed.inputs.filter {$0.targets.allSatisfy {$0.componentID==id}}.map(\.id))
                for index in proposed.inputs.indices {proposed.inputs[index].targets.removeAll {$0.componentID==id}}
                proposed.inputs.removeAll {$0.targets.isEmpty};proposed.outputs.removeAll {$0.source.componentID==id}
                result.edges.removeAll {$0.to.address==root && emptyInputs.contains($0.to.port)}
                proposed.body.edges.removeAll {$0.source==id || $0.target==id};proposed.body.nodes.remove(at:index)
                let owner=result.nodes.first {$0.address==root}!
                result.nodes.append(.init(address:address,name:child.name,x:position?.x ?? (owner.x+child.x),y:position?.y ?? (owner.y+child.y),payload:.modelPrimitive(type:child.type,configurationVersion:2,config:child.config),parameterSettings:child.parameterSettings))
            case .moveIn(let address,let position):
                if let position {guard position.x.isFinite,position.y.isFinite else {throw failure("Container move coordinates must be finite.")}}
                guard address.domain == .model,address != root,let index=result.nodes.firstIndex(where:{$0.address==address}),case .modelPrimitive(let type,let version,let config)=result.nodes[index].payload,
                      !["input","index_input","bag_input","output"].contains(type),!proposed.body.nodes.contains(where:{$0.id==address.localID}),!proposed.body.instances.contains(where:{$0.id==address.localID}) else {throw failure("Move a unique ordinary root primitive into this container.")}
                let original=result.nodes[index]
                let base=GraphNode(id:address.localID,type:type,name:original.name,x:position?.x ?? (original.x-(result.nodes.first {$0.address==root}?.x ?? 0)),y:position?.y ?? (original.y-(result.nodes.first {$0.address==root}?.y ?? 0)),config:config,parameterSettings:original.parameterSettings)
                var child=base;child.config=try GraphMigration.effectiveConfiguration(of:base,schemaVersion:version,registry:registry)
                proposed.body.nodes.append(child)
                let incident=result.edges.filter {$0.from.address==address || $0.to.address==address}
                guard incident.allSatisfy({$0.kind == .tensor && $0.from.address.domain == .model && $0.to.address.domain == .model}) else {throw failure("Only actual tensor wires can cross this model container boundary.")}
                for edge in incident {
                    if edge.from.address != address && edge.from.address != root {try externalPort(edge.from,output:true)}
                    if edge.to.address != address && edge.to.address != root {try externalPort(edge.to,output:false)}
                }
                result.edges.removeAll {$0.from.address==address || $0.to.address==address}
                for edge in incident where edge.to.address==address {
                    if edge.from.address==root {
                        guard let source=proposed.outputs.first(where:{$0.id==edge.from.port})?.source else {throw failure("The crossed container output no longer exists.")}
                        proposed.body.edges.append(.init(id:try fresh(),source:source.componentID,sourcePort:source.portID,target:child.id,targetPort:edge.to.port))
                    } else {
                        let input=freshPort("input");proposed.inputs.append(.init(input,targets:[.init(child.id,edge.to.port)]))
                        result.edges.append(.init(id:edge.id,from:edge.from,to:.init(address:root,port:input),kind:.tensor))
                    }
                }
                for edge in incident where edge.from.address==address {
                    if edge.to.address==root {
                        guard let input=proposed.inputs.first(where:{$0.id==edge.to.port}),!result.edges.contains(where:{$0.to==edge.to}) else {throw failure("The crossed container input no longer exists or has another parent connection.")}
                        for target in input.targets {proposed.body.edges.append(.init(id:try fresh(),source:child.id,sourcePort:edge.from.port,target:target.componentID,targetPort:target.portID))}
                        proposed.inputs.removeAll {$0.id==input.id}
                    } else {
                        let output:String
                        if let existing=proposed.outputs.first(where:{$0.source == .init(child.id,edge.from.port)}) {output=existing.id}
                        else {output=freshPort("output");proposed.outputs.append(.init(output,source:.init(child.id,edge.from.port)))}
                        result.edges.append(.init(id:edge.id,from:.init(address:root,port:output),to:edge.to,kind:.tensor))
                    }
                }
                result.nodes.remove(at:index)
            }
            policy()
        }
        try perform(command)
        guard !proposed.body.nodes.isEmpty || !proposed.body.instances.isEmpty else {
            throw failure("A container must keep at least one component. Remove the container itself if it is no longer needed.")
        }
        try validateAuthoring(proposed,overrides:overrides,definitions:result.definitions,registry:registry)
        let privateID="inline-private-"+(try fresh())
        guard privateID.unicodeScalars.count<=128 else {throw failure("The private container definition identity exceeds its bounds.")}
        proposed.appearance=proposed.resolvedAppearance
        proposed.reference = .init(privateID,version:proposed.reference.version)
        if retirePreviousPrivateReference != nil {
            // The trusted caller supplies current-session ownership/hash proof. Prefix
            // alone never grants retirement. Shared or imported references stay intact.
            let otherRoot=result.nodes.contains {node in
                guard node.address != root,case .modelBlock(let reference,_)=node.payload else {return false}
                return reference==previousReference
            }
            let nested=result.definitions.contains {$0.body.instances.contains {$0.definition==previousReference}}
            let draft=result.drafts.contains {$0.baseReference==previousReference || $0.proposed.body.instances.contains {$0.definition==previousReference}}
            if !otherRoot && !nested && !draft {result.definitions.removeAll {$0.reference==previousReference}}
        }
        guard result.definitions.count<64 else {throw failure("This edit would exceed64 retained definitions. No imported, original or shared definition was pruned.")}
        result.definitions.append(proposed)
        guard let owner=result.nodes.firstIndex(where:{$0.address==root}) else {throw failure("Container disappeared during its edit.")}
        result.nodes[owner].payload = .modelBlock(reference:proposed.reference,overrides:overrides)
        let outgoing=Dictionary(grouping:result.edges.filter {$0.kind == .tensor && $0.from.address.domain == .model && $0.to.address.domain == .model},by:{$0.from.address})
        var visiting=Set<WorkflowNodeAddress>(),done=Set<WorkflowNodeAddress>()
        func visit(_ node:WorkflowNodeAddress)throws {
            guard !visiting.contains(node) else {throw failure("This move or connection creates a container scope cycle. Keep intermediate dependencies together or use explicit branch boundaries.")}
            guard done.insert(node).inserted else {return};visiting.insert(node)
            for edge in outgoing[node] ?? [] {try visit(edge.to.address)}
            visiting.remove(node)
        }
        try visit(root)
        result.revision=document.revision+1
        try WorkflowCodec.validate(result)
        return result
    }
    private static func validateAuthoring(_ definition:CompositeDefinition,overrides:[String:JSONValue],definitions:[CompositeDefinition],registry:ComponentRegistry)throws {
        let body=try CompositeConfigurationResolver.resolve(definition,overrides:overrides,definitions:definitions,registry:registry)
        var ports:[String:CompositionPolicy.ChildPorts]=[:]
        for node in body.nodes {
            guard !["input","index_input","bag_input","output"].contains(node.type),let component=registry.component(type:node.type) else {throw failure("Container children must be known ordinary primitives.")}
            _ = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry)
            ports[node.id] = .init(inputs:component.configuredInputs(node.config),outputs:component.configuredOutputs(node.config))
        }
        for instance in body.instances {guard let child=definitions.first(where:{$0.reference==instance.definition}) else {throw failure("Nested Block definition is missing.")};ports[instance.id] = .init(inputs:child.inputs.map(\.id),outputs:child.outputs.map(\.id))}
        try definition.composition!.validate(childPorts:ports,edges:body.edges,inputs:definition.inputs,outputs:definition.outputs)
        var remaining=Set(ports.keys),done=Set<String>()
        while !remaining.isEmpty {let ready=remaining.filter {id in body.edges.filter {$0.target==id}.allSatisfy {done.contains($0.source)}};guard !ready.isEmpty else {throw failure("Internal connections cannot form a cycle.")};remaining.subtract(ready);done.formUnion(ready)}
    }
    private static func failure(_ message:String)->CompositeContractError {.init(.invalidBoundary,message)}
}
