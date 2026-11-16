import Foundation
import KwipMLCore

extension WorkflowAuthoringStore {
    func authoringNode(at address:WorkflowNodeAddress)->WorkflowNode? {
        if let node=document.nodes.first(where:{$0.address==address}) {return node}
        guard let child=try? WorkflowInlineContainer.parse(in:document,address:address) else {return nil}
        return try? WorkflowInlineContainer.node(in:document,root:child.root,childID:child.childID,registry:modelRegistry)
    }
    func inlineSelectionIsValid(_ address:WorkflowNodeAddress)->Bool {authoringNode(at:address) != nil}
    func inlineCommand(_ command:WorkflowCommand,in proposed:WorkflowDocument?=nil)throws->WorkflowCommand {
        let document=proposed ?? self.document
        switch command {
        case .rename(let address,let name):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.rename(childID:child.childID,name:name))}
        case .configure(let address,let config):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.configure(childID:child.childID,config:config))}
        case .setAttention(let address,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setAttention(childID:child.childID,configuration:configuration))}
        case .setEmbeddingBag(let address,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setEmbeddingBag(childID:child.childID,configuration:configuration))}
        case .setEmbedding(let address,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setEmbedding(childID:child.childID,configuration:configuration))}
        case .setBilinear(let address,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setBilinear(childID:child.childID,configuration:configuration))}
        case .setRecurrent(let address,let method,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setRecurrent(childID:child.childID,method:method,configuration:configuration))}
        case .setTensorOperation(let address,let type,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setTensorOperation(childID:child.childID,type:type,configuration:configuration))}
        case .setReshape(let address,let variant,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setReshape(childID:child.childID,variant:variant,configuration:configuration))}
        case .setLinear(let address,let variant,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setLinear(childID:child.childID,variant:variant,configuration:configuration))}
        case .setNormalization(let address,let variant,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setNormalization(childID:child.childID,variant:variant,configuration:configuration))}
        case .setConvolution(let address,let variant,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setConvolution(childID:child.childID,variant:variant,configuration:configuration))}
        case .setPooling(let address,let variant,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setPooling(childID:child.childID,variant:variant,configuration:configuration))}
        case .setActivation(let address,let function,let configuration):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setActivation(childID:child.childID,function:function,configuration:configuration))}
        case .setLayerParameters(let address,let settings):
            if let child=try WorkflowInlineContainer.parse(in:document,address:address) {return .inlineContainer(root:child.root,command:.setLayerParameters(childID:child.childID,settings:settings))}
        case .connect(let from,let to):
            if let child=try WorkflowInlineContainer.parse(in:document,address:from.address) ?? WorkflowInlineContainer.parse(in:document,address:to.address) {
                return .inlineContainer(root:child.root,command:.connectWorkflow(from:from,to:to))
            }
        case .reconnect(let edgeID,let from,let to):
            if let wire=WorkflowInlineContainerProjection.internalEdge(in:document,displayID:edgeID) {
                return .inlineContainer(root:wire.root,command:.batch([.disconnect(edgeID:wire.edgeID),.connectWorkflow(from:from,to:to)]))
            }
        case .delete(let selected):
            let members=try selected.nodeAddresses.compactMap {try WorkflowInlineContainer.parse(in:document,address:$0)}
            let wires=selected.edgeIDs.compactMap {WorkflowInlineContainerProjection.internalEdge(in:document,displayID:$0)}
            let roots=Set(members.map(\.root)+wires.map(\.root))
            if let root=roots.first,roots.count==1,members.count==selected.nodeAddresses.count,wires.count==selected.edgeIDs.count {
                return .inlineContainer(root:root,command:.batch(members.map {.delete(childID:$0.childID)}+wires.map {.disconnect(edgeID:$0.edgeID)}))
            }
            if !members.isEmpty || !wires.isEmpty {throw DataFlowContractError("Select components within one container, or parent components, for this edit.")}
        default:break
        }
        return command
    }
}

extension WorkflowAuthoringStore {
    func inlineDefinitionHash(_ definition:CompositeDefinition)throws->String {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return WorkflowArchiveCodec.hash(try encoder.encode(definition))
    }
    /// Only a successful commit in this owner/session creates retirement authority.
    func authorizingInlineRetirement(_ command:WorkflowCommand,in proposed:WorkflowDocument?=nil,ownedDefinitions:[CompositeDefinitionReference:String]?=nil)throws->WorkflowCommand {
        guard case .inlineContainer(let root,let edit,_)=command else {return command}
        let current=try WorkflowInlineContainer.definition(in:proposed ?? document,root:root)
        let retire=try (ownedDefinitions ?? inlineOwnedDefinitions)[current.reference]==inlineDefinitionHash(current) ? current.reference:nil
        return .inlineContainer(root:root,command:edit,retirePreviousPrivateReference:retire)
    }
    func recordInlineOwnership(_ command:WorkflowCommand) {
        guard case .inlineContainer(let root,_,_)=command,let current=try? WorkflowInlineContainer.definition(in:document,root:root),
              let hash=try? inlineDefinitionHash(current) else {return}
        let retained=Set(document.definitions.map(\.reference))
        inlineOwnedDefinitions=inlineOwnedDefinitions.filter {retained.contains($0.key)}
        inlineOwnedDefinitions[current.reference]=hash
    }
}
