import KwipMLCore

/// Presentation only: labels come from actual connected source identities, never inferred shapes.
enum TensorOperationInputLabels {
    static func graph(_ graph:GraphDocument,nodeID:String,registry:ComponentRegistry)->[String:String] {
        var result:[String:String]=[:]
        for edge in graph.edges where edge.target==nodeID {
            guard let node=graph.nodes.first(where:{$0.id==edge.source}) else {continue}
            result[edge.targetPort]=(node.name.isEmpty ? registry.component(type:node.type)?.title ?? "Component":node.name)+" · "+edge.sourcePort
        }
        return result
    }
    static func workflow(_ document:WorkflowDocument,address:WorkflowNodeAddress,registry:ComponentRegistry)->[String:String] {
        if let child=try? WorkflowInlineContainer.parse(in:document,address:address),let definition=try? WorkflowInlineContainer.definition(in:document,root:child.root) {
            let graph=GraphDocument(nodes:definition.body.nodes,edges:definition.body.edges)
            var result=self.graph(graph,nodeID:child.childID,registry:registry)
            for port in definition.inputs where port.targets.contains(where:{$0.componentID==child.childID}) {
                for target in port.targets where target.componentID==child.childID {result[target.portID]="Container · "+port.id}
            }
            return result
        }
        var result:[String:String]=[:]
        for edge in document.edges where edge.to.address==address {
            guard let node=document.nodes.first(where:{$0.address==edge.from.address}) else {continue}
            let fallback:String
            switch node.payload {
            case .modelPrimitive(let type,_,_):fallback=registry.component(type:type)?.title ?? "Component"
            case .modelBlock(let reference,_):fallback=document.definitions.first {$0.reference==reference}?.title ?? "Container"
            default:fallback="Component"
            }
            result[edge.to.port]=(node.name.isEmpty ? fallback:node.name)+" · "+edge.from.port
        }
        return result
    }
}
