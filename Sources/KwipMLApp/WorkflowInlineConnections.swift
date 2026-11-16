import Foundation
import KwipMLCore

extension WorkflowDocument {
    var inlineInternalEdges:[WorkflowEdge] {
        nodes.flatMap {root -> [WorkflowEdge] in
            guard let definition=try? WorkflowInlineContainer.definition(in:self,root:root.address) else {return []}
            let members=Set(definition.body.nodes.map(\.id))
            return definition.body.edges.filter {members.contains($0.source) && members.contains($0.target)}.map {edge in
                .init(id:WorkflowInlineContainerProjection.internalEdgeID(root:root.address,edgeID:edge.id),
                    from:.init(address:WorkflowInlineContainer.address(root:root.address,childID:edge.source),port:edge.sourcePort),
                    to:.init(address:WorkflowInlineContainer.address(root:root.address,childID:edge.target),port:edge.targetPort),kind:.tensor)
            }
        }
    }
}
extension WorkflowCanvasAdapter {
    var inlineEditableEdges:[WorkflowEdge] {owner.document.edges+owner.document.inlineInternalEdges}
    /// Boundary rows disclose the actual source/destination through the reusable interface.
    /// Boundary changes use the real frame ports; these derived rows are read-only.
    var inlineInspectorEdges:[WorkflowEdge] {
        let document=owner.document
        var result=inlineEditableEdges
        for root in document.nodes {
            guard let definition=try? WorkflowInlineContainer.definition(in:document,root:root.address) else {continue}
            for input in definition.inputs {
                for edge in document.edges where edge.to.address==root.address && edge.to.port==input.id {
                    for target in input.targets {
                        result.append(.init(id:"inline-boundary-"+edge.id+"-"+target.componentID+"-"+target.portID,
                            from:edge.from,to:.init(address:WorkflowInlineContainer.address(root:root.address,childID:target.componentID),port:target.portID),kind:edge.kind))
                    }
                }
            }
            for output in definition.outputs {
                for edge in document.edges where edge.from.address==root.address && edge.from.port==output.id {
                    result.append(.init(id:"inline-boundary-"+edge.id,from:.init(address:WorkflowInlineContainer.address(root:root.address,childID:output.source.componentID),port:output.source.portID),to:edge.to,kind:edge.kind))
                }
            }
        }
        return result
    }
}
