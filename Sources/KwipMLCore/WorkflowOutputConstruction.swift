import Foundation

/// Downstream authoring does not change an already captured provider's computation.
/// Actual upstream wiring/configuration/definitions and owned input settings still must match.
public enum WorkflowOutputConstruction {
    public static func matches(_ current:WorkflowDocument,captured:WorkflowDocument,provider:WorkflowNodeAddress)->Bool {
        func slice(_ document:WorkflowDocument)->WorkflowDocument? {
            guard document.nodes.contains(where:{$0.address==provider}) else {return nil}
            let region=WorkflowModelInput.ancestors(of:provider,in:document)
            var value=document
            value.nodes=document.nodes.filter {region.contains($0.address)}
            value.edges=document.edges.filter {region.contains($0.to.address)}
            let owners=Set(value.nodes.filter(WorkflowModelBoundary.isDefinition).map {$0.address.localID})
            value.modelInputs=document.modelInputs.map {$0.filter {owners.contains($0.key)}}
            value.modelSourceVersions=document.modelSourceVersions.map {$0.filter {owners.contains($0.key)}}
            var references=Set(value.nodes.compactMap {node->CompositeDefinitionReference? in if case .modelBlock(let reference,_)=node.payload {return reference};return nil}),pending=Array(references)
            while let reference=pending.popLast() {
                guard let definition=document.definitions.first(where:{$0.reference==reference}) else {return nil}
                for child in definition.body.instances where references.insert(child.definition).inserted {pending.append(child.definition)}
            }
            value.definitions=document.definitions.filter {references.contains($0.reference)}
            value.guidedBuild=nil
            return value
        }
        guard current.projectID==captured.projectID,let now=slice(current),let before=slice(captured) else {return false}
        return now.matchesCapturedConstruction(before)
    }
}
