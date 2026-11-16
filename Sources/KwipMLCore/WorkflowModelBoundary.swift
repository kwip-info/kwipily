import Foundation

/// The executable output primitive is also the workflow's model-definition boundary.
/// Legacy pass-through carriers remain decodable for frozen History.
public enum WorkflowModelBoundary {
    public static func isDefinition(_ node:WorkflowNode)->Bool {
        switch node.payload {case .modelDefinition,.modelPrimitive("output",_,_):return true;default:return false}
    }
    public static func isVisibleDefinition(_ node:WorkflowNode,in document:WorkflowDocument)->Bool {
        guard isDefinition(node) else {return false}
        if case .modelDefinition=node.payload {return true}
        return !document.edges.contains {edge in
            guard edge.from.address==node.address,edge.to.port=="source" else {return false}
            return document.nodes.contains {candidate in guard candidate.address==edge.to.address else {return false};if case .modelDefinition=candidate.payload {return true};return false}
        }
    }
    public static func output(in document:WorkflowDocument,definition:WorkflowNodeAddress)throws->WorkflowNode {
        guard let node=document.nodes.first(where:{$0.address==definition}),isDefinition(node) else {throw DataFlowContractError("Select a Model definition boundary.")}
        if case .modelPrimitive("output",_,_)=node.payload {return node}
        let wires=document.edges.filter {$0.to.address==definition && $0.to.port=="source"}
        guard wires.count==1,let wire=wires.first,wire.kind == .modelDefinition,wire.from.port=="definition",
              let output=document.nodes.first(where:{$0.address==wire.from.address}),case .modelPrimitive("output",_,_)=output.payload else {throw DataFlowContractError("Repair the Model definition bridge: connect exactly one Model Output boundary.")}
        return output
    }
    /// Editable-source upgrade only; architecture node IDs/configuration and tensor wires are exact.
    public static func consolidate(_ source:WorkflowDocument)throws->WorkflowDocument {
        var next=source
        for carrier in source.nodes {
            guard case .modelDefinition=carrier.payload else {continue}
            let boundary=try output(in:source,definition:carrier.address)
            let incoming=source.edges.filter {$0.to.address==carrier.address}
            guard incoming.count==1,source.edges.filter({$0.from.address==carrier.address}).allSatisfy({$0.from.port=="model" && $0.kind == .modelDefinition}) else {throw DataFlowContractError("Repair unsupported Model definition bridge connections before consolidation.")}
            for (key,value) in source.modelInputs ?? [:] where key==carrier.address.localID {
                guard next.modelInputs?[boundary.address.localID] == nil || next.modelInputs?[boundary.address.localID]==value else {throw DataFlowContractError("Shared model boundaries have conflicting input ownership. Repair them before consolidation.")}
                next.modelInputs?[boundary.address.localID]=value;if key != boundary.address.localID {next.modelInputs?.removeValue(forKey:key)}
            }
            for (key,value) in source.namedModelInputs ?? [:] where key==carrier.address.localID {
                let existing=next.namedModelInputs?[boundary.address.localID] ?? [:]
                guard value.allSatisfy({existing[$0.key] == nil || existing[$0.key] == $0.value}) else {throw DataFlowContractError("Shared model boundaries have conflicting named inputs. Repair them before consolidation.")}
                var names=next.namedModelInputs ?? [:]
                names[boundary.address.localID]=existing.merging(value){_,new in new}
                if key != boundary.address.localID {names.removeValue(forKey:key)}
                next.namedModelInputs=names
            }
            for (key,value) in source.modelSourceVersions ?? [:] where key==carrier.address.localID {
                guard next.modelSourceVersions?[boundary.address.localID] == nil || next.modelSourceVersions?[boundary.address.localID]==value else {throw DataFlowContractError("Shared model boundaries have conflicting source versions. Repair them before consolidation.")}
                next.modelSourceVersions?[boundary.address.localID]=value;if key != boundary.address.localID {next.modelSourceVersions?.removeValue(forKey:key)}
            }
            next.nodes.removeAll {$0.address==carrier.address}
            next.edges.removeAll {$0.to.address==carrier.address}
            for index in next.edges.indices where next.edges[index].from.address==carrier.address {
                next.edges[index].from = .init(address:boundary.address,port:"definition")
            }
        }
        if let guided=source.guidedBuild,guided.courseID=="kwip.course.universal-classifier",guided.version==3 {
            guard (0...17).contains(guided.completedSteps) else {throw DataFlowContractError("Repair the saved guided-course stage before consolidating its boundary.")}
            next.guidedBuild = .init(courseID:guided.courseID,version:4,completedSteps:guided.completedSteps-(guided.completedSteps>=15 ? 1:0))
            let output=WorkflowNodeAddress(domain:.model,localID:"output")
            if next.nodes.contains(where:{$0.address==output && isDefinition($0)}),next.modelSourceVersions?[output.localID]==nil {
                // The original v3 course explicitly pins legacy model source2. Retain
                // that source identity when its separate ownership stage disappears.
                var versions=next.modelSourceVersions ?? [:];versions[output.localID]=2;next.modelSourceVersions=versions
            }
        }
        try WorkflowCodec.validate(next)
        return next
    }
}
