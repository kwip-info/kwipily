import Foundation
import KwipMLCore

/// Human-readable choices over captured model identities. These strings never bind a parameter.
enum TrainLayerPresentation {
    static func parameterTitle(_ parameter:String)->String {
        if AttentionParameterPresentation.isAttention(parameter) {return AttentionParameterPresentation.title(parameter)}
        let pieces=parameter.split(separator:"_").map(String.init)
        if pieces.count>=2,["weight","bias"].contains(pieces[0]),["ih","hh","hr"].contains(pieces[1]) {
            let root=(pieces[1]=="ih" ? "Input":pieces[1]=="hh" ? "Recurrent":"Projection")+" "+(pieces[0]=="weight" ? "weights":"biases")
            if pieces.count==2 {return root}
            if [3,4].contains(pieces.count),pieces[2].hasPrefix("l"),let layer=Int(pieces[2].dropFirst()),(0..<8).contains(layer),pieces.count==3 || pieces[3]=="reverse" {
                return root+" · Layer \(layer+1) · "+(pieces.count==4 ? "Reverse":"Forward")
            }
        }
        switch parameter {
        case "weight":return "Weights"
        case "bias":return "Biases"
        default:return parameter.replacingOccurrences(of:"_",with:" ").capitalized
        }
    }
    static func names(for graph:GraphDocument,origins:[String:CompositeNodeOrigin],document:WorkflowDocument,registry:ComponentRegistry)->[String:String] {
        func label(_ name:String,fallback:String)->String {name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty ? fallback:name}
        func component(_ type:String)->String {registry.component(type:type)?.title ?? "Layer"}
        func numbered(_ entries:[(String,String)])->[String:String] {
            let counts=Dictionary(grouping:entries,by:{$0.1}).mapValues(\.count)
            var occurrences:[String:Int]=[:],result:[String:String]=[:]
            for (id,name) in entries {
                occurrences[name,default:0]+=1
                result[id]=counts[name,default:0]>1 ? name+" (\(occurrences[name]!))":name
            }
            return result
        }
        var definitions:[CompositeDefinitionReference:CompositeDefinition]=[:]
        for definition in document.definitions where definitions[definition.reference] == nil {definitions[definition.reference]=definition}
        let activeRootIDs=Set(graph.nodes.map { node in origins[node.id].map {$0.instancePath.first ?? $0.localNodeID} ?? node.id })
        let roots=document.nodes.filter {$0.address.domain == .model && activeRootIDs.contains($0.address.localID)}
        let rootNames=numbered(roots.compactMap {node in
            switch node.payload {
            case .modelPrimitive(let type,_,_):return (node.address.localID,label(node.name,fallback:component(type)))
            case .modelBlock(let reference,_):return (node.address.localID,label(node.name,fallback:definitions[reference]?.title ?? "Block"))
            default:return nil
            }
        })
        func bodyNames(_ body:CompositeGraph)->[String:String] {
            numbered(body.nodes.map {($0.id,label($0.name,fallback:component($0.type)))}+body.instances.map {($0.id,label($0.name,fallback:definitions[$0.definition]?.title ?? "Block"))})
        }
        var paths:[(String,String)]=[]
        for node in graph.nodes {
            guard let origin=origins[node.id] else {paths.append((node.id,label(node.name,fallback:component(node.type))));continue}
            if origin.instancePath.isEmpty {
                paths.append((node.id,rootNames[origin.localNodeID] ?? label(node.name,fallback:component(node.type))))
                continue
            }
            var parts:[String]=[],current:CompositeDefinition?
            for (index,instanceID) in origin.instancePath.enumerated() {
                if index==0 {
                    guard let instance=roots.first(where:{$0.address.localID==instanceID}),case .modelBlock(let reference,_)=instance.payload else {break}
                    parts.append(rootNames[instanceID] ?? label(instance.name,fallback:definitions[reference]?.title ?? "Block"));current=definitions[reference]
                } else {
                    guard let body=current?.body,let instance=body.instances.first(where:{$0.id==instanceID}) else {break}
                    parts.append(bodyNames(body)[instanceID] ?? label(instance.name,fallback:definitions[instance.definition]?.title ?? "Block"));current=definitions[instance.definition]
                }
            }
            let leaf=current.flatMap {bodyNames($0.body)[origin.localNodeID]} ?? label(node.name,fallback:component(node.type))
            paths.append((node.id,(parts+[leaf]).joined(separator:" / ")))
        }
        // Also disambiguate absent/incomplete provenance without exposing a flattened identifier.
        return numbered(paths)
    }
}
