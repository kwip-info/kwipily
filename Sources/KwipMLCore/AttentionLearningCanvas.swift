import Foundation
public enum AttentionLearningCanvas {
    public static func registry(task: AttentionLearningTask) -> ComponentRegistry {
        var input = ComponentDefinition(id:"input",title:"Input",category:"Boundaries",outputs:["out"])
        input.fields = [.init(key:"fixture",title:"Supplied sequence",kind:.choice,defaultValue:.string(task.sourceRoles[0]),min:nil,max:nil,choices:task.sourceRoles.map { .init(value:$0,title:$0.capitalized) })]
        var output = ComponentDefinition(id:"output",title:"Model Output",category:"Boundaries",inputs:["in"])
        output.fields = [.init(key:"role",title:"Result role",kind:.choice,defaultValue:.string("result"),min:nil,max:nil,choices:task.outputRoles.map { .init(value:$0,title:$0.capitalized) })]
        return .init(components:[input,output,BundledAttentionComponents.definition(task.componentID),BundledAttentionComponents.definition("attention_mask")])
    }
    public static func inspect(_ graph: GraphDocument, task: AttentionLearningTask) -> ValidationResult {
        var result = ValidationResult()
        do { _ = try AttentionLearningSubmission.validatedGraph(graph,task:task,complete:false) } catch { result.issues.append(.init(id:"attention.structure",message:error.localizedDescription)); return result }
        do { _ = try AttentionLearningSubmission.validatedGraph(graph,task:task,complete:true) } catch { result.issues.append(.init(id:"attention.connections",message:error.localizedDescription)) }
        let inputs = graph.nodes.filter { $0.type == "input" }, masks = graph.nodes.filter { $0.type == "attention_mask" }, operators = graph.nodes.filter { $0.type == task.componentID }, outputs = graph.nodes.filter { $0.type == "output" }
        result.orderedNodeIDs = (inputs+masks+operators+outputs).map(\.id)
        for node in inputs { if let role = node.config["fixture"]?.stringValue { result.outputTensorContracts[.init(node.id,"out")] = try? .init(sample:task.sampleShapes[role]!,dtype:.float64) } }
        for node in masks+operators {
            guard let config = try? AttentionConfiguration(componentID:node.type,values:node.config) else { continue }
            let edges = config.configuredInputs.compactMap { port in graph.edges.first { $0.target == node.id && $0.targetPort == port } }
            let contracts = edges.compactMap { result.outputTensorContracts[.init($0.source,$0.sourcePort)] }
            guard contracts.count == config.configuredInputs.count, let resolved = try? config.resolved(layouts:contracts.map(\.layout)) else { continue }
            for (port,layout) in resolved.outputs { result.outputTensorContracts[.init(node.id,port)] = .init(layout:layout,dtype:node.type == "attention_mask" ? .int64 : .float64) }
        }
        for node in outputs { if let edge = graph.edges.first(where:{$0.target == node.id}), let contract = result.outputTensorContracts[.init(edge.source,edge.sourcePort)] { result.outputTensorContracts[.init(node.id,"out")] = contract } }
        return result
    }
}
