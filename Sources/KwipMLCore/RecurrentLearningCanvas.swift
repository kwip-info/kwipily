import Foundation

public enum RecurrentLearningCanvas {
    public static func registry(task:RecurrentLearningTask)->ComponentRegistry {
        var output=ComponentDefinition(id:"output",title:"Model Output",category:"Boundaries",inputs:["in"])
        output.fields=[.init(key:"role",title:"Result role",kind:.choice,defaultValue:.string("result"),min:nil,max:nil,choices:task.outputRoles.map {.init(value:$0,title:$0.capitalized)})]
        var input=ComponentDefinition(id:"input",title:"Input",category:"Boundaries",outputs:["out"])
        input.fields=[.init(key:"fixture",title:"Supplied source",kind:.choice,defaultValue:.string("input"),min:nil,max:nil,choices:task.sourceRoles.map {.init(value:$0,title:$0.capitalized)})]
        return .init(components:[input,BundledRecurrentComponents.definition(task.componentID),output])
    }
    public static func inspect(_ graph:GraphDocument,task:RecurrentLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {_ = try RecurrentLearningSubmission.validatedGraph(graph,task:task,complete:false)} catch {result.issues.append(.init(id:"recurrent.structure",message:error.localizedDescription));return result}
        do {_ = try RecurrentLearningSubmission.validatedGraph(graph,task:task,complete:true)} catch {result.issues.append(.init(id:"recurrent.connections",message:error.localizedDescription))}
        let inputs=graph.nodes.filter {$0.type=="input"},operators=graph.nodes.filter {$0.type==task.componentID},outputs=graph.nodes.filter {$0.type=="output"}
        result.orderedNodeIDs=(inputs+operators+outputs).map(\.id)
        for node in inputs {if let role=node.config["fixture"]?.stringValue {result.outputTensorContracts[.init(node.id,"out")]=try? .init(sample:task.sampleShapes[role]!,dtype:.float64)}}
        if let node=operators.first,let c=try? RecurrentConfiguration(componentID:node.type,values:node.config) {
            let edges=c.configuredInputs.compactMap {port in graph.edges.first {$0.target==node.id && $0.targetPort==port}}
            let layouts=edges.compactMap {result.outputTensorContracts[.init($0.source,$0.sourcePort)]?.layout}
            if layouts.count==c.configuredInputs.count,let resolved=try? c.resolved(layouts:layouts) {for (port,layout) in resolved.outputs {result.outputTensorContracts[.init(node.id,port)] = .init(layout:layout,dtype:.float64)}}
        }
        for node in outputs {if let edge=graph.edges.first(where:{$0.target==node.id}),let contract=result.outputTensorContracts[.init(edge.source,edge.sourcePort)] {result.outputTensorContracts[.init(node.id,"out")]=contract}}
        return result
    }
}
