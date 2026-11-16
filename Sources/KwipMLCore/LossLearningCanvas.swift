import Foundation

public enum LossLearningCanvas {
    public struct PortPresentation:Equatable {
        public let shape:[Int]
        public let dtype:ContractDType
        public let observations:Bool
        public var displayShape:String {"["+(observations ? ["N"]+shape.dropFirst().map(String.init):shape.map(String.init)).joined(separator:", ")+"]"}
    }
    public static let resultRoles=["raw","objective","inputInfluence"]
    public static func registry(task:LossLearningTask)->ComponentRegistry {
        var input=ComponentDefinition(id:"input",title:"Supplied data",category:"Boundaries",outputs:["out"])
        input.fields=[.init(key:"fixture",title:"Supplied role",kind:.choice,defaultValue:.string(task.sourceRoles[0]),min:nil,max:nil,choices:task.sourceRoles.map {.init(value:$0,title:$0.replacingOccurrences(of:"_",with:" ").capitalized)})]
        var output=ComponentDefinition(id:"output",title:"Measured result",category:"Boundaries",inputs:["in"],outputs:["out"])
        output.fields=[.init(key:"role",title:"Measured role",kind:.choice,defaultValue:.string("raw"),min:nil,max:nil,choices:resultRoles.map {.init(value:$0,title:$0 == "raw" ? "Individual scores":$0 == "objective" ? "Combined score":"Input influence")})]
        var check=ComponentDefinition(id:"objective_check",title:"Objective check",category:"Evaluation",inputs:task.sourceRoles+["objective"],outputs:resultRoles)
        check.fields=[.init(key:"scalarization",title:"Combine kept scores",kind:.choice,defaultValue:.string("mean"),min:nil,max:nil,choices:[.init(value:"mean",title:"Mean"),.init(value:"sum",title:"Sum")])]
        var definitions=[input,BundledLossComponents.definition(task.componentID),check,output]
        if task.usesLogSoftmax {definitions.append(BundledActivationComponents.definition("log_softmax"))}
        return .init(components:definitions)
    }
    public static func inspect(_ graph:GraphDocument,task:LossLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {_ = try LossLearningSubmission.validatedGraph(graph,task:task,complete:false)} catch {result.issues.append(.init(id:"loss.structure",message:error.localizedDescription));return result}
        do {_ = try LossLearningSubmission.validatedGraph(graph,task:task,complete:true)} catch {result.issues.append(.init(id:"loss.connections",message:error.localizedDescription))}
        result.orderedNodeIDs=graph.nodes.sorted {rank($0)<rank($1)}.map(\.id)
        func rank(_ node:GraphNode)->Int {node.type == "input" ? 0:node.type == "output" ? 3:node.type == "objective_check" ? 2:1}
        for node in graph.nodes {
            for port in registry(task:task).component(type:node.type)?.outputs ?? [] {
                guard let presented=presentation(graph,task:task,nodeID:node.id,port:port,direction:.output),presented.observations,presented.shape.count>1,let contract=try? TypedTensorContract(sample:Array(presented.shape.dropFirst()),dtype:presented.dtype) else {continue}
                result.outputTensorContracts[.init(node.id,port)]=contract
            }
        }
        return result
    }
    /// Full fixture shapes also represent scalar roles honestly; batched sample contracts
    /// remain unchanged and never pretend that [N] is [N,1].
    public static func presentation(_ graph:GraphDocument,task:LossLearningTask,nodeID:String,port:String,direction:PortDirection)->PortPresentation? {
        guard let node=graph.nodes.first(where:{$0.id == nodeID}) else {return nil}
        if direction == .input {
            guard let edge=graph.edges.first(where:{$0.target==nodeID && $0.targetPort==port}) else {return nil}
            return presentation(graph,task:task,nodeID:edge.source,port:edge.sourcePort,direction:.output)
        }
        let inputs=task.fixtures(seed:0)[0].inputs
        if node.type == "input",port == "out",let role=node.config["fixture"]?.stringValue,let input=inputs[role] {
            return .init(shape:input.shape,dtype:role != "prediction" && [.crossEntropy,.nll,.ctc].contains(task) ? .int64:.float64,observations:true)
        }
        if node.type == "log_softmax",port == "out",graph.edges.contains(where:{$0.target==nodeID}) {return .init(shape:inputs["prediction"]!.shape,dtype:.float64,observations:true)}
        if node.type == "objective_check",resultRoles.contains(port),graph.edges.contains(where:{$0.target==nodeID && $0.targetPort=="prediction"}) {
            let shape=port == "objective" ? [1]:port == "inputInfluence" || [.bce,.mse,.l1,.kl].contains(task) ? inputs["prediction"]!.shape:[2]
            return .init(shape:shape,dtype:.float64,observations:port != "objective")
        }
        if node.type == "output",port == "out",let edge=graph.edges.first(where:{$0.target==nodeID}) {return presentation(graph,task:task,nodeID:edge.source,port:edge.sourcePort,direction:.output)}
        return nil
    }
    public static func preparedGraph(task:LossLearningTask)->GraphDocument {
        var nodes=task.sourceRoles.enumerated().map {index,role in GraphNode(id:role,type:"input",name:role,x:0,y:Double(index*90),config:["fixture":.string(role)])}
        nodes.append(.init(id:"objective",type:task.componentID,name:task.title,x:180,y:0,config:task.configuration))
        nodes.append(.init(id:"check",type:"objective_check",name:"Objective check",x:360,y:90,config:["scalarization":.string("mean")]))
        var edges:[GraphEdge]=[]
        if task.usesLogSoftmax {
            nodes.append(.init(id:"normalization",type:"log_softmax",name:"LogSoftmax · Prepare probabilities",x:180,y:180,config:["dim":-1]))
            edges.append(.init(source:"prediction",target:"normalization"));edges.append(.init(source:"normalization",target:"check",targetPort:"prediction"))
        } else {edges.append(.init(source:"prediction",target:"check",targetPort:"prediction"))}
        for role in task.sourceRoles where role != "prediction" {edges.append(.init(source:role,target:"check",targetPort:role))}
        edges.append(.init(source:"objective",target:"check",targetPort:"objective"))
        for (index,role) in ["raw","objective","inputInfluence"].enumerated() {
            nodes.append(.init(id:"result-"+role,type:"output",name:role == "raw" ? "Individual scores":role == "objective" ? "Combined score":"Input influence",x:540,y:Double(index*90),config:["role":.string(role)]))
            edges.append(.init(source:"check",sourcePort:role,target:"result-"+role))
        }
        return .init(name:task.title,nodes:nodes,edges:edges)
    }
}
