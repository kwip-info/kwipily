import Foundation

public enum EmbeddingBagExamples {
    /// A complete no-code construction. Runtime fixtures supply real source rows.
    public static func lessonGraph(task:EmbeddingBagLearningTask)throws->GraphDocument {
        let source=GraphNode(id:"bags",type:"bag_input",name:"Observation groups",x:30,y:180,config:try task.sourceConfiguration.values)
        let bag=GraphNode(id:"combine",type:"embedding_bag",name:task.title,x:280,y:180,config:try task.configuration.values)
        let output=GraphNode(id:"result",type:"output",name:"One vector per observation",x:530,y:180)
        let ports=try task.configuration.inputPorts
        let edges=ports.map {GraphEdge(id:"bag-"+$0,source:source.id,sourcePort:$0,target:bag.id,targetPort:$0)}+[GraphEdge(id:"bag-output",source:bag.id,target:output.id)]
        return GraphDocument(name:task.title,nodes:[source,bag,output],edges:edges)
    }
}
