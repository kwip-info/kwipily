import Foundation

extension ExampleFlows {
    public static func recurrentClassifier(componentID:String)throws->GraphDocument {
        let cell=componentID.hasSuffix("_cell")
        let config=try RecurrentConfiguration(componentID:componentID,values:cell ? [:]:["input_size":1]).values
        var nodes:[GraphNode] = [.init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2])]
        if !cell {nodes.append(.init(id:"sequence",type:"reshape",name:"Two ordered values",x:250,y:180,config:["shape":.array([2,1])]))}
        nodes.append(.init(id:"memory",type:componentID,name:"Remember earlier values",x:cell ? 250:470,y:180,config:config))
        if !cell {nodes.append(.init(id:"flat",type:"flatten",name:"Sequence features",x:690,y:180,config:["start_dim":1,"end_dim":-1]))}
        nodes.append(.init(id:"head",type:"linear",name:"Class scores",x:cell ? 470:910,y:180,config:["in_features":cell ? 2:4,"out_features":2,"bias":true]))
        nodes.append(.init(id:"output",type:"output",name:"Prediction",x:cell ? 690:1130,y:180))
        let edges=zip(nodes,nodes.dropFirst()).enumerated().map {index,pair in GraphEdge(id:"memory-wire-\(index)",source:pair.0.id,sourcePort:"out",target:pair.1.id,targetPort:"in")}
        return GraphDocument(name:"\(componentID) memory classifier",nodes:nodes,edges:edges,execution:ExecutionSettings(device:"auto",epochs:3,batchSize:32,sampleLimit:64,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}
extension WorkflowExamples {
    public static func preparedRNNClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.recurrentClassifier(componentID:"rnn"))}
    public static func preparedLSTMClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.recurrentClassifier(componentID:"lstm"))}
    public static func preparedGRUClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.recurrentClassifier(componentID:"gru"))}
    public static func preparedRNNCellClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.recurrentClassifier(componentID:"rnn_cell"))}
    public static func preparedLSTMCellClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.recurrentClassifier(componentID:"lstm_cell"))}
    public static func preparedGRUCellClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.recurrentClassifier(componentID:"gru_cell"))}
}
