import Foundation

extension ExampleFlows {
    public static var tensorFlowClassifier:GraphDocument {
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",name:"Two prepared measurements",x:40,y:380,config:["features":2]),
            .init(id:"hidden",type:"linear",name:"Eight learned features",x:270,y:380,config:["in_features":2,"out_features":8,"bias":true]),
            .init(id:"activation",type:"relu",name:"Positive responses",x:500,y:380),
            .init(id:"logits",type:"linear",name:"Two class scores",x:730,y:380,config:["in_features":8,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Classifier output",x:960,y:380)]
        return .init(name:"TensorFlow dense classifier",nodes:nodes,edges:zip(nodes,nodes.dropFirst()).enumerated().map{.init(id:"tf-wire-\($0.offset)",source:$0.element.0.id,target:$0.element.1.id)})
    }
}
extension WorkflowExamples {
    public static let tensorFlowEstimatorAddress=WorkflowNodeAddress(domain:.control,localID:"estimator")
    public static func preparedTensorFlowClassifier(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var document=try WorkflowCodec.migrateToVersion5(preparedClassifier(projectID:projectID,device:"cpu",model:ExampleFlows.tensorFlowClassifier))
        let removed:Set<WorkflowNodeAddress>=Set(["mapping","setup","train"].map{.init(domain:.control,localID:$0)})
        document.nodes.removeAll{removed.contains($0.address)}
        document.edges.removeAll{removed.contains($0.from.address) || removed.contains($0.to.address)}
        document.nodes.append(.init(address:tensorFlowEstimatorAddress,name:"Train TensorFlow classifier",x:1840,y:350,payload:.estimator(config:["backendID":.string("tensorflow"),"featureColumns":.array([.string("signal"),.string("context")]),"targetColumn":.string("label"),"settings":.object(TensorFlowEstimatorConfiguration.defaultSettings)])))
        for (id,domain,port) in [("training",WorkflowDomain.data,"table"),("validation",.data,"table"),("model",.control,"model")] {
            document.edges.append(.init(id:"tf-estimator-\(id)",from:.init(address:.init(domain:domain,localID:id == "model" ? id:"fit"),port:id == "model" ? port:id),to:.init(address:tensorFlowEstimatorAddress,port:id),kind:id=="model" ? .modelDefinition : .table))
        }
        try WorkflowCodec.validate(document)
        return document
    }
}
