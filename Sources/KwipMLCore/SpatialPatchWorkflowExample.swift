import Foundation

extension ExampleFlows {
    public static func spatialPatchClassifier(componentID: String) throws -> GraphDocument {
        guard SpatialPatchConfiguration.componentIDs.contains(componentID) else { throw GraphConfigurationError("Unknown patch example.") }
        let extracting = componentID == "unfold"
        let learnedWidth = extracting ? 20 : 48
        let sampleShape = extracting ? [1,4,5] : [6,8]
        let flatWidth = extracting ? 48 : 20
        var configuration: [String:JSONValue] = ["kernel_size":.array([3,2]),"stride":.array([1,1]),"dilation":.array([1,1]),"padding":.array([0,0])]
        if !extracting { configuration["output_size"] = .array([4,5]) }
        let nodes: [GraphNode] = [
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"learn",type:"linear",name:"Learn sample values",x:250,y:180,config:["in_features":2,"out_features":.integer(Int64(learnedWidth)),"bias":true]),
            .init(id:"array",type:"reshape",name:extracting ? "Sample grid":"Patch columns",x:470,y:180,config:["shape":.array(sampleShape.map { .integer(Int64($0)) })]),
            .init(id:"patch",type:componentID,name:extracting ? "Extract nearby cells":"Add overlapping parts",x:690,y:180,config:configuration),
            .init(id:"flat",type:"flatten",name:"Classification features",x:910,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1130,y:180,config:["in_features":.integer(Int64(flatWidth)),"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1350,y:180)
        ]
        let edges = zip(nodes,nodes.dropFirst()).enumerated().map { index,pair in GraphEdge(id:"patch-wire-\(index)",source:pair.0.id,sourcePort:"out",target:pair.1.id,targetPort:"in") }
        return GraphDocument(name:"\(componentID.capitalized) patch-feature classifier",nodes:nodes,edges:edges,execution:ExecutionSettings(device:"auto",epochs:20,batchSize:128,sampleLimit:512,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}
extension WorkflowExamples {
    public static func preparedUnfoldClassifier(projectID:String=UUID().uuidString,device:String="auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.spatialPatchClassifier(componentID:"unfold"))
    }
    public static func preparedFoldClassifier(projectID:String=UUID().uuidString,device:String="auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.spatialPatchClassifier(componentID:"fold"))
    }
}
