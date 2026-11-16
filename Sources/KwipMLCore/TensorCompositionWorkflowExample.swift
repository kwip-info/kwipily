import Foundation

extension ExampleFlows {
    /// Learned feature arrays demonstrate joining/reordering without mixing observations.
    public static func tensorCompositionClassifier(componentID: String) throws -> GraphDocument {
        let left: [Int], right: [Int], configuration: [String: JSONValue]
        switch componentID {
        case "concat": left = [2,3]; right = [1,3]; configuration = ["dim":1]
        case "add": left = [2,3]; right = [3]; configuration = ["alpha":-0.5,"sample_broadcast":true]
        case "stack": left = [2,3]; right = [2,3]; configuration = ["dim":2]
        case "permute": left = [2,3,4]; right = []; configuration = ["dims":.array([0,2,3,1])]
        default: throw GraphConfigurationError("Unknown tensor composition example.")
        }
        let settings = try TensorCompositionConfiguration(componentID: componentID, values: configuration)
        let shapes = try ([left] + (right.isEmpty ? [] : [right])).map { try BatchedTensorLayout(sample:$0) }
        let output = try settings.resolved(layouts: shapes)
        var nodes: [GraphNode] = [
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"left",type:"linear",name:"Learn first array",x:260,y:80,config:["in_features":2,"out_features":.integer(Int64(shapes[0].sample.elementCount)),"bias":true]),
            .init(id:"a",type:"reshape",name:"First sample directions",x:490,y:80,config:["shape":.array(left.map { .integer(Int64($0)) })]),
            .init(id:"composition",type:componentID,name:"Combine sample information",x:720,y:180,config:configuration),
            .init(id:"flat",type:"flatten",name:"Classification features",x:950,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1180,y:180,config:["in_features":.integer(Int64(output.sample.elementCount)),"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1410,y:180)
        ]
        var links = [("input","left","in"),("left","a","in"),("a","composition",right.isEmpty ? "in" : "a"),("composition","flat","in"),("flat","head","in"),("head","output","in")]
        if !right.isEmpty {
            nodes.append(.init(id:"right",type:"linear",name:"Learn second array",x:260,y:300,config:["in_features":2,"out_features":.integer(Int64(shapes[1].sample.elementCount)),"bias":true]))
            nodes.append(.init(id:"b",type:"reshape",name:"Second sample directions",x:490,y:300,config:["shape":.array(right.map { .integer(Int64($0)) })]))
            links += [("input","right","in"),("right","b","in"),("b","composition","b")]
        }
        return GraphDocument(name:"\(componentID.capitalized) feature-array classifier",nodes:nodes,edges:links.enumerated().map {
            GraphEdge(id:"composition-wire-\($0.offset)",source:$0.element.0,target:$0.element.1,targetPort:$0.element.2)
        },execution:ExecutionSettings(device:"auto",epochs:20,batchSize:128,sampleLimit:512,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}

extension WorkflowExamples {
    public static func preparedConcatClassifier(projectID:String = UUID().uuidString,device:String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorCompositionClassifier(componentID:"concat"))
    }
    public static func preparedAddClassifier(projectID:String = UUID().uuidString,device:String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorCompositionClassifier(componentID:"add"))
    }
    public static func preparedStackClassifier(projectID:String = UUID().uuidString,device:String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorCompositionClassifier(componentID:"stack"))
    }
    public static func preparedPermuteClassifier(projectID:String = UUID().uuidString,device:String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorCompositionClassifier(componentID:"permute"))
    }
}
