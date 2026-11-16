import Foundation

extension ExampleFlows {
    /// Hidden-axis normalization; the final Linear emits unnormalized class scores.
    public static var softmaxClassifier:GraphDocument {
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"project",type:"linear",name:"Learn six scores",x:260,y:180,config:["in_features":2,"out_features":6,"bias":true]),
            .init(id:"shape",type:"reshape",name:"Two groups of three",x:490,y:180,config:["shape":.array([2,3])]),
            .init(id:"probabilities",type:"softmax",name:"Normalize each group",x:720,y:180,config:["dim":-1]),
            .init(id:"flat",type:"flatten",name:"Combine group probabilities",x:950,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1180,y:180,config:["in_features":6,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1410,y:180)
        ]
        return GraphDocument(name:"Softmax axis classifier",nodes:nodes,edges:zip(nodes,nodes.dropFirst()).enumerated().map {
            GraphEdge(id:"e\($0.offset)",source:$0.element.0.id,target:$0.element.1.id)
        })
    }
}
