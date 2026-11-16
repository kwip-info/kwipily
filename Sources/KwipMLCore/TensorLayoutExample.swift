import Foundation

extension ExampleFlows {
    /// A trainable demonstration of coordinate rearrangement, not a claim of added model capacity.
    public static var tensorLayoutClassifier:GraphDocument {
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"project",type:"linear",name:"Learn twelve features",x:260,y:180,config:["in_features":2,"out_features":12,"bias":true]),
            .init(id:"shape",type:"reshape",name:"Arrange as three by four",x:490,y:180,config:["shape":.array([3,4])]),
            .init(id:"swap",type:"transpose",name:"Swap sample axes",x:720,y:180,config:["dim0":1,"dim1":2]),
            .init(id:"flat",type:"flatten",name:"Return to feature vector",x:950,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1180,y:180,config:["in_features":12,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1410,y:180)
        ]
        return GraphDocument(name:"Tensor layout classifier",nodes:nodes,edges:zip(nodes,nodes.dropFirst()).enumerated().map {
            GraphEdge(id:"e\($0.offset)",source:$0.element.0.id,target:$0.element.1.id)
        })
    }
}
