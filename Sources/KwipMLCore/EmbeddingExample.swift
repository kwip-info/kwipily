import Foundation

extension ExampleFlows {
    /// Two categorical coordinates share a learned table; flatten retains their order.
    public static var embeddingClassifier:GraphDocument {
        let nodes:[GraphNode]=[
            .init(id:"input",type:"index_input",name:"Category pair",x:30,y:180,config:["shape":.array([2])]),
            .init(id:"lookup",type:"embedding",name:"Shared category vectors",x:260,y:180,config:["num_embeddings":4,"embedding_dim":3,"padding_idx":.null,"max_norm":.null,"norm_type":2,"scale_grad_by_freq":false,"sparse":false,"freeze":false]),
            .init(id:"flat",type:"flatten",name:"Keep category order",x:490,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:720,y:180,config:["in_features":6,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:950,y:180)
        ]
        return GraphDocument(name:"Embedding classifier",nodes:nodes,edges:zip(nodes,nodes.dropFirst()).enumerated().map {GraphEdge(id:"e\($0.offset)",source:$0.element.0.id,target:$0.element.1.id)})
    }
}
