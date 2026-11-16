import Foundation

extension ExampleFlows {
    /// Compare normalization across each feature group with normalization across the whole sample.
    /// LayerNorm learns scale/bias; the final Linear still emits raw classifier logits.
    public static func layerNormClassifier(normalizeWholeSample: Bool = false) -> GraphDocument {
        let nodes: [GraphNode] = [
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"project",type:"linear",name:"Learn six features",x:260,y:180,config:["in_features":2,"out_features":6,"bias":true]),
            .init(id:"shape",type:"reshape",name:"Two groups of three",x:490,y:180,config:["shape":.array([2,3])]),
            .init(id:"normalize",type:"layer_norm",name:normalizeWholeSample ? "Normalize the sample":"Normalize each group",x:720,y:180,
                  config:["normalized_shape":.array(normalizeWholeSample ? [2,3]:[3]),"eps":.number(1e-5),"elementwise_affine":true,"bias":true]),
            .init(id:"activation",type:"gelu",name:"Smooth activation",x:950,y:180,config:["approximate_tanh":false]),
            .init(id:"flat",type:"flatten",name:"Combine learned features",x:1180,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1410,y:180,config:["in_features":6,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1640,y:180)
        ]
        return GraphDocument(name:normalizeWholeSample ? "LayerNorm classifier · whole sample":"LayerNorm classifier · grouped features",
            nodes:nodes,edges:zip(nodes,nodes.dropFirst()).enumerated().map {
                GraphEdge(id:"e\($0.offset)",source:$0.element.0.id,target:$0.element.1.id)
            },execution:ExecutionSettings(device:"auto",epochs:20,batchSize:128,sampleLimit:512,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}
