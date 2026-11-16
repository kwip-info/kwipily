import Foundation

extension ExampleFlows {
    /// Learned matrix interactions with a right operand shared across two sample heads.
    /// This is a classifier teaching example, not an attention implementation.
    public static var matrixClassifier:GraphDocument {
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"left",type:"linear",name:"Learn left matrices",x:260,y:80,config:["in_features":2,"out_features":24,"bias":true]),
            .init(id:"right",type:"linear",name:"Learn shared right matrix",x:260,y:300,config:["in_features":2,"out_features":8,"bias":true]),
            .init(id:"a",type:"reshape",name:"Two heads of 3 by 4",x:490,y:80,config:["shape":.array([2,3,4])]),
            .init(id:"b",type:"reshape",name:"Shared 4 by 2",x:490,y:300,config:["shape":.array([4,2])]),
            .init(id:"product",type:"matmul",name:"Row–column interactions",x:720,y:180),
            .init(id:"flat",type:"flatten",name:"Combine head features",x:950,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1180,y:180,config:["in_features":12,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1410,y:180)
        ]
        let links=[("input","left","in"),("input","right","in"),("left","a","in"),("right","b","in"),
                   ("a","product","a"),("b","product","b"),("product","flat","in"),("flat","head","in"),("head","output","in")]
        return GraphDocument(name:"Matrix interaction classifier",nodes:nodes,edges:links.enumerated().map {
            GraphEdge(id:"matrix-wire-\($0.offset)",source:$0.element.0,target:$0.element.1,targetPort:$0.element.2)
        },execution:ExecutionSettings(device:"auto",epochs:20,batchSize:128,sampleLimit:512,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}

extension WorkflowExamples {
    public static func preparedMatrixClassifier(projectID:String = UUID().uuidString,device:String = "auto")throws->WorkflowDocument {
        var document=try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.matrixClassifier)
        // Authored positions apply only when creating this example. Persisted layouts are untouched.
        let data:[String:(Int,Double)]=["observations":(0,40),"cast":(1,40),"split":(2,40),
            "fit":(3,40),"training":(4,40),"validation":(4,210)]
        let model:[String:(Int,Double)]=["input":(0,380),"left":(1,300),"right":(1,470),
            "a":(2,300),"b":(2,470),"product":(3,380),"flat":(4,380),
            "head":(5,380),"output":(6,380)]
        let control:[String:(Int,Double)]=["mapping":(5,125),"model":(7,380),"setup":(7,210),"train":(8,300)]
        for index in document.nodes.indices {
            let address=document.nodes[index].address
            let positions=address.domain == .data ? data : address.domain == .model ? model : control
            if let (column,y)=positions[address.localID] {
                document.nodes[index].x=40+Double(column)*230
                document.nodes[index].y=y
            }
        }
        return document
    }
}
