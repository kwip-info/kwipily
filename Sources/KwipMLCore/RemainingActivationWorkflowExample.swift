import Foundation

extension ExampleFlows {
    public static func remainingActivationClassifier(componentID:String)throws->GraphDocument {
        let settings:[String:JSONValue],title:String
        switch componentID {
        case "prelu":settings=["num_parameters":2,"init":0.25];title="Learned channel slopes"
        case "log_softmax":settings=["dim":1];title="Compressed sample scores"
        case "dropout":settings=["p":0.35];title="Training-only signal removal"
        default:throw GraphConfigurationError("Unknown activation example.")
        }
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2]),
            .init(id:"project",type:"linear",name:"Learn six scores",x:260,y:180,config:["in_features":2,"out_features":6,"bias":true]),
            .init(id:"shape",type:"reshape",name:"Two channels of three",x:490,y:180,config:["shape":.array([2,3])]),
            .init(id:"activation",type:componentID,name:title,x:720,y:180,config:settings),
            .init(id:"flat",type:"flatten",name:"Classification features",x:950,y:180,config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",name:"Class scores",x:1180,y:180,config:["in_features":6,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Prediction",x:1410,y:180)
        ]
        return GraphDocument(name:title+" classifier",nodes:nodes,edges:zip(nodes,nodes.dropFirst()).enumerated().map {GraphEdge(id:"activation-wire-\($0.offset)",source:$0.element.0.id,target:$0.element.1.id)})
    }
}
extension WorkflowExamples {
    public static func preparedPReLUClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedRemainingActivationClassifier("prelu",projectID:projectID,device:device)}
    public static func preparedLogSoftmaxClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedRemainingActivationClassifier("log_softmax",projectID:projectID,device:device)}
    public static func preparedDropoutClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedRemainingActivationClassifier("dropout",projectID:projectID,device:device)}
    private static func preparedRemainingActivationClassifier(_ id:String,projectID:String,device:String)throws->WorkflowDocument {
        var document=try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.remainingActivationClassifier(componentID:id))
        // Compact authored placement only applies to new examples, never restored user files.
        let data:[String:(Int,Double)]=["observations":(0,40),"cast":(1,40),"split":(2,40),"fit":(3,40),"training":(4,40),"validation":(4,210)]
        let model:[String:(Int,Double)]=["input":(0,380),"project":(1,380),"shape":(2,380),"activation":(3,380),"flat":(4,380),"head":(5,380),"output":(6,380)]
        let control:[String:(Int,Double)]=["mapping":(5,125),"model":(7,380),"setup":(7,210),"train":(8,300)]
        for i in document.nodes.indices {
            let address=document.nodes[i].address
            let positions=address.domain == .data ? data:address.domain == .model ? model:control
            if let (column,y)=positions[address.localID] {document.nodes[i].x=40+Double(column)*230;document.nodes[i].y=y}
        }
        return document
    }
}
