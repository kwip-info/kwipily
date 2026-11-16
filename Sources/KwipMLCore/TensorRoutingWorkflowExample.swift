import Foundation

extension ExampleFlows {
    /// Real learned arrays expose named parts, typed winning positions and two-source arrangements.
    public static func tensorRoutingClassifier(componentID:String)throws->GraphDocument {
        var nodes:[GraphNode]=[.init(id:"input",type:"input",name:"Measurements",x:30,y:180,config:["features":2])]
        var edges:[GraphEdge]=[]
        func node(_ id:String,_ type:String,_ name:String,_ x:Double,_ y:Double,_ config:[String:JSONValue]=[:]) {
            nodes.append(.init(id:id,type:type,name:name,x:x,y:y,config:config))
        }
        func wire(_ source:String,_ target:String,_ targetPort:String="in",_ sourcePort:String="out") {
            edges.append(.init(id:"routing-wire-\(edges.count)",source:source,sourcePort:sourcePort,target:target,targetPort:targetPort))
        }
        func branch(_ id:String,_ shape:[Int],_ y:Double) {
            let count=shape.reduce(1,*)
            node(id+"-learn","linear","Learn sample values",250,y,["in_features":2,"out_features":.integer(Int64(count)),"bias":true])
            node(id,"reshape","Sample directions",470,y,["shape":.array(shape.map {.integer(Int64($0))})])
            wire("input",id+"-learn");wire(id+"-learn",id)
        }
        let flatWidth:Int
        switch componentID {
        case "split":
            branch("array",[2,6],180)
            node("routing","split","Separate ordered parts",690,180,["mode":.string("size"),"dim":2,"split_size":2,"sections":.array([1,1])])
            node("join-first","concat","Last then first",910,100,["dim":2])
            node("join-all","concat","Append middle part",1130,180,["dim":2])
            wire("array","routing");wire("routing","join-first","a","part2");wire("routing","join-first","b","part0")
            wire("routing","join-all","b","part1");wire("join-first","join-all","a");wire("join-all","flat")
            flatWidth=12
        case "chunk":
            branch("array",[2,13],180)
            node("routing","chunk","Request six chunks",690,180,["chunks":6,"dim":2])
            wire("array","routing");wire("routing","flat","in","part4")
            flatWidth=2
        case "gather":
            branch("array",[2,4],180)
            node("positions","max_pool1d","Winning positions",690,330,["kernel_size":2,"stride":2,"padding":0,"dilation":1,"ceil_mode":false,"return_indices":true])
            node("routing","gather","Select winning source values",910,180,["dim":2])
            wire("array","positions");wire("array","routing","source");wire("positions","routing","indices","indices");wire("routing","flat")
            flatWidth=4
        case "einsum":
            branch("a",[2,3],80);branch("b",[3,2],300)
            node("routing","einsum","Combine learned tables",690,180,["pattern":.string("multiply"),"equation":.string("nab,nbc->nac")])
            wire("a","routing","a");wire("b","routing","b");wire("routing","flat")
            flatWidth=4
        default:throw GraphConfigurationError("Unknown tensor routing example.")
        }
        node("flat","flatten","Classification features",1350,180,["start_dim":1,"end_dim":-1])
        node("head","linear","Class scores",1570,180,["in_features":.integer(Int64(flatWidth)),"out_features":2,"bias":true])
        node("output","output","Prediction",1790,180)
        wire("flat","head");wire("head","output")
        return GraphDocument(name:"\(componentID.capitalized) feature-array classifier",nodes:nodes,edges:edges,
            execution:ExecutionSettings(device:"auto",epochs:20,batchSize:128,sampleLimit:512,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}

extension WorkflowExamples {
    public static func preparedSplitClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorRoutingClassifier(componentID:"split"))
    }
    public static func preparedChunkClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorRoutingClassifier(componentID:"chunk"))
    }
    public static func preparedGatherClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorRoutingClassifier(componentID:"gather"))
    }
    public static func preparedEinsumClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {
        try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.tensorRoutingClassifier(componentID:"einsum"))
    }
}
