import Foundation

/// Small causal token model; demonstrates rotary Q/K, not a complete Llama architecture.
public enum RotaryPositionPracticeExample {
    public static let componentID="rotary_position"
    public static let address=WorkflowNodeAddress(domain:.model,localID:"rotary")
    public static func document(projectID:String=UUID().uuidString,device:String="cpu")throws->WorkflowDocument {
        var d=try TextGenerationPracticeExample.document(projectID:projectID,device:device)
        let lookup=WorkflowNodeAddress(domain:.model,localID:"lookup"),head=WorkflowNodeAddress(domain:.model,localID:"head")
        d.edges.removeAll {$0.from.address==lookup && $0.to.address==head}
        for i in d.nodes.indices {
            if d.nodes[i].address==lookup,case .modelPrimitive(let type,let version,var c)=d.nodes[i].payload {c["embedding_dim"]=4;d.nodes[i].payload = .modelPrimitive(type:type,configurationVersion:version,config:c)}
            if d.nodes[i].address==head,case .modelPrimitive(let type,let version,var c)=d.nodes[i].payload {c["in_features"]=4;d.nodes[i].payload = .modelPrimitive(type:type,configurationVersion:version,config:c);d.nodes[i].x=2500}
            if d.nodes[i].address.localID=="model" {d.nodes[i].name="Causal rotary token model"}
        }
        func address(_ id:String)->WorkflowNodeAddress {.init(domain:.model,localID:id)}
        func add(_ id:String,_ type:String,_ name:String,_ x:Double,_ y:Double,_ config:[String:JSONValue]) {
            d.nodes.append(.init(address:address(id),name:name,x:x,y:y,payload:.modelPrimitive(type:type,configurationVersion:2,config:config)))
        }
        func wire(_ from:WorkflowNodeAddress,_ out:String="out",_ to:WorkflowNodeAddress,_ input:String="in") {
            d.edges.append(.init(from:.init(address:from,port:out),to:.init(address:to,port:input),kind:.tensor))
        }
        add("rms","rms_norm","Normalize token features",600,600,["normalized_shape":.array([4]),"eps":.number(1e-5),"elementwise_affine":true]);wire(lookup,"out",address("rms"))
        for (i,role) in ["query","key","value"].enumerated() {
            let y=400+Double(i)*200
            add(role+"-projection","linear",role.capitalized+" projection",900,y,["in_features":4,"out_features":4,"bias":false])
            add(role+"-heads","reshape",role.capitalized+" head",1200,y,["shape":.array([1,3,4])])
            wire(address("rms"),"out",address(role+"-projection"));wire(address(role+"-projection"),"out",address(role+"-heads"))
        }
        add("rotary","rotary_position","Rotate Query and Key",1500,400,RotaryPositionConfiguration.defaults)
        wire(address("query-heads"),"out",Self.address,"query");wire(address("key-heads"),"out",Self.address,"key")
        add("causal-attention","scaled_dot_product_attention","Attend to earlier positions",1800,600,["mask_mode":.string("causal"),"dropout":0,"scale":.null])
        wire(Self.address,"query",address("causal-attention"),"query");wire(Self.address,"key",address("causal-attention"),"key");wire(address("value-heads"),"out",address("causal-attention"),"value")
        add("merge-heads","reshape","Sequence features",2150,600,["shape":.array([3,4])]);wire(address("causal-attention"),"out",address("merge-heads"));wire(address("merge-heads"),"out",head)
        d.name="Rotate Query and Key in a causal token model"
        try WorkflowCodec.validate(d);return d
    }
}
extension WorkflowExamples {
    public static func preparedRotaryPositionClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try RotaryPositionPracticeExample.document(projectID:projectID,device:device)}
}
