import Foundation

public enum MixtureExpertsPractice {
    public static let lessonID="mixture-experts"
    public static func graph()throws->GraphDocument {
        let registry=try ComponentRegistry.builtIn()
        guard let definition=registry.component(type:"mixture_experts") else {throw GraphConfigurationError("MoE component is unavailable.")}
        return .init(name:"Editable routed classifier",nodes:[.init(id:"input",type:"input",name:"Observation features",x:40,y:160,config:["features":2]),.init(id:"mixture",type:"mixture_experts",name:"Feature mixture",x:300,y:160,config:Dictionary(uniqueKeysWithValues:definition.fields.map{($0.key,$0.defaultValue)})),.init(id:"output",type:"output",name:"Class scores",x:2300,y:160)],edges:[.init(id:"input-mixture",source:"input",target:"mixture"),.init(id:"mixture-output",source:"mixture",target:"output")])
    }
    public static func document(projectID:String=UUID().uuidString,device:String="cpu")throws->WorkflowDocument {
        var result=try WorkflowExamples.preparedClassifier(projectID:projectID,device:device,model:graph())
        // Keep the expanded mixture's chains separate from the table pipeline and controls.
        for index in result.nodes.indices {
            if result.nodes[index].address.domain == .model{result.nodes[index].y+=450}
            if result.nodes[index].address.domain == .control{result.nodes[index].x+=1200}
        }
        result.name="Route observations through experts"
        return result
    }
}
