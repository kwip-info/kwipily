import Foundation

/// A pointwise bigram model: no future-token attention or generative-quality claim.
public enum SequenceTrainingPracticeExample {
    public static let exampleID="sequence-training"
    public static func document(projectID:String=UUID().uuidString,device:String="cpu")throws->WorkflowDocument {
        var d=try TextEncodingPracticeExample.trainingDocument(projectID:projectID,device:device)
        let windows=SequenceWindowsPracticeExample.address,text=TextEncodingPracticeExample.address
        var c=SequenceWindowsConfiguration.defaults
        c["mode"] = .string("partitions");c["contextLength"]=3;c["stride"]=3;c["incomplete"] = .string("pad")
        d.nodes.append(.init(address:windows,name:"Next-token examples",x:1000,y:40,payload:.dataOperator(operatorID:SequenceWindowsConfiguration.operatorID,config:c)))
        for i in d.edges.indices where d.edges[i].to.address==WorkflowExamples.trainAddress && ["training","validation"].contains(d.edges[i].to.port) {d.edges[i].from.address=windows}
        for role in ["training","validation"] {d.edges.append(.init(from:.init(address:text,port:role),to:.init(address:windows,port:role),kind:.table))}
        let flat=WorkflowNodeAddress(domain:.model,localID:"flat")
        let incoming=d.edges.first {$0.to.address==flat},outgoing=d.edges.first {$0.from.address==flat}
        d.edges.removeAll {$0.to.address==flat || $0.from.address==flat};d.nodes.removeAll {$0.address==flat}
        if let incoming,let outgoing {d.edges.append(.init(from:incoming.from,to:outgoing.to,kind:.tensor))}
        var plan=NamedInputPlan(inputs:[.init(name:"tokens",nodeID:"input",representation:"tokens",dtype:"int64",sampleShape:[3],columns:["input_ids"],lengthPolicy:"error",paddingValue:1,vocabularySize:4)],roles:[.init(name:"target",representation:"tokens",dtype:"int64",sampleShape:[3],columns:["target_ids"],lengthPolicy:"error",paddingValue:1,vocabularySize:4)])
        plan.schemaVersion=2;plan.sequenceObjective = .init(inputName:"tokens")
        var target=LossTargetMapping(componentID:"loss_cross_entropy",inputColumns:["input_ids"],roles:[.init(role:"target",columns:["target_ids"],sampleShape:[3],dtype:.int64)])
        target.schemaVersion=2
        let mapping:[String:JSONValue]=["namedInputPlan":.object(plan.values),"lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(target))]
        for i in d.nodes.indices {
            if d.nodes[i].address == .init(domain:.model,localID:"input") {d.nodes[i].payload = .modelPrimitive(type:"index_input",configurationVersion:2,config:["shape":.array([3])])}
            if d.nodes[i].address == .init(domain:.model,localID:"head"),case .modelPrimitive(let type,let version,var config)=d.nodes[i].payload {config["in_features"]=3;config["out_features"]=4;d.nodes[i].payload = .modelPrimitive(type:type,configurationVersion:version,config:config)}
            if d.nodes[i].address==WorkflowExamples.trainAddress,case .train(var config)=d.nodes[i].payload {
                config["objective"]=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]).recipeValue
                config["dataMapping"] = .object(mapping)
                var settings=try WorkflowTrainRunSetup.configuration(config,schemaVersion:d.schemaVersion) ?? [:]
                settings["epochs"]=5;settings["batchSize"]=7
                config=try WorkflowTrainRunSetup.setting(settings,in:config,schemaVersion:d.schemaVersion)
                d.nodes[i].payload = .train(config:config);d.nodes[i].name="Learn next-token transitions"
            }
        }
        let names:[String:String]=["observations":"Short text examples","cast":"Text and partition label","lookup":"Token embeddings","head":"Next-token logits","output":"Token predictions","model":"Bigram architecture"]
        for i in d.nodes.indices {if let name=names[d.nodes[i].address.localID] {d.nodes[i].name=name}}
        d.name="Train next-token sequences and inspect perplexity"
        try WorkflowCodec.validate(d);return d
    }
}
