import Foundation

public struct WorkflowMappingRequirement:Equatable {
    public let train:WorkflowNodeAddress
    public let label:String
    public let contract:TypedTensorContract?
    public let problem:String?
    public var bagRequirement:BagWorkflowInputRequirement? = nil
}

/// Authoring guidance only: model contracts can be inspected before data and
/// run setup exist. This never certifies the columns' actual values or schema.
public enum WorkflowMappingRequirements {
    public static func bagInputRequirement(_ graph:GraphDocument,registry:ComponentRegistry,context:GraphValidationContext = .preparedWorkflow)throws->BagWorkflowInputRequirement {
        let validation=GraphValidator.validate(graph,registry:registry,context:context)
        let sources=graph.nodes.filter {["input","index_input","bag_input"].contains($0.type)}
        guard validation.isValid,sources.count==1,let source=sources.first,source.type == "bag_input" else {
            throw DataFlowContractError("Bag input requirements are unavailable. "+validation.issues.map(\.message).joined(separator:" "))
        }
        let configuration=try BagSourceConfiguration(values:GraphMigration.effectiveConfiguration(of:source,schemaVersion:graph.schemaVersion,registry:registry))
        let consumers=graph.nodes.filter {$0.type == "embedding_bag"}
        guard !consumers.isEmpty else {throw DataFlowContractError("Connect the bag source to an EmbeddingBag consumer.")}
        var vocabulary=Int.max,width=0
        for consumer in consumers {
            let incoming=graph.edges.filter {$0.target == consumer.id}
            let native=try EmbeddingBagConfiguration(values:GraphMigration.effectiveConfiguration(of:consumer,schemaVersion:graph.schemaVersion,registry:registry))
            guard Set(incoming.map(\.targetPort))==Set(configuration.outputPorts),incoming.allSatisfy({ edge in
                guard let contract=validation.outputBagContracts[.init(edge.source,edge.sourcePort)] else {return false}
                return contract.sourceNodeID==source.id && contract.configuration==configuration && contract.role.rawValue==edge.targetPort
            }) else {throw DataFlowContractError("All bag consumers require coherent enabled roles from the same actual source.")}
            vocabulary=min(vocabulary,native.embedding.numberOfEmbeddings)
            width=max(width,native.embedding.embeddingDimension)
        }
        return .init(sourceNodeID:source.id,configuration:configuration,numEmbeddings:vocabulary,embeddingDim:width)
    }
    public static func inputContract(_ graph:GraphDocument,registry:ComponentRegistry,context:GraphValidationContext = .preparedWorkflow)throws->TypedTensorContract {
        let validation=GraphValidator.validate(graph,registry:registry,context:context)
        guard validation.isValid,
              let source=graph.nodes.first(where:{["input","index_input"].contains($0.type)}),
              let contract=validation.tensorContracts[source.id] else {
            throw DataFlowContractError("Model input shape is unavailable. "+validation.issues.map(\.message).joined(separator:" "))
        }
        return contract
    }
    public static func outputContract(_ graph:GraphDocument,registry:ComponentRegistry)throws->TypedTensorContract {
        let validation=GraphValidator.validate(graph,registry:registry,context:.explicitObjective)
        guard validation.isValid,let output=graph.nodes.first(where:{$0.type == "output"}),let contract=validation.tensorContracts[output.id] else {throw DataFlowContractError("Objective prediction shape is unavailable. "+validation.issues.map(\.message).joined(separator:" "))}
        return contract
    }
    /// Structural admission only; source values and distributions are checked by execution.
    public static func validateObjective(prediction:TypedTensorContract,recipe:LossRecipe)throws {
        let p=prediction.layout.sample.dimensions,c=recipe.configuration.values,id=recipe.configuration.componentID
        guard prediction.dtype == .float32,(1...7).contains(p.count),p.allSatisfy({(1...4096).contains($0)}) else {throw DataFlowContractError("Objectives require bounded observation-first float32 predictions.")}
        try recipe.configuration.validate(targetMapping:recipe.targetMapping)
        guard recipe.targetMapping.roles.allSatisfy({$0.sampleShape.count<=7}) else {throw DataFlowContractError("Objective target roles support at most seven sample directions; observations remain separate.")}
        let roles=Dictionary(uniqueKeysWithValues:recipe.targetMapping.roles.map {($0.role,$0)})
        func require(_ role:String,_ dtype:ContractDType,_ shape:[Int]? = nil)throws {
            guard let value=roles[role],value.dtype == dtype,shape.map({$0 == value.sampleShape}) ?? true else {throw DataFlowContractError("Target role \(role) needs the selected prediction's exact dtype and sample shape.")}
        }
        func broadcast(_ first:[Int],_ second:[Int])throws->[Int] {
            let rank=max(first.count,second.count),a=Array(repeating:1,count:rank-first.count)+first,b=Array(repeating:1,count:rank-second.count)+second
            guard zip(a,b).allSatisfy({$0.0 == $0.1 || $0.0 == 1 || $0.1 == 1}) else {throw DataFlowContractError("Target/weight directions cannot broadcast; observations remain independent.")}
            let output=zip(a,b).map {max($0.0,$0.1)}
            var count=1
            for size in output {guard size>0,count<=100_000_000/size else {throw DataFlowContractError("Objective broadcast exceeds the 100M-element limit.")};count *= size}
            return output
        }
        switch id {
        case "loss_cross_entropy","loss_nll":
            guard let axis=c["class_axis"]?.integerValue,(-p.count-1...p.count).contains(axis) else {throw DataFlowContractError("Choose a class direction within the prediction array.")}
            let rank=p.count+1,full=(axis%rank+rank)%rank
            guard full != 0 else {throw DataFlowContractError("The observation direction cannot be the class direction.")}
            let classes=p[full-1]
            guard (2...512).contains(classes) else {throw DataFlowContractError("The class direction requires 2–512 classes.")}
            if case .array(let weights)=c["class_weights"],!weights.isEmpty,weights.count != classes {throw DataFlowContractError("Class importance must have one value per actual class.")}
            let probability=id == "loss_cross_entropy" && c["target_mode"] == .string("probabilities")
            var target=p;target.remove(at:full-1)
            try require("target",probability ? .float32:.int64,probability ? p:target)
        case "loss_bce_with_logits":
            try require("target",.float32,p)
            for name in ["weight","pos_weight"] {
                let mode=c[name+"_mode"]?.stringValue ?? "none"
                if mode == "none" {continue}
                let shape:[Int]
                if mode == "mapped" {try require(name,.float32);shape=roles[name]!.sampleShape}
                else {guard case .array(let values)=c[name+"_shape"] else {throw DataFlowContractError("Declare configured importance shape.")};shape=values.compactMap(\.integerValue)}
                guard shape.count<=p.count,try broadcast(p,shape)==p else {throw DataFlowContractError("Importance must broadcast into existing sample directions, never observations.")}
            }
        case "loss_mse","loss_l1":
            try require("target",.float32)
            let shape=roles["target"]!.sampleShape
            if c["target_broadcast"] == .bool(true) {_ = try broadcast(p,shape)}
            else {guard shape == p else {throw DataFlowContractError("Match target directions exactly or explicitly enable sample broadcasting.")}}
        case "loss_kl_div":
            try require("target",.float32,p)
            guard (2...512).contains(p.last!) else {throw DataFlowContractError("Distribution comparison requires 2–512 final-direction probabilities.")}
        case "loss_ctc":
            guard p.count == 2,(2...512).contains(p[1]),let blank=c["blank"]?.integerValue,blank<p[1] else {throw DataFlowContractError("Sequence predictions require [time,classes] and an existing blank class.")}
            try require("target",.int64);try require("input_lengths",.int64,[]);try require("target_lengths",.int64,[])
            guard roles["target"]!.sampleShape.count == 1,p[0]<=100_000_000/(2*roles["target"]!.sampleShape[0]+1) else {throw DataFlowContractError("Sequence alignment target geometry exceeds its work limit.")}
        case "loss_triplet_margin":
            guard p.count == 2,p[0] == 3,roles.isEmpty else {throw DataFlowContractError("Triplet predictions need [anchor/positive/negative,width] without target labels.")}
        default:throw DataFlowContractError("Unknown objective prediction contract.")
        }
    }
    public static func resolve(_ document:WorkflowDocument,mapping:WorkflowNodeAddress,
                               modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)throws->[WorkflowMappingRequirement] {
        try WorkflowCodec.validate(document)
        guard let node=document.nodes.first(where:{$0.address==mapping}),({if case .trainingData=node.payload {return true};return false}() || {if case .train=node.payload {return document.schemaVersion>=18};return false}()) else {
            throw DataFlowContractError("Select Train’s Data mapping.")
        }
        let inline=document.schemaVersion>=18 && {if case .train=node.payload {return true};return false}()
        let consumers=inline ? Set([mapping]):Set(document.edges.filter {$0.from.address==mapping}.map { $0.to.address })
        return document.nodes.filter {consumers.contains($0.address)}.compactMap {consumer in
            guard case .train=consumer.payload else {return nil}
            let label=consumer.name.isEmpty ? "Train" : consumer.name
            do {
                let data=document.edges.filter {$0.to.address==consumer.address && $0.to.port=="data"}
                guard inline || (data.count==1 && data.first?.from.address==mapping && data.first?.from.port=="data" && data.first?.kind == .trainingData) else {throw DataFlowContractError("Connect this mapping to Train with one training-data connection.")}
                let models=document.edges.filter {$0.to.address==consumer.address && $0.to.port=="model"}
                guard models.count==1,let model=models.first,["model","definition"].contains(model.from.port),model.kind == .modelDefinition else {
                    throw DataFlowContractError("Connect one Model definition to the Train model input.")
                }
                let objectives=document.edges.filter {$0.to.address == consumer.address && $0.to.port == "objective" && $0.kind == .objectiveRecipe}
                guard objectives.count<=1 else {throw DataFlowContractError("Choose one objective authority for this Train.")}
                guard case .train(let trainConfig)=consumer.payload else {throw DataFlowContractError("Objective mapping requires Train.")}
                let inline=try WorkflowTrainRecipes(configuration:trainConfig,schemaVersion:document.schemaVersion).objective
                guard inline == nil || objectives.isEmpty else {throw DataFlowContractError("Objective has both inline and connected authority.")}
                let context:GraphValidationContext
                if let objective=objectives.first {
                    guard let source=document.nodes.first(where:{$0.address == objective.from.address}),case .lossRecipe(let id,1,let config)=source.payload,objective.from.port == "out" else {throw DataFlowContractError("Choose a supported objective recipe.")}
                    _ = try LossConfiguration(componentID:id,values:config);context = .explicitObjective
                } else {context = inline == nil ? .preparedWorkflow:.explicitObjective}
                let snapshot=try WorkflowCompiler.modelSnapshot(document,definition:model.from.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,context:context)
                if snapshot.graph.nodes.contains(where:{$0.type == "bag_input"}) {
                    return WorkflowMappingRequirement(train:consumer.address,label:label,contract:nil,problem:nil,
                        bagRequirement:try bagInputRequirement(snapshot.graph,registry:modelRegistry,context:context))
                }
                return WorkflowMappingRequirement(train:consumer.address,label:label,
                    contract:try inputContract(snapshot.graph,registry:modelRegistry,context:context),problem:nil)
            } catch {
                return WorkflowMappingRequirement(train:consumer.address,label:label,contract:nil,problem:error.localizedDescription)
            }
        }
    }
}
