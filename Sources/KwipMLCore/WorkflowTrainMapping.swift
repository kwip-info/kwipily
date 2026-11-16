import Foundation

/// Train-owned table/tensor mapping. Authoring promotion never changes captured History.
public enum WorkflowTrainMapping {
    public static func configuration(_ config:[String:JSONValue],schemaVersion:Int)throws->[String:JSONValue]? {
        guard schemaVersion>=18 else {
            guard config["dataMapping"] == nil else {throw DataFlowContractError("Train-owned mapping requires workflow schema18.")}
            return nil
        }
        guard case .object(let mapping)=config["dataMapping"] else {throw DataFlowContractError("Train requires a dataMapping object; configure input and target columns before running.")}
        guard mapping["namedInputPlan"] == nil || schemaVersion>=19 else {throw DataFlowContractError("Named tensor mappings require workflow schema19.")}
        try validate(mapping)
        return mapping
    }
    public static func validate(_ mapping:[String:JSONValue])throws {
        if let raw=mapping["namedInputPlan"] {
            guard case .object(let values)=raw,mapping["bagInput"] == nil else {throw DataFlowContractError("Named inputs require an object and cannot use bag mapping.")}
            _=try NamedInputPlan(values:values)
            try validate(mapping.filter {$0.key != "namedInputPlan"})
            return
        }
        if mapping["bagInput"] != nil {_ = try WorkflowControlContracts.bagInputMapping(mapping,address:.init(domain:.control,localID:"train"))}
        else if mapping["lossTargetMapping"] != nil {_ = try WorkflowControlContracts.lossTargetMapping(mapping,address:.init(domain:.control,localID:"train"))}
        else {
            guard Set(mapping.keys).isSubset(of:["featureColumns","targetColumn"]) else {throw DataFlowContractError("Unknown Train data-mapping controls.")}
            if let value=mapping["featureColumns"] {guard case .array(let columns)=value,columns.allSatisfy({$0.stringValue != nil}) else {throw DataFlowContractError("Input columns must be an ordered list of names.")}}
            if let value=mapping["targetColumn"] {guard value.stringValue != nil else {throw DataFlowContractError("Target column must be a name.")}}
        }
    }
    public static func requiresUpgrade(_ document:WorkflowDocument)->Bool {
        document.nodes.contains {if case .trainingData=$0.payload {return true};if case .train=$0.payload {return document.schemaVersion<18};return false}
    }
    /// Exact pure transformation; caller backs up original bytes before atomic persistence.
    public static func migrate(_ document:WorkflowDocument)throws->WorkflowDocument {
        try WorkflowCodec.validate(document)
        guard requiresUpgrade(document) else {return document}
        // Resume older guided construction using its captured stage sequence.
        if let progress=document.guidedBuild,progress.version<5 {return document}
        let mappings=document.nodes.filter {if case .trainingData=$0.payload {return true};return false}
        let addresses=Set(mappings.map(\.address))
        var result=document,updates:[WorkflowNodeAddress:[String:JSONValue]]=[:],newEdges:[WorkflowEdge]=[]
        for mapping in mappings {
            guard case .trainingData(let config)=mapping.payload else {continue}
            try validate(config)
            let inputs=document.edges.filter {$0.to.address==mapping.address}
            guard inputs.allSatisfy({["training","validation"].contains($0.to.port) && $0.kind == .table && $0.from.address.domain == .data}),Set(inputs.map(\.to.port)).count==inputs.count else {throw DataFlowContractError("Training mapping has unsupported or duplicate source connections. Repair the original mapping before upgrading.")}
            let links=document.edges.filter {$0.from.address==mapping.address}
            guard !links.isEmpty else {throw DataFlowContractError("An unused Training data mapping has settings with no Train destination. Original source is unchanged; choose its Train before upgrading.")}
            for link in links {
                guard link.kind == .trainingData,link.from.port=="data",link.to.port=="data",let train=document.nodes.first(where:{$0.address==link.to.address}),case .train(let trainConfig)=train.payload,
                      trainConfig["dataMapping"] == nil,updates[train.address] == nil else {throw DataFlowContractError("Training mapping has an unsupported or conflicting consumer. Repair its authority before upgrading.")}
                var next=trainConfig;next["dataMapping"] = .object(config);updates[train.address]=next
                for input in inputs {
                    guard !document.edges.contains(where:{$0.to.address==train.address && $0.to.port==input.to.port}) else {throw DataFlowContractError("Train already has a connected table authority. Original mapping is unchanged.")}
                    newEdges.append(.init(id:link.id+"-"+input.to.port,from:input.from,to:.init(address:train.address,port:input.to.port),kind:.table))
                }
            }
        }
        guard document.edges.filter({$0.kind == .trainingData}).allSatisfy({addresses.contains($0.from.address)}) else {throw DataFlowContractError("Unsupported training-data connection; original source unchanged.")}
        for i in result.nodes.indices {
            guard case .train(let config)=result.nodes[i].payload,config["selfPlay"] == nil else {continue}
            var next=updates[result.nodes[i].address] ?? config
            if next["dataMapping"] == nil {next["dataMapping"] = .object([:])}
            result.nodes[i].payload = .train(config:next)
        }
        result.nodes.removeAll {addresses.contains($0.address)}
        result.edges.removeAll {addresses.contains($0.from.address) || addresses.contains($0.to.address)}
        result.edges += newEdges
        if result.schemaVersion<14 {WorkflowModelInput.migrateOwnershipMetadata(&result)}
        result.schemaVersion=max(18,result.schemaVersion)
        try WorkflowCodec.validate(result)
        return result
    }
}
