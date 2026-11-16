import Foundation

/// Train-owned run controls. Raw values remain source; defaults are resolved only for execution.
public enum WorkflowTrainRunSetup {
    public struct Authority:Equatable {
        public let address:WorkflowNodeAddress
        public let configuration:[String:JSONValue]
    }
    public static func configuration(_ trainConfiguration:[String:JSONValue],schemaVersion:Int)throws->[String:JSONValue]? {
        guard (1...24).contains(schemaVersion) else {throw DataFlowContractError("Unsupported Train run settings schema.")}
        guard schemaVersion>=16 else {
            guard trainConfiguration["runSetup"] == nil else {throw DataFlowContractError("Train-owned run settings require workflow schema16.")}
            return nil
        }
        guard case .object(let value)=trainConfiguration["runSetup"] else {throw DataFlowContractError("Train requires its runSetup settings object; an empty object uses the validated defaults.")}
        if let capture=value["gradientCapture"],capture != .null {
            guard case .object(let fields)=capture,Set(fields.keys).isSubset(of:["schemaVersion","everyEpochs","maxBatchesPerEpoch","maxCaptures"]) else {throw DataFlowContractError("Gradient capture contains unknown or malformed controls.")}
        }
        _ = try WorkflowControlContracts.runSettings(value,address:.init(domain:.control,localID:"train"))
        return value
    }
    public static func setting(_ runSetup:[String:JSONValue],in trainConfiguration:[String:JSONValue],schemaVersion:Int=16)throws->[String:JSONValue] {
        guard (16...22).contains(schemaVersion) else {throw DataFlowContractError("Train-owned run settings require workflow schema16.")}
        var result=trainConfiguration;result["runSetup"] = .object(runSetup)
        _ = try configuration(result,schemaVersion:schemaVersion)
        return result
    }
    public static func resolve(in document:WorkflowDocument,train:WorkflowNodeAddress)throws->Authority {
        guard let node=document.nodes.first(where:{$0.address==train}),case .train(let config)=node.payload else {throw DataFlowContractError("Choose an existing Train for run settings.")}
        let links=document.edges.filter {$0.to.address==train && $0.to.port=="configuration"}
        if document.schemaVersion>=16 {
            guard links.isEmpty else {throw workflowFailure(.configuration,"Train-owned settings cannot also have a connected Run setup.",train)}
            do {
                guard let inline=try configuration(config,schemaVersion:document.schemaVersion) else {throw DataFlowContractError("Train requires its run settings.")}
                return .init(address:train,configuration:inline)
            } catch {throw workflowFailure(.configuration,error.localizedDescription,train)}
        }
        _ = try configuration(config,schemaVersion:document.schemaVersion)
        guard links.count==1,let link=links.first,link.kind == .runConfiguration,link.from.port=="configuration",
              let setup=document.nodes.first(where:{$0.address==link.from.address}),case .runSetup(let values)=setup.payload else {throw DataFlowContractError("The legacy Train requires exactly one connected Run setup; repair its settings authority before upgrading.")}
        _ = try WorkflowControlContracts.runSettings(values,address:setup.address)
        return .init(address:setup.address,configuration:values)
    }
    public static func isLegacyNode(_ node:WorkflowNode)->Bool {if case .runSetup=node.payload {return true};return false}
    public static func requiresUpgrade(_ document:WorkflowDocument)->Bool {
        document.nodes.contains(where:isLegacyNode) || document.edges.contains {$0.kind == .runConfiguration} || document.nodes.contains {node in
            if case .train(let config)=node.payload,config["selfPlay"] == nil {return document.schemaVersion<16 || config["runSetup"] == nil};return false
        }
    }
    /// Explicit editing promotion when adding a new inline Train to a data/model-only document.
    public static func migrateAuthoringSchema(_ document:WorkflowDocument)throws->WorkflowDocument {
        var result=try migrate(document)
        if result.schemaVersion<14 {WorkflowModelInput.migrateOwnershipMetadata(&result)}
        result.schemaVersion=max(16,result.schemaVersion)
        try WorkflowCodec.validate(result)
        return result
    }
    /// Pure authoring upgrade; callers preserve original bytes and publish one atomic revision.
    public static func migrate(_ document:WorkflowDocument)throws->WorkflowDocument {
        try WorkflowCodec.validate(document)
        guard requiresUpgrade(document) else {return document}
        let setups=document.nodes.filter(isLegacyNode),addresses=Set(setups.map(\.address))
        guard !document.edges.contains(where:{addresses.contains($0.to.address)}) else {throw DataFlowContractError("Run setup has unsupported incoming connections. Original settings require explicit repair.")}
        for setup in setups {
            let outgoing=document.edges.filter {$0.from.address==setup.address}
            guard !outgoing.isEmpty else {throw DataFlowContractError("An unused Run setup contains settings with no Train destination. Original source is unchanged; choose its intended Train during repair.")}
            guard outgoing.allSatisfy({edge in
                guard edge.kind == .runConfiguration,edge.from.port=="configuration",edge.to.port=="configuration",
                      let target=document.nodes.first(where:{$0.address==edge.to.address}),case .train=target.payload else {return false}
                return true
            }) else {throw DataFlowContractError("Run setup has an unsupported consumer. Original settings require explicit repair.")}
        }
        guard document.edges.filter({$0.kind == .runConfiguration}).allSatisfy({addresses.contains($0.from.address)}) else {throw DataFlowContractError("A run configuration connection has an unsupported source. Original source is unchanged.")}
        var configurations:[WorkflowNodeAddress:[String:JSONValue]]=[:]
        for node in document.nodes {
            guard case .train(let config)=node.payload,config["selfPlay"] == nil else {continue}
            let authority=try resolve(in:document,train:node.address)
            configurations[node.address]=try setting(authority.configuration,in:config)
        }
        var result=document
        if result.schemaVersion<14 {WorkflowModelInput.migrateOwnershipMetadata(&result)}
        result.schemaVersion=max(16,result.schemaVersion)
        result.nodes.removeAll {addresses.contains($0.address)}
        result.edges.removeAll {addresses.contains($0.from.address)}
        for index in result.nodes.indices {if let config=configurations[result.nodes[index].address] {result.nodes[index].payload = .train(config:config)}}
        try WorkflowCodec.validate(result)
        return result
    }
}
