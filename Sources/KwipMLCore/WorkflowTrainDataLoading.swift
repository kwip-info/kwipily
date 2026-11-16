import Foundation

/// Optional Train-owned views over verified prepared data. Absence preserves the
/// legacy Run Setup execution path; reading this value never persists defaults.
public struct WorkflowTrainDataLoading:Equatable {
    public let datasetAccess:DatasetConfiguration?
    public let batching:DataLoaderConfiguration?
    public init(configuration:[String:JSONValue],schemaVersion:Int)throws {
        guard (1...24).contains(schemaVersion) else {throw DataFlowContractError("Unsupported workflow data-loading schema.")}
        guard configuration["datasetAccess"] == nil && configuration["batching"] == nil || schemaVersion>=13 else {
            throw DataFlowContractError("Train data access and batching require workflow schema13 or later.")
        }
        if let raw=configuration["datasetAccess"] {
            guard case .object(let values)=raw else {throw DataFlowContractError("Dataset access needs the exact Dataset configuration object; omit it to use existing defaults.")}
            datasetAccess=try DatasetConfiguration(values:values)
        } else {datasetAccess=nil}
        if let raw=configuration["batching"] {
            guard case .object(let values)=raw else {throw DataFlowContractError("Batching needs the exact DataLoader configuration object; omit it to use Run Setup.")}
            batching=try DataLoaderConfiguration(values:values)
        } else {batching=nil}
    }
    @discardableResult public static func validate(_ configuration:[String:JSONValue],schemaVersion:Int)throws->Self {
        try .init(configuration:configuration,schemaVersion:schemaVersion)
    }
}

public extension WorkflowTrainDataLoading {
    /// Ordinary authoring-open migration only. Historical decoding and compilation
    /// remain unchanged. The caller owns recovery, revision increment and persistence.
    /// Invalid/orphan/ambiguous recipes fail atomically, preserving the original graph.
    static func migrateLegacyRecipes(_ document:WorkflowDocument)throws->WorkflowDocument {
        try WorkflowCodec.validate(document)
        let recipes=document.nodes.filter {node in
            switch node.payload {case .datasetRecipe,.batchRecipe,.optimizerRecipe,.schedulerRecipe,.lossRecipe,.initializationRecipe,.gradientPolicyRecipe,.trainabilityRecipe:return true;default:return false}
        }
        let addresses=Set(recipes.map(\.address))
        let recipeKinds:Set<WorkflowEdgeKind>=[.datasetRecipe,.batchRecipe,.optimizerRecipe,.schedulerRecipe,.objectiveRecipe,.initializationRecipe,.gradientPolicyRecipe,.trainabilityRecipe]
        guard !recipes.isEmpty else {
            guard !document.edges.contains(where:{recipeKinds.contains($0.kind)}) else {
                throw DataFlowContractError("A connected training edge has no supported legacy recipe. The original project was preserved for recovery.")
            }
            return document
        }
        guard !document.edges.contains(where:{addresses.contains($0.to.address)}) else {
            throw DataFlowContractError("A legacy training recipe has unsupported incoming connections. The original project was preserved for recovery.")
        }
        guard document.edges.filter({recipeKinds.contains($0.kind)}).allSatisfy({addresses.contains($0.from.address)}) else {
            throw DataFlowContractError("A connected training edge has an unsupported source. The original project was preserved for recovery.")
        }
        var updates:[WorkflowNodeAddress:[String:JSONValue]]=[:]
        for node in recipes {
            let role:String,key:String,kind:WorkflowEdgeKind,value:JSONValue
            switch node.payload {
            case .datasetRecipe(let type,let version,let config):
                guard type == DatasetConfiguration.componentID,version == 1 else {throw DataFlowContractError("Unsupported legacy Dataset recipe; original project preserved.")}
                _ = try DatasetConfiguration(values:config)
                role="dataset";key="datasetAccess";kind = .datasetRecipe;value = .object(config)
            case .batchRecipe(let type,let version,let config):
                guard type == DataLoaderConfiguration.componentID,version == 1 else {throw DataFlowContractError("Unsupported legacy DataLoader recipe; original project preserved.")}
                _ = try DataLoaderConfiguration(values:config)
                role="loader";key="batching";kind = .batchRecipe;value = .object(config)
            case .optimizerRecipe(let type,let version,let config):
                guard version == 1 else {throw DataFlowContractError("Unsupported optimizer version.")}
                role="optimizer";key=role;kind = .optimizerRecipe;value=try OptimizerConfiguration(componentID:type,values:config).recipeValue
            case .schedulerRecipe(let type,let version,let config):
                guard version == 1 else {throw DataFlowContractError("Unsupported scheduler version.")}
                role="scheduler";key=role;kind = .schedulerRecipe;value=try SchedulerConfiguration(componentID:type,values:config).recipeValue
            case .lossRecipe(let type,let version,let config):
                guard version == 1 else {throw DataFlowContractError("Unsupported objective version.")}
                role="objective";key=role;kind = .objectiveRecipe;value=try LossConfiguration(componentID:type,values:config).recipeValue
            case .initializationRecipe(let type,let version,let config):
                guard version == 1 else {throw DataFlowContractError("Unsupported initialization version.")}
                role="initialization";key=role;kind = .initializationRecipe;value=try InitializationConfiguration(componentID:type,values:config).recipeValue
            case .gradientPolicyRecipe(let type,let version,let config):
                guard version == 1 else {throw DataFlowContractError("Unsupported gradient policy version.")}
                role="gradientPolicy";key=role;kind = .gradientPolicyRecipe;value=try GradientPolicyConfiguration(componentID:type,values:config).recipeValue
            case .trainabilityRecipe(let type,let version,let config):
                guard version == 1 else {throw DataFlowContractError("Unsupported trainability version.")}
                role="trainability";key=role;kind = .trainabilityRecipe;value=try TrainabilityConfiguration(componentID:type,values:config).recipeValue
            default:throw DataFlowContractError("Unsupported training recipe.")
            }
            let outgoing=document.edges.filter {$0.from.address == node.address}
            guard !outgoing.isEmpty else {
                throw DataFlowContractError("Unconnected legacy training recipe \(node.name.isEmpty ? node.address.localID:node.name) has settings with no Train destination. Its source is unchanged. Automatic migration cannot choose where those settings belong; manual project repair is required before running or exporting code.")
            }
            for edge in outgoing {
                guard edge.from.port == "out",edge.to.port == role,edge.kind == kind,
                      let target=document.nodes.first(where:{$0.address == edge.to.address}),case .train(let original)=target.payload else {
                    throw DataFlowContractError("A legacy training recipe is not connected to its supported Train role. The original project was preserved for recovery.")
                }
                var config=updates[target.address] ?? original
                guard config[key] == nil else {
                    throw DataFlowContractError("Train \(target.name.isEmpty ? target.address.localID:target.name) has both inline and connected \(role) settings, or multiple legacy sources. Resolve the competing authorities before running or exporting code; no configuration was dropped.")
                }
                config[key] = value
                updates[target.address]=config
            }
        }
        var migrated=try WorkflowTrainRecipes.migrateAuthoringSchema(document)
        migrated.nodes.removeAll {addresses.contains($0.address)}
        migrated.edges.removeAll {addresses.contains($0.from.address)}
        for index in migrated.nodes.indices {
            if let config=updates[migrated.nodes[index].address] {migrated.nodes[index].payload = .train(config:config)}
        }
        try WorkflowCodec.validate(migrated)
        return migrated
    }
}
