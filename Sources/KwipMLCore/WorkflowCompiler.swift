import Foundation

public struct WorkflowModelValidationIssue: Equatable {
    public let address: WorkflowNodeAddress
    public let message: String
    public init(address: WorkflowNodeAddress, message: String) { self.address = address; self.message = message }
}

public struct WorkflowCompilerError:Error,LocalizedError,Equatable {
    public enum Code:String {case invalidDocument,invalidSelection,cycle,requiredInput,invalidPort,portKind,configuration,unsupportedOperator,lineage,unresolvedDraft,mixedModelVersions,invalidModel}
    public let code:Code
    public let message:String
    public let node:WorkflowNodeAddress?
    public let port:String?
    public var modelIssues: [WorkflowModelValidationIssue] = []
    public var errorDescription:String? {
        let location=node.map {"\($0.domain.rawValue):\($0.localID)"+(port.map{".\($0)"} ?? "")+": "} ?? ""
        return location+message
    }
}
func workflowFailure(_ code:WorkflowCompilerError.Code,_ message:String,_ node:WorkflowNodeAddress?=nil,_ port:String?=nil)->WorkflowCompilerError {
    .init(code:code,message:message,node:node,port:port)
}
public struct WorkflowResolvedPort:Equatable {
    public let id:String
    public let kind:WorkflowEdgeKind
    public let required:Bool
    public let maximumConnections:Int?
}
public struct WorkflowNodePorts:Equatable {
    public let inputs:[WorkflowResolvedPort]
    public let outputs:[WorkflowResolvedPort]
}
/// Pure, immutable phase requests. No files, grants, artifact paths or source bytes are read here.
public struct WorkflowExecutionPlan:Equatable {
    public let projectID:String
    public let revision:UInt64
    public let train:WorkflowNodeAddress
    public let dataflow:DataFlowDocument
    public let sourceAddresses:[WorkflowNodeAddress]
    public let model:ModelExecutionSnapshot
    public let training:DataFlowEndpoint
    public let validation:DataFlowEndpoint
    public let featureColumns:[String]
    public let targetColumn:String?
    public let settings:ExecutionSettings
    public let preparationCachePolicy:WorkflowPreparationCachePolicy
    public let trainingMode:WorkflowTrainingMode
    public var datasetRecipe:DatasetRecipe? = nil
    public var loaderRecipe:DataLoaderRecipe? = nil
    /// Recipes derived at the Train terminal (inline controls or schema15 layer policies).
    /// Legacy recipe nodes retain their separate origin unless combined with layer defaults.
    public var inlineDataLoadingRoles:Set<String> = []
    public var inlineTrainingRecipeRoles:Set<String> = []
    public var namedInputPlan:NamedInputPlan? = nil
    public var bagInput:BagInputMapping? = nil
    public var optimizerRecipe:OptimizerRecipe? = nil
    public var schedulerRecipe:SchedulerRecipe? = nil
    public var lossRecipe:LossRecipe? = nil
    public var initializationRecipe:InitializationRecipe? = nil
    public var gradientPolicyRecipe:GradientPolicyRecipe? = nil
    public var trainabilityRecipe:TrainabilityRecipe? = nil
    public var objectiveScalarization:WorkflowObjectiveScalarization = .mean
}
public enum WorkflowCompiler {
    public static func compile(_ document:WorkflowDocument,train:WorkflowNodeAddress,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,contractVersion:Int=2)throws->WorkflowExecutionPlan {
        try RetiredTrainingRecipeMigration.requireCurrent(document)
        try RetiredDataOperatorMigration.requireCurrent(document)
        return try compilePlan(document,train:train,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:contractVersion,historicalEvidence:false)
    }
    /// Read-only reconstruction for captured-plan integrity checks. Never use this
    /// compatibility entry point to authorize a new run or Code export.
    public static func compileHistoricalEvidence(_ document:WorkflowDocument,train:WorkflowNodeAddress,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,contractVersion:Int=2)throws->WorkflowExecutionPlan {
        try compilePlan(document,train:train,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:contractVersion,historicalEvidence:true)
    }
    private static func compilePlan(_ document:WorkflowDocument,train:WorkflowNodeAddress,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,contractVersion:Int,historicalEvidence:Bool)throws->WorkflowExecutionPlan {
        guard [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16].contains(contractVersion) else {throw workflowFailure(.configuration,"Unsupported workflow compiler contract version.",train)}
        guard document.schemaVersion<17 || contractVersion>=15 else {throw workflowFailure(.configuration,"Trained model artifact edges require compiler contract15.",train)}
        guard document.schemaVersion<16 || contractVersion>=14 else {throw workflowFailure(.configuration,"Train-owned run settings require compiler contract14.",train)}
        guard document.schemaVersion<15 || contractVersion>=13 else {throw workflowFailure(.configuration,"Layer parameter settings require compiler contract13.",train)}
        guard document.schemaVersion<13 || contractVersion>=12 else {throw workflowFailure(.configuration,"Data reader documents require compiler contract12.",train)}
        guard document.schemaVersion<12 || contractVersion>=11 else {throw workflowFailure(.configuration,"Bag mapping documents require compiler contract11.",train)}
        guard document.schemaVersion<11 || contractVersion>=10 else {throw workflowFailure(.configuration,"Native composition documents require compiler contract10.",train)}
        guard document.schemaVersion<10 || contractVersion>=9 else {throw workflowFailure(.configuration,"Gradient/trainability documents require compiler contract9.",train)}
        guard document.schemaVersion<9 || contractVersion>=8 else {throw workflowFailure(.configuration,"Initialization documents require compiler contract8.",train)}
        guard document.schemaVersion<8 || contractVersion>=7 else {throw workflowFailure(.configuration,"Objective documents require compiler contract7.",train)}
        guard document.schemaVersion<7 || contractVersion>=6 else {throw workflowFailure(.configuration,"Scheduler documents require compiler contract6.",train)}
        guard document.schemaVersion<6 || contractVersion>=5 else {throw workflowFailure(.configuration,"Optimizer documents require compiler contract5.",train)}
        guard document.schemaVersion < 4 || contractVersion >= 4 else {throw workflowFailure(.configuration,"Experiment documents require compiler contract 4.",train)}
        do {try WorkflowCodec.validate(document)} catch {throw workflowFailure(.invalidDocument,error.localizedDescription)}
        if document.schemaVersion>=16 {
            if let retired=document.nodes.first(where:WorkflowTrainRunSetup.isLegacyNode) {
                throw workflowFailure(.configuration,"Run settings belong to Train. A standalone Run setup must be explicitly repaired before this workflow can run.",retired.address)
            }
            guard !document.edges.contains(where:{$0.kind == .runConfiguration}) else {
                throw workflowFailure(.configuration,"Train-owned run settings cannot retain configuration connections.",train)
            }
        }
        guard document.drafts.isEmpty else {throw workflowFailure(.unresolvedDraft,"Apply or discard unresolved Block drafts before compiling.")}
        guard let terminal=document.nodes.first(where: {$0.address == train}),case .train(let terminalConfig)=terminal.payload else {throw workflowFailure(.invalidSelection,"Select a Train terminal.",train)}
        let trainingMode = try WorkflowControlContracts.trainingMode(terminalConfig, schemaVersion: document.schemaVersion)
        let inlineLoading = try WorkflowTrainDataLoading(configuration:terminalConfig,schemaVersion:document.schemaVersion)
        let inlineRecipes = try WorkflowTrainRecipes(configuration:terminalConfig,schemaVersion:document.schemaVersion)
        guard contractVersion >= 3 || trainingMode == .fresh else {
            throw workflowFailure(.configuration, "Checkpoint continuation execution is not available in this compiler version. The saved training mode was preserved; it will not be run as fresh training.", train)
        }
        let region=try dependencies(document,terminal:train,modelRegistry:modelRegistry,dataRegistry:dataRegistry,historicalEvidence:historicalEvidence)
        let nodes=region.nodes,incoming=region.incoming,selected=region.selected
        func dependency(_ address:WorkflowNodeAddress,_ port:String)throws->WorkflowEndpoint {
            guard let edge=incoming[address]?.first(where:{$0.to.port==port}) else {throw workflowFailure(.requiredInput,"Missing required input.",address,port)}
            return edge.from
        }
        let modelEndpoint=try dependency(train,"model")
        guard let modelCarrier=nodes[modelEndpoint.address],WorkflowModelBoundary.isDefinition(modelCarrier) else {throw workflowFailure(.invalidSelection,"Train requires a Model definition.",train)}
        let mapping:WorkflowNode,mappingConfig:[String:JSONValue]
        if document.schemaVersion>=18 {
            mapping=terminal
            mappingConfig=try WorkflowTrainMapping.configuration(terminalConfig,schemaVersion:document.schemaVersion) ?? [:]
            guard !document.nodes.contains(where:{if case .trainingData=$0.payload {return true};return false}),!document.edges.contains(where:{$0.kind == .trainingData}) else {throw workflowFailure(.configuration,"Mapping belongs to Train; repair remaining legacy mapping authority.",train)}
        } else {
            let endpoint=try dependency(train,"data")
            guard let legacy=nodes[endpoint.address],case .trainingData(let config)=legacy.payload else {throw workflowFailure(.invalidSelection,"Legacy Train requires its Training data mapping.",train)}
            mapping=legacy;mappingConfig=config
        }
        let setup=try WorkflowTrainRunSetup.resolve(in:document,train:train)
        let setupConfig=setup.configuration
        var settings=try WorkflowControlContracts.runSettings(setupConfig,address:setup.address)
        let optimizerRecipe:OptimizerRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port=="optimizer"}) {
            guard inlineRecipes.optimizer == nil else {throw workflowFailure(.configuration,"Optimizer has both inline and connected authority.",train,"optimizer")}
            guard let recipe=nodes[edge.from.address],case .optimizerRecipe(let id,let version,let config)=recipe.payload,version==1 else {throw workflowFailure(.configuration,"Connect a supported optimizer recipe to Train.optimizer.",train,"optimizer")}
            let configuration=try OptimizerConfiguration(componentID:id,values:config)
            optimizerRecipe = .init(configuration:configuration,source:recipe.address)
        } else if let config=inlineRecipes.optimizer {optimizerRecipe = .init(configuration:config,source:train)} else {optimizerRecipe=nil}
        if let optimizerRecipe {settings.learningRate=optimizerRecipe.configuration.learningRate}
        let schedulerRecipe:SchedulerRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port=="scheduler"}) {
            guard inlineRecipes.scheduler == nil else {throw workflowFailure(.configuration,"Scheduler has both inline and connected authority.",train,"scheduler")}
            guard let recipe=nodes[edge.from.address],case .schedulerRecipe(let id,let version,let config)=recipe.payload,version==1 else {throw workflowFailure(.configuration,"Connect a supported scheduler recipe to Train.scheduler.",train,"scheduler")}
            schedulerRecipe = .init(configuration:try SchedulerConfiguration(componentID:id,values:config),source:recipe.address)
        } else if let config=inlineRecipes.scheduler {schedulerRecipe = .init(configuration:config,source:train)} else {schedulerRecipe=nil}
        if let schedulerRecipe {
            guard let optimizerRecipe else {throw workflowFailure(.configuration,"Choose an explicit optimizer before choosing a scheduler.",train,"optimizer")}
            do {try schedulerRecipe.configuration.validateOptimizer(optimizerRecipe.configuration)} catch {throw workflowFailure(.configuration,error.localizedDescription,schedulerRecipe.source,schedulerRecipe.source == train ? nil:"out")}
        }
        let preparationCachePolicy=try WorkflowControlContracts.preparationCachePolicy(setupConfig,schemaVersion:document.schemaVersion)
        let lossRecipe:LossRecipe?
        let columns:[String]
        let target:String?
        let objectiveScalarization=try WorkflowControlContracts.objectiveScalarization(terminalConfig,schemaVersion:document.schemaVersion)
        let bagInput:BagInputMapping?
        if mappingConfig["bagInput"] != nil {bagInput=try WorkflowControlContracts.bagInputMapping(mappingConfig,address:mapping.address)} else {bagInput=nil}
        let namedInputPlan:NamedInputPlan?
        if case .object(let values)=mappingConfig["namedInputPlan"] {namedInputPlan=try NamedInputPlan(values:values)} else {namedInputPlan=nil}
        let baseMapping=mappingConfig.filter {$0.key != "namedInputPlan"}
        let selectedObjective:(configuration:LossConfiguration,source:WorkflowNodeAddress)?
        if let edge=incoming[train]?.first(where:{$0.to.port == "objective"}) {
            guard inlineRecipes.objective == nil else {throw workflowFailure(.configuration,"Objective has both inline and connected authority.",train,"objective")}
            guard let recipe=nodes[edge.from.address],case .lossRecipe(let id,let version,let config)=recipe.payload,version==1 else {throw workflowFailure(.configuration,"Connect a supported objective recipe to Train.objective.",train,"objective")}
            selectedObjective=(try LossConfiguration(componentID:id,values:config),recipe.address)
        } else if let config=inlineRecipes.objective {selectedObjective=(config,train)} else {selectedObjective=nil}
        if let objective=selectedObjective {
            let targetMapping=try WorkflowControlContracts.lossTargetMapping(bagInput == nil ? baseMapping : baseMapping.filter {$0.key != "bagInput"},address:mapping.address)
            do {lossRecipe=try LossRecipe(configuration:objective.configuration,source:objective.source,targetMapping:targetMapping)}
            catch {throw workflowFailure(.configuration,error.localizedDescription,objective.source,objective.source == train ? nil:"out")}
            columns=targetMapping.inputColumns;target=nil
        } else {
            guard mappingConfig["lossTargetMapping"] == nil else {throw workflowFailure(.configuration,"Connect the objective matching this typed target mapping.",train,"objective")}
            if let bagInput {
                guard case .string(let value)=mappingConfig["targetColumn"] else {throw workflowFailure(.configuration,"Choose a binary target column.",mapping.address)}
                columns=bagInput.indicesColumns;target=value;lossRecipe=nil
            } else {
                let legacy=try WorkflowControlContracts.trainingMapping(baseMapping,address:mapping.address)
                columns=legacy.0;target=legacy.1;lossRecipe=nil
            }
        }
        let training=try dependency(mapping.address,"training"),validation=try dependency(mapping.address,"validation")
        let evidence=try WorkflowDataLineage.resolve(region,dataRegistry:dataRegistry)
        func rows(_ endpoint:WorkflowEndpoint,consumer:WorkflowNodeAddress)throws->WorkflowDataLineage.Rows {
            guard case .rows(let value)=evidence[endpoint] else {throw workflowFailure(.lineage,"Input does not carry known table partition lineage.",consumer)}
            return value
        }
        let trainRows=try rows(training,consumer:mapping.address),validationRows=try rows(validation,consumer:mapping.address)
        if trainRows.sequenceWindows != nil || validationRows.sequenceWindows != nil {
            guard let plan=namedInputPlan,plan.sequenceObjective != nil,let recipe=lossRecipe,recipe.targetMapping.schemaVersion==2,training.address==validation.address,training.port=="training",validation.port=="validation",training.address==trainRows.sequenceWindows,validation.address==validationRows.sequenceWindows else {throw workflowFailure(.configuration,"Choose Next-token objective mapping and connect matching Training/Validation directly from Sequence windows.",mapping.address)}
            guard case .dataOperator(_,let windows)=nodes[training.address]?.payload else {throw workflowFailure(.configuration,"Sequence configuration is unavailable.",mapping.address)}
            try SequenceTrainingContract.validate(plan:plan,recipe:recipe,windows:windows)
        } else if namedInputPlan?.sequenceObjective != nil {throw workflowFailure(.configuration,"Next-token training requires matching Sequence windows outputs.",mapping.address)}
        guard trainRows.role=="training",validationRows.role=="validation",trainRows.source==validationRows.source,trainRows.split != nil,trainRows.split==validationRows.split,trainRows.states==validationRows.states else {throw workflowFailure(.lineage,"Training and validation require matching source/split, distinct roles and the same fitted-state chain.",mapping.address)}
        let model=try modelSnapshot(document,definition:modelCarrier.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,settings:settings,context:lossRecipe == nil ? .preparedWorkflow:.explicitObjective)
        if !historicalEvidence, let names=model.graph.namedInputs, !names.isEmpty, namedInputPlan == nil {
            let message="Model inputs (" + names.keys.sorted().joined(separator:", ") + ") require named tensor mapping. Open Train → Data mapping → Use named tensor mapping and map each declared input."
            var failure=workflowFailure(.configuration,message,mapping.address)
            failure.modelIssues=[.init(address:modelCarrier.address,message:message),.init(address:mapping.address,message:message)]
            throw failure
        }
        var initializationRecipe:InitializationRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port == "initialization"}) {
            guard inlineRecipes.initialization == nil else {throw workflowFailure(.configuration,"Initialization has both inline and connected authority.",train,"initialization")}
            guard let recipe=nodes[edge.from.address],case .initializationRecipe(let id,let version,let config)=recipe.payload,version==1 else {throw workflowFailure(.configuration,"Connect supported starting settings to Train.initialization.",train,"initialization")}
            initializationRecipe = .init(configuration:try InitializationConfiguration(componentID:id,values:config),source:recipe.address)
        } else if let config=inlineRecipes.initialization {initializationRecipe = .init(configuration:config,source:train)} else {initializationRecipe=nil}
        if let initializationRecipe {
            let selectedModelIDs=Set(model.graph.nodes.map(\.id))
            for rule in initializationRecipe.configuration.rules {
                guard selectedModelIDs.contains(rule.owner.node),model.origins[rule.owner.node] != nil else {throw workflowFailure(.configuration,"Selected starting-settings owner is not in this captured expanded model. Choose a current model owner.",initializationRecipe.source,initializationRecipe.source == train ? nil:"out")}
            }
        }
        let gradientPolicyRecipe:GradientPolicyRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port == "gradientPolicy"}) {
            guard inlineRecipes.gradientPolicy == nil else {throw workflowFailure(.configuration,"Gradient policy has both inline and connected authority.",train,"gradientPolicy")}
            guard let recipe=nodes[edge.from.address],case .gradientPolicyRecipe(let id,let version,let config)=recipe.payload,version==1 else {throw workflowFailure(.configuration,"Connect a supported gradient policy to Train.gradientPolicy.",train,"gradientPolicy")}
            gradientPolicyRecipe = .init(configuration:try GradientPolicyConfiguration(componentID:id,values:config),source:recipe.address)
        } else if let config=inlineRecipes.gradientPolicy {gradientPolicyRecipe = .init(configuration:config,source:train)} else {gradientPolicyRecipe=nil}
        try gradientPolicyRecipe?.configuration.validateObjectiveCompatibility(reduction:lossRecipe?.configuration.values["reduction"]?.stringValue ?? "mean",scalarization:objectiveScalarization.rawValue)
        var trainabilityRecipe:TrainabilityRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port == "trainability"}) {
            guard inlineRecipes.trainability == nil else {throw workflowFailure(.configuration,"Trainability has both inline and connected authority.",train,"trainability")}
            guard let recipe=nodes[edge.from.address],case .trainabilityRecipe(let id,let version,let config)=recipe.payload,version==1 else {throw workflowFailure(.configuration,"Connect supported trainability controls to Train.trainability.",train,"trainability")}
            trainabilityRecipe = .init(configuration:try TrainabilityConfiguration(componentID:id,values:config),source:recipe.address)
        } else if let config=inlineRecipes.trainability {trainabilityRecipe = .init(configuration:config,source:train)} else {trainabilityRecipe=nil}
        let layerRecipes=try LayerParameterSettings.recipes(document:document,model:model,registry:modelRegistry,seed:settings.seed,initialization:initializationRecipe,trainability:trainabilityRecipe,train:train)
        initializationRecipe=layerRecipes.0;trainabilityRecipe=layerRecipes.1
        let datasetRecipe:DatasetRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port == "dataset"}) {
            guard inlineLoading.datasetAccess == nil else {throw workflowFailure(.configuration,"Choose either Train dataset access or a connected Dataset view; both cannot control the same role.",train,"dataset")}
            guard let recipe=nodes[edge.from.address],case .datasetRecipe(let id,let version,let config)=recipe.payload,id==DatasetConfiguration.componentID,version==1 else {throw workflowFailure(.configuration,"Connect a supported dataset view to Train.dataset.",train,"dataset")}
            datasetRecipe = .init(configuration:try DatasetConfiguration(values:config),source:recipe.address)
            try datasetRecipe?.validate()
        } else if let configuration=inlineLoading.datasetAccess {
            datasetRecipe = .init(configuration:configuration,source:train)
            try datasetRecipe?.validate()
        } else {datasetRecipe=nil}
        let loaderRecipe:DataLoaderRecipe?
        if let edge=incoming[train]?.first(where:{$0.to.port == "loader"}) {
            guard inlineLoading.batching == nil else {throw workflowFailure(.configuration,"Choose either Train batching or a connected DataLoader; both cannot control the same role.",train,"loader")}
            guard let recipe=nodes[edge.from.address],case .batchRecipe(let id,let version,let config)=recipe.payload,id==DataLoaderConfiguration.componentID,version==1 else {throw workflowFailure(.configuration,"Connect supported batching controls to Train.loader.",train,"loader")}
            loaderRecipe = .init(configuration:try DataLoaderConfiguration(values:config),source:recipe.address)
            try loaderRecipe?.validate()
        } else if let configuration=inlineLoading.batching {
            loaderRecipe = .init(configuration:configuration,source:train)
            try loaderRecipe?.validate()
        } else {loaderRecipe=nil}
        if let loaderRecipe {settings.batchSize=loaderRecipe.configuration.batchSize}
        if datasetRecipe?.configuration.kind == "boundedIterable",loaderRecipe?.configuration.shuffle == true {
            throw workflowFailure(.configuration,"A bounded stream preserves its source order. Turn off Shuffle or choose an indexed dataset.",loaderRecipe?.source,"out")
        }
        if bagInput != nil,datasetRecipe?.configuration.kind == "tensor" {
            throw workflowFailure(.configuration,"A tensor tuple requires aligned dense observations. Choose an indexed, subset or bounded stream view for whole bags.",datasetRecipe?.source,"out")
        }
        try GradientWorkflowRecipeContext.validate(policy:gradientPolicyRecipe?.configuration,trainability:trainabilityRecipe?.configuration,model:model,registry:modelRegistry,context:lossRecipe == nil ? .preparedWorkflow:.explicitObjective)
        if let bagInput {
            do {
                let requirement=try WorkflowMappingRequirements.bagInputRequirement(model.graph,registry:modelRegistry,context:lossRecipe == nil ? .preparedWorkflow:.explicitObjective)
                try bagInput.validate(sourceConfiguration:requirement.configuration,targetColumn:target,objectiveMapping:lossRecipe?.targetMapping)
            } catch {throw workflowFailure(.configuration,error.localizedDescription,mapping.address)}
        } else if let namedInputPlan {
            let validation=GraphValidator.validate(model.graph,registry:modelRegistry,context:lossRecipe == nil ? .preparedWorkflow:.explicitObjective)
            guard validation.isValid else {throw workflowFailure(.configuration,"Repair the model before mapping named inputs.",mapping.address)}
            if let names=model.graph.namedInputs {guard names==Dictionary(uniqueKeysWithValues:namedInputPlan.inputs.map {($0.name,$0.nodeID!)}) else {throw workflowFailure(.configuration,"Mapped names differ from model inputs.",mapping.address)}}
            else {guard namedInputPlan.inputs.count==1 else {throw workflowFailure(.configuration,"Declare multiple named inputs in Model configuration first.",mapping.address)}}
            for input in namedInputPlan.inputs {
                guard let id=input.nodeID,let actual=validation.tensorContracts[id],actual.dtype.rawValue==input.dtype,actual.layout.sample.dimensions==input.sampleShape else {throw workflowFailure(.configuration,"Mapped input shape/type differs from its model boundary.",mapping.address)}
            }
        } else {
            let contract=try WorkflowMappingRequirements.inputContract(model.graph,registry:modelRegistry,context:lossRecipe == nil ? .preparedWorkflow:.explicitObjective)
            let required=contract.layout.sample.elementCount
            guard columns.count==required else {
                throw workflowFailure(.configuration,"Model input shape \(contract.layout.sample.dimensions) requires \(required) ordered columns; the mapping has \(columns.count). Map one column per sample element in row-major order.",mapping.address)
            }
        }
        if let lossRecipe {
            let output=try WorkflowMappingRequirements.outputContract(model.graph,registry:modelRegistry)
            try WorkflowMappingRequirements.validateObjective(prediction:output,recipe:lossRecipe)
            if let plan=namedInputPlan,let sequence=plan.sequenceObjective,let input=plan.inputs.first(where:{$0.name==sequence.inputName}) {
                guard output.layout.sample.dimensions==[input.sampleShape[0],input.vocabularySize!] else {throw workflowFailure(.configuration,"Next-token logits must match [context, vocabulary] exactly.",train)}
            }
        }
        // Compiler16 captures inspection branches as part of this training run.
        // Earlier recorded contracts keep their original selected preparation graph.
        var dataSelected=selected
        let predictionBranches=WorkflowPredictionPreparation.dependentData(in:document)
        for sink in document.nodes {
            guard !predictionBranches.contains(sink.address),case .dataOperator(let operation,_)=sink.payload,
                  operation == "kwip.data.outputDelimited" || (contractVersion>=16 && operation == "kwip.data.view") else {continue}
            let branch=try dependencies(document,terminal:sink.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,historicalEvidence:historicalEvidence)
            guard branch.selected.allSatisfy({$0.domain == .data}),!branch.selected.intersection(selected).isEmpty else {
                if operation == "kwip.data.view" {continue}
                throw workflowFailure(.invalidSelection,"Connect this output to the training preparation flow, or run it in a separate data-only canvas. Unrelated outputs cannot be silently skipped.",sink.address)
            }
            dataSelected.formUnion(branch.selected)
        }
        let dataNodes=document.nodes.filter {dataSelected.contains($0.address) && $0.address.domain == .data}
        let dataflow=DataFlowDocument(nodes:dataNodes.compactMap {node in if case .dataOperator(let id,let config)=node.payload{return .init(id:node.address.localID,operatorID:id,config:config)};return nil},edges:document.edges.filter {dataSelected.contains($0.to.address) && dataSelected.contains($0.from.address) && $0.from.address.domain == .data && $0.to.address.domain == .data}.map {.init(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))})
        do {try DataFlowCodec.validate(dataflow)} catch {throw workflowFailure(.invalidDocument,error.localizedDescription)}
        return .init(projectID:document.projectID,revision:document.revision,train:train,dataflow:dataflow,sourceAddresses:dataNodes.filter(DataFlowSourceOperators.isSource).map(\.address),model:model,training:.init(node:training.address.localID,port:training.port),validation:.init(node:validation.address.localID,port:validation.port),featureColumns:columns,targetColumn:target,settings:settings,preparationCachePolicy:preparationCachePolicy,trainingMode:trainingMode,datasetRecipe:datasetRecipe,loaderRecipe:loaderRecipe,inlineDataLoadingRoles:Set([(inlineLoading.datasetAccess != nil ? "dataset":nil),(inlineLoading.batching != nil ? "loader":nil)].compactMap {$0}),inlineTrainingRecipeRoles:inlineRecipes.roles.union(layerRecipes.2),namedInputPlan:namedInputPlan,bagInput:bagInput,optimizerRecipe:optimizerRecipe,schedulerRecipe:schedulerRecipe,lossRecipe:lossRecipe,initializationRecipe:initializationRecipe,gradientPolicyRecipe:gradientPolicyRecipe,trainabilityRecipe:trainabilityRecipe,objectiveScalarization:objectiveScalarization)
    }

    /// Compile an explicitly selected architecture without requiring data bindings,
    /// training configuration or a Train terminal. This is the same model path used
    /// by full execution; unrelated unfinished regions remain authoring drafts.
    public static func modelSnapshot(_ document:WorkflowDocument,definition:WorkflowNodeAddress,
                                     modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,
                                     settings:ExecutionSettings? = nil,context:GraphValidationContext = .preparedWorkflow)throws->ModelExecutionSnapshot {
        do {try WorkflowCodec.validate(document)} catch {throw workflowFailure(.invalidDocument,error.localizedDescription)}
        guard document.drafts.isEmpty else {throw workflowFailure(.unresolvedDraft,"Apply or discard unresolved Block drafts before compiling.")}
        guard let carrier=document.nodes.first(where: {$0.address == definition}),WorkflowModelBoundary.isDefinition(carrier) else {
            throw workflowFailure(.invalidSelection,"Select a Model definition component.",definition)
        }
        let region=try dependencies(document,terminal:definition,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        let nodes=region.nodes,incoming=region.incoming,selected=region.selected
        _ = try WorkflowModelBoundary.output(in:document,definition:definition)
        let modelNodes=document.nodes.filter {selected.contains($0.address) && $0.address.domain == .model}
        let versions=Set(modelNodes.compactMap {node->Int? in if case .modelPrimitive(_,let version,_)=node.payload{return version};return nil})
        guard versions.count==1,let version=versions.first else {throw workflowFailure(.mixedModelVersions,"Migrate the model region explicitly before combining configuration versions.",definition)}
        let primitives=modelNodes.compactMap {node->GraphNode? in
            guard case .modelPrimitive(let type,_,let config)=node.payload else{return nil}
            var primitive=GraphNode(id:node.address.localID,type:type,name:node.name,x:node.x,y:node.y,config:config)
            // The convenience initializer supplies current defaults. Compilation preserves
            // historical schema1 configuration exactly; migration is a separate transaction.
            primitive.config=config
            return primitive
        }
        let instances=modelNodes.compactMap {node->CompositeInstance? in
            guard case .modelBlock(let reference,let overrides)=node.payload else{return nil}
            return .init(id:node.address.localID,definition:reference,name:node.name,x:node.x,y:node.y,overrides:overrides)
        }
        let modelEdges=document.edges.filter {selected.contains($0.to.address) && $0.from.address.domain == .model && $0.to.address.domain == .model}.map {GraphEdge(id:$0.id,source:$0.from.address.localID,sourcePort:$0.from.port,target:$0.to.address.localID,targetPort:$0.to.port)}
        let model:ModelExecutionSnapshot
        do {
            if instances.isEmpty {
                var graph=GraphDocument(schemaVersion:version,name:document.name,nodes:primitives,edges:modelEdges,execution:settings)
                graph.namedInputs=document.namedModelInputs?[definition.localID]
                model=try ModelExecutionSnapshotBuilder.build(.legacyFlat(graph),registry:modelRegistry,context:context)
            } else {
                // Authoring ownership must not change legacy native parameter structure.
                // Explicit native container policies still require their source format.
                let retained=document.modelSourceVersions?[definition.localID] ?? (document.schemaVersion>=11 ? 3 : 2)
                var required=Set(instances.map(\.definition)),pending=Array(required)
                while let reference=pending.popLast() {
                    guard let definition=document.definitions.first(where:{$0.reference==reference}) else {continue}
                    for child in definition.body.instances where required.insert(child.definition).inserted {pending.append(child.definition)}
                }
                let needsNative=document.definitions.contains {required.contains($0.reference) && $0.composition != nil}
                let sourceVersion=needsNative ? 3 : retained
                // An unrelated native definition cannot promote a retained legacy model.
                // A source2 snapshot must exclude that unrelated native-only collection.
                let definitions=LayerParameterSettings.stripped(sourceVersion==2 && document.definitions.contains(where:{$0.composition != nil}) ? document.definitions.filter {required.contains($0.reference)} : document.definitions)
                let source=CompositeProjectDocument(schemaVersion:sourceVersion,name:document.name,root:.init(nodes:primitives,instances:instances,edges:modelEdges,configurationVersion:version),definitions:definitions,execution:settings,namedInputs:document.namedModelInputs?[definition.localID])
                model=try ModelExecutionSnapshotBuilder.build(.blockModel(.init(projectID:document.projectID,revision:document.revision,source:source,schemaVersion:sourceVersion==3 ? 2 : 1)),registry:modelRegistry,context:context)
            }
        } catch {
            if let failure = error as? CompositeContractError, !failure.validationIssues.isEmpty {
                let issues = failure.validationIssues.map { issue -> WorkflowModelValidationIssue in
                    let origin = issue.nodeID.flatMap { failure.nodeOrigins[$0] }
                    let localID = origin?.instancePath.first ?? origin?.localNodeID ?? issue.nodeID
                    let address = localID.map { WorkflowNodeAddress(domain: .model, localID: $0) } ?? definition
                    return .init(address: address, message: issue.message)
                }
                var attributed = workflowFailure(.invalidModel, failure.localizedDescription, issues.first?.address ?? definition)
                attributed.modelIssues = issues
                throw attributed
            }
            throw workflowFailure(.invalidModel,error.localizedDescription,definition)
        }
        return model
    }

    struct DependencyRegion {
        let nodes:[WorkflowNodeAddress:WorkflowNode]
        let incoming:[WorkflowNodeAddress:[WorkflowEdge]]
        let selected:Set<WorkflowNodeAddress>
        let order:[WorkflowNodeAddress]
    }
    static func dependencies(_ document:WorkflowDocument,terminal:WorkflowNodeAddress,
                                     modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,historicalEvidence:Bool=false,artifactSources:Set<WorkflowNodeAddress>=[])throws->DependencyRegion {
        let nodes=Dictionary(uniqueKeysWithValues:document.nodes.map {($0.address,$0)})
        let incoming=Dictionary(grouping:document.edges.filter {!artifactSources.contains($0.to.address)},by:{$0.to.address})
        var visiting=Set<WorkflowNodeAddress>(),selected=Set<WorkflowNodeAddress>(),order:[WorkflowNodeAddress]=[]
        func visit(_ address:WorkflowNodeAddress)throws {
            if visiting.contains(address) {throw workflowFailure(.cycle,"Dependency cycle prevents execution.",address)}
            if selected.contains(address) {return}
            guard nodes[address] != nil else {throw workflowFailure(.invalidPort,"Dependency refers to a missing node.",address)}
            visiting.insert(address)
            for edge in incoming[address] ?? [] {try visit(edge.from.address)}
            visiting.remove(address);selected.insert(address);order.append(address)
        }
        try visit(terminal)
        var contracts:[WorkflowNodeAddress:WorkflowNodePorts]=[:]
        for address in order {
            let ports=try WorkflowPortContracts.resolvePlanPorts(nodes[address]!,document:document,modelRegistry:modelRegistry,dataRegistry:dataRegistry,historicalEvidence:historicalEvidence)
            contracts[address]=artifactSources.contains(address) ? .init(inputs:[],outputs:ports.outputs):ports
        }
        let selectedOutgoing=Dictionary(grouping:document.edges.filter{selected.contains($0.to.address) && !artifactSources.contains($0.to.address)},by:{$0.from.address})
        for address in order {
            let ports=contracts[address]!,edges=incoming[address] ?? []
            for output in ports.outputs {
                let count=(selectedOutgoing[address] ?? []).filter{$0.from.port==output.id}.count
                guard output.maximumConnections.map({count<=$0}) ?? true else {
                    throw workflowFailure(.invalidPort,"This output exceeds its declared connection limit.",address,output.id)
                }
            }
            for input in ports.inputs {
                let count=edges.filter {$0.to.port==input.id}.count
                guard (!input.required || count>0),input.maximumConnections.map({count<=$0}) ?? true else {throw workflowFailure(.requiredInput,"Connect required input exactly once.",address,input.id)}
            }
            for edge in edges {
                guard let source=contracts[edge.from.address]?.outputs.first(where:{$0.id==edge.from.port}),let target=ports.inputs.first(where:{$0.id==edge.to.port}) else {throw workflowFailure(.invalidPort,"Unknown source or destination port.",address,edge.to.port)}
                let tableLike:Set<WorkflowEdgeKind>=[.table,.dataset]
                // Dataset is a table with separately retained role evidence, never a fitted state.
                guard source.kind==edge.kind,(source.kind==target.kind || (tableLike.contains(source.kind) && tableLike.contains(target.kind))) else {throw workflowFailure(.portKind,"Declared edge kind or input payload kind is incompatible.",address,edge.to.port)}
            }
        }
        return .init(nodes:nodes,incoming:incoming,selected:selected,order:order)
    }

}

/// Bounded subset used by the existing worker-owned registry. Unknown schema features fail closed.
enum WorkflowSchema {
    static func check(_ value:JSONValue,schema:JSONValue,depth:Int=0)throws {
        func fail(_ reason:String)throws {throw DataFlowContractError("Invalid operator configuration: "+reason)}
        guard depth<=32,case .object(let s)=schema else {try fail("unsupported schema structure");return}
        let keys:Set<String>=["type","properties","required","additionalProperties","enum","const","minimum","maximum","minLength","maxLength","minItems","maxItems","uniqueItems","items","minProperties","maxProperties","oneOf","not","default"]
        // Discovery annotations are inert; keep the assertion allowlist closed.
        let annotations:Set<String>=["description","examples","x-functionSignatures","x-regexOptions"]
        guard Set(s.keys).isSubset(of:keys.union(annotations)) else {try fail("unsupported schema keyword");return}
        if let value=s["description"],case .string=value {} else if s["description"] != nil {try fail("description annotation must be text")}
        if let value=s["examples"],case .array=value {} else if s["examples"] != nil {try fail("examples annotation must be an array")}
        if let value=s["x-functionSignatures"] {
            guard case .object(let signatures)=value,signatures.values.allSatisfy({if case .string=$0{return true};return false}) else {try fail("function signatures annotation must map names to text");return}
        }
        if let value=s["x-regexOptions"] {
            guard case .object(let options)=value else {try fail("regex options annotation must be an object");return}
            let limits:Set<String>=["patternCharactersMaximum","replacementCharactersMaximum","textResultCharactersMaximum"]
            guard Set(options.keys).isSubset(of:limits.union(["flags"])) else {try fail("unknown regex options annotation");return}
            for (key,value) in options {
                if key=="flags" {guard case .string=value else {try fail("regex flags annotation must be text");return}}
                else {guard case .integer(let count)=value,count>=0 else {try fail("regex limit annotation must be a nonnegative integer");return}}
            }
        }
        if let expected=s["const"],value != expected {try fail("constant differs")}
        if case .array(let choices)=s["enum"],!choices.contains(value) {try fail("unsupported choice")}
        func matches(_ type:String)->Bool {
            switch (type,value) {case ("object",.object),("array",.array),("string",.string),("boolean",.bool),("null",.null),("integer",.integer),("number",.integer),("number",.number):return true;default:return false}
        }
        if let type=s["type"] {
            let types:[String]
            if case .string(let name)=type {types=[name]} else if case .array(let names)=type {types=names.compactMap {if case .string(let name)=$0{return name};return nil}} else {try fail("unsupported type declaration");return}
            guard types.contains(where:matches) else {try fail("value type differs");return}
        }
        func count(_ value:Int,min:String,max:String)throws {
            if let minimum=s[min]?.doubleValue,Double(value)<minimum {try fail("below \(min)")}
            if let maximum=s[max]?.doubleValue,Double(value)>maximum {try fail("above \(max)")}
        }
        switch value {
        case .object(let object):
            try count(object.count,min:"minProperties",max:"maxProperties")
            if case .array(let required)=s["required"] {
                for key in required {guard case .string(let name)=key,object[name] != nil else {try fail("missing required field");return}}
            }
            let properties:[String:JSONValue]
            if case .object(let p)=s["properties"] {properties=p} else {properties=[:]}
            for (key,child) in object {
                if let field=properties[key] {try check(child,schema:field,depth:depth+1)}
                else if s["additionalProperties"] == .bool(false) {try fail("unknown field \(key)")}
                else if let extra=s["additionalProperties"],extra != .bool(true) {try check(child,schema:extra,depth:depth+1)}
            }
        case .array(let items):
            try count(items.count,min:"minItems",max:"maxItems")
            if s["uniqueItems"] == .bool(true),Set(items).count != items.count {try fail("duplicate array items")}
            if let itemSchema=s["items"] {for item in items {try check(item,schema:itemSchema,depth:depth+1)}}
        case .string(let text):try count(text.unicodeScalars.count,min:"minLength",max:"maxLength")
        case .integer,.number:
            guard let number=value.doubleValue,number.isFinite else {try fail("nonfinite number");return}
            if let minimum=s["minimum"]?.doubleValue,number<minimum {try fail("number below minimum")}
            if let maximum=s["maximum"]?.doubleValue,number>maximum {try fail("number above maximum")}
        default:break
        }
        if case .array(let alternatives)=s["oneOf"] {
            let count=alternatives.filter {(try? check(value,schema:$0,depth:depth+1)) != nil}.count
            if count != 1 {try fail("conditional fields do not match exactly one schema branch")}
        }
        if let forbidden=s["not"],(try? check(value,schema:forbidden,depth:depth+1)) != nil {try fail("forbidden field combination")}
    }
}
