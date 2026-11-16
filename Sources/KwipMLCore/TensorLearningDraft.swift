import Foundation

/// Source-only learning state. Grades and measured evidence belong to separate receipts.
public struct TensorLearningDraft:Codable,Equatable {
    public var schemaVersion=1
    public var contentVersion=ContractVersion()
    public var projectID:String
    public var operation:String
    public var mode:LearningMode
    public var graph=GraphDocument(name:"Tensor exercise")
    public var notes=""
    public var pendingFields:[String:String]=[:]
    public var canvasView:CanvasViewState?
    public var parameters:[String:[String:LearningTensor]]?
    public var gradientNodeID:String?
    public var programExportRecipe:JSONValue?
    public var checkpointConfiguration:CheckpointLearningConfiguration?
    public var compositionModel:ModelAuthoringDocument?
    public init(projectID:String,operation:String,mode:LearningMode) {self.projectID=projectID;self.operation=operation;self.mode=mode;if let task=ProgramExportLearningTask(rawValue:operation) {programExportRecipe=try! ProgramExportConfiguration(values:task.values).recipeValue};if let task=ModuleCompositionLearningTask(rawValue:operation) {compositionModel=try? ModuleCompositionLearningDraft.model(projectID:projectID,task:task)};if let task=GradientLearningTask(rawValue:operation) {graph=try! GradientLearningExamples.defaultGraph(task:task)};if let task=InitializationLearningTask(rawValue:operation) {graph=try! InitializationLearningExamples.defaultGraph(task:task)};if CheckpointLearningTask(rawValue:operation) != nil {checkpointConfiguration=try! .init()};if SchedulerLearningTask(rawValue:operation) != nil {graph.schemaVersion=7;graph.name="Scheduler exercise"};if OptimizerLearningTask(rawValue:operation) != nil {graph.schemaVersion=6;graph.name="Optimizer exercise"}}
    private func isInitializationEditor(key:String)->Bool {
        InitializationLearningTask(rawValue:operation) != nil && graph.nodes.contains {node in
            node.type == "initialization" && key == "\(node.id.utf8.count):\(node.id):initializationEditor"
        }
    }
    private func isGradientEditor(key:String)->Bool {
        GradientLearningTask(rawValue:operation) != nil && graph.nodes.contains {node in
            let field = node.type == "gradient_policy" ? "gradientPolicyEditor":node.type == "trainability" ? "trainabilityEditor":""
            return !field.isEmpty && key == "\(node.id.utf8.count):\(node.id):\(field)"
        }
    }
    private func pendingTextCount(key:String,value:String)->Int {
        (isInitializationEditor(key:key) || isGradientEditor(key:key)) ? value.utf8.count:value.count
    }
    private func pendingTextLimit(key:String)->Int {
        if isInitializationEditor(key:key) || isGradientEditor(key:key) {return 131072}
        if DataLoadingLearningTask(rawValue:operation) != nil {
            for node in graph.nodes where node.type == "dataset_source" {
                if key == "\(node.id.utf8.count):\(node.id):indices" {return 32768}
            }
        }
        guard LossLearningTask(rawValue:operation) != nil else {return 512}
        for node in graph.nodes where LossConfiguration.componentIDs.contains(node.type) {
            for field in BundledLossComponents.definition(node.type).fields where field.kind == .numberList {
                if key == "\(node.id.utf8.count):\(node.id):\(field.key)" {return 32768}
            }
        }
        return 512
    }
    public func validate()throws {
        guard schemaVersion == 1,contentVersion == ContractVersion(),UUID(uuidString:projectID) != nil,(TensorLearningTask.operations.contains(operation) || LayerNormLearningTask(rawValue:operation) != nil),notes.count<=20_000,pendingFields.count<=64,pendingFields.allSatisfy({!$0.key.isEmpty && $0.key.count<=256 && pendingTextLimit(key:$0.key)>=pendingTextCount(key:$0.key,value:$0.value)}) else {
            throw LearningContractError("Invalid or unsupported tensor lesson draft identity, version or text bounds.")
        }
        if let task=DataLoadingLearningTask(rawValue:operation) {
            _=try DataLoadingLearningSubmission(task:task,graph:graph,complete:false)
        } else if let task=ProgramExportLearningTask(rawValue:operation) {
            guard let programExportRecipe,graph.schemaVersion==2,graph.nodes.isEmpty,graph.edges.isEmpty,graph.execution==nil,graph.compositionPlan==nil else {throw LearningContractError("Program lessons configure an artifact action, not a model graph.")}
            _=try ProgramExportLearningSubmission(task:task,values:ProgramExportConfiguration.decodeRecipe(programExportRecipe).values)
        } else if let task=ModuleCompositionLearningTask(rawValue:operation) {
            guard let compositionModel,graph.schemaVersion==2,graph.nodes.isEmpty,graph.edges.isEmpty,graph.execution==nil,graph.compositionPlan==nil else {throw LearningContractError("Composition exercises edit only their supplied Block model source.")}
            try ModuleCompositionLearningDraft.validate(model:compositionModel,task:task,projectID:projectID,complete:false)
        } else if CheckpointLearningTask(rawValue:operation) != nil {
            guard let checkpointConfiguration,graph.nodes.isEmpty,graph.edges.isEmpty,graph.execution == nil else {throw LearningContractError("Checkpoint lessons use lifecycle controls and a supplied architecture, not an editable model graph.")}
            try checkpointConfiguration.validate()
        } else if let task=GradientLearningTask(rawValue:operation) {
            _ = try GradientLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=InitializationLearningTask(rawValue:operation) {
            _ = try InitializationLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=LossLearningTask(rawValue:operation) {
            _ = try LossLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=AttentionLearningTask(rawValue:operation) {
            _ = try AttentionLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=RecurrentLearningTask(rawValue:operation) {
            _ = try RecurrentLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=SchedulerLearningTask(rawValue:operation) {
            _ = try SchedulerLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=OptimizerLearningTask(rawValue:operation) {
            _ = try OptimizerLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=StatefulNormalizationLearningTask(rawValue:operation) {
            _ = try StatefulNormalizationLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=SpatialPatchLearningTask(rawValue:operation) {
            _ = try SpatialPatchLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=TensorRoutingLearningTask(rawValue:operation) {
            _ = try TensorRoutingLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=RemainingActivationLearningTask(rawValue:operation) {
            _ = try RemainingActivationLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=TensorCompositionLearningTask(rawValue:operation) {
            _ = try TensorCompositionLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=AdaptivePoolingLearningTask(rawValue:operation) {
            _ = try AdaptivePoolingLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=ConfigurableActivationLearningTask(rawValue:operation) {
            _ = try ConfigurableActivationLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=PoolingLearningTask(rawValue:operation) {
            _ = try PoolingLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if let task=ConvolutionLearningTask(rawValue:operation) {
            _ = try ConvolutionLearningSubmission.validatedGraph(graph,task:task,complete:false)
        } else if operation == LazyLinearAssessment.operation {
            _ = try LazyLinearLearningSubmission.validatedGraph(graph,complete:false)
        } else if BilinearLearningTask(rawValue:operation) != nil {
            _ = try BilinearLearningSubmission.validatedGraph(graph,complete:false)
        } else if EmbeddingBagLearningTask(rawValue:operation) != nil {
            _ = try EmbeddingBagLearningSubmission.validatedGraph(graph,complete:false)
        } else if MatrixLearningTask(rawValue:operation) != nil {
            _ = try MatrixLearningSubmission.validatedGraph(graph,complete:false)
        } else {
        _ = try TensorLearningSubmission.validatedGraph(graph,complete:false,allowSoftmax:TensorLearningTask.softmaxAxis(for:operation) != nil,allowLayerNorm:LayerNormLearningTask(rawValue:operation) != nil,allowEmbedding:EmbeddingLearningTask(rawValue:operation) != nil)
        }
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {
            try StatefulNormalizationLearningSubmission.validateParameters(graph:graph,parameters:parameters ?? [:],gradientNodeID:gradientNodeID,task:task,complete:false)
        } else if let task=RemainingActivationLearningTask(rawValue:operation) {
            try RemainingActivationLearningSubmission.validateParameters(graph:graph,parameters:parameters ?? [:],gradientNodeID:gradientNodeID,task:task,complete:false)
        } else if LayerNormLearningTask(rawValue:operation) != nil {
            let identities=Set(graph.nodes.filter {$0.type == "layer_norm"}.map(\.id))
            guard gradientNodeID == nil || identities.contains(gradientNodeID!),
                  (parameters?.count ?? 0)<=32,
                  (parameters ?? [:]).allSatisfy({ident,values in
                      identities.contains(ident) && Set(values.keys).isSubset(of:["weight","bias"]) && values.values.allSatisfy(\.wellFormed)
                  }) else {throw LearningContractError("Normalization draft parameters require existing nodes and bounded finite tensors.")}
        } else if parameters != nil || gradientNodeID != nil {
            throw LearningContractError("Parameter state is not supported by this tensor task.")
        }
        if ProgramExportLearningTask(rawValue:operation)==nil,programExportRecipe != nil {throw LearningContractError("Program controls belong only to program lessons.")}
        if ModuleCompositionLearningTask(rawValue:operation)==nil,compositionModel != nil {throw LearningContractError("Block model lesson source belongs only to composition exercises.")}
        if CheckpointLearningTask(rawValue:operation) == nil,checkpointConfiguration != nil {throw LearningContractError("Lifecycle controls belong only to checkpoint lessons.")}
        if let canvasView {try WorkspaceViewState(modelCanvas:canvasView).validate()}
    }
    public func submissionData(compositionRegistry:ComponentRegistry?=nil)throws->Data {
        try validate()
        guard pendingFields.isEmpty else {throw LearningContractError("Apply or discard every pending field before submitting.")}
        if let task=DataLoadingLearningTask(rawValue:operation) {return try DataLoadingLearningSubmission(task:task,graph:graph).encoded()}
        if let task=ProgramExportLearningTask(rawValue:operation),let programExportRecipe {return try ProgramExportLearningSubmission(task:task,values:ProgramExportConfiguration.decodeRecipe(programExportRecipe).values).encoded()}
        if let task=ModuleCompositionLearningTask(rawValue:operation) {
            guard let compositionModel,let compositionRegistry else {throw LearningContractError("Select the captured verified registry before checking this Block exercise.")}
            return try ModuleCompositionLearningDraft.submission(model:compositionModel,task:task,registry:compositionRegistry).encoded()
        }
        if let task=CheckpointLearningTask(rawValue:operation),let checkpointConfiguration {return try CheckpointLearningSubmission(configuration:checkpointConfiguration,task:task).encoded()}
        if let task=GradientLearningTask(rawValue:operation) {return try GradientLearningSubmission(graph:graph,task:task).encoded()}
        if let task=InitializationLearningTask(rawValue:operation) {return try InitializationLearningSubmission(graph:graph,task:task).encoded()}
        if LayerNormLearningTask(rawValue:operation) != nil {
            return try LayerNormLearningSubmission(graph:graph,parameters:parameters ?? [:],gradientNodeID:gradientNodeID).encoded()
        }
        if let task=LossLearningTask(rawValue:operation) {return try LossLearningSubmission(graph:graph,task:task).encoded()}
        if let task=AttentionLearningTask(rawValue:operation) {return try AttentionLearningSubmission(graph:graph,task:task).encoded()}
        if let task=RecurrentLearningTask(rawValue:operation) {return try RecurrentLearningSubmission(graph:graph,task:task).encoded()}
        if let task=SchedulerLearningTask(rawValue:operation) {return try SchedulerLearningSubmission(graph:graph,task:task).encoded()}
        if let task=OptimizerLearningTask(rawValue:operation) {return try OptimizerLearningSubmission(graph:graph,task:task).encoded()}
        if let task=StatefulNormalizationLearningTask(rawValue:operation) {return try StatefulNormalizationLearningSubmission.fromDraft(graph:graph,parameters:parameters ?? [:],gradientNodeID:gradientNodeID,task:task).encoded()}
        if let task=SpatialPatchLearningTask(rawValue:operation) {return try SpatialPatchLearningSubmission(graph:graph,task:task).encoded()}
        if let task=TensorRoutingLearningTask(rawValue:operation) {return try TensorRoutingLearningSubmission(graph:graph,task:task).encoded()}
        if let task=RemainingActivationLearningTask(rawValue:operation) {return try RemainingActivationLearningSubmission(graph:graph,parameters:parameters ?? [:],gradientNodeID:gradientNodeID,task:task).encoded()}
        if let task=TensorCompositionLearningTask(rawValue:operation) {return try TensorCompositionLearningSubmission(graph:graph,task:task).encoded()}
        if let task=AdaptivePoolingLearningTask(rawValue:operation) {return try AdaptivePoolingLearningSubmission(graph:graph,task:task).encoded()}
        if let task=ConfigurableActivationLearningTask(rawValue:operation) {return try ConfigurableActivationLearningSubmission(graph:graph,task:task).encoded()}
        if let task=PoolingLearningTask(rawValue:operation) {return try PoolingLearningSubmission(graph:graph,task:task).encoded()}
        if let task=ConvolutionLearningTask(rawValue:operation) {return try ConvolutionLearningSubmission(graph:graph,task:task).encoded()}
        if operation == LazyLinearAssessment.operation {return try LazyLinearLearningSubmission(graph:graph).encoded()}
        if BilinearLearningTask(rawValue:operation) != nil {return try BilinearLearningSubmission(graph:graph).encoded()}
        if MatrixLearningTask(rawValue:operation) != nil {return try MatrixLearningSubmission(graph:graph).encoded()}
        if EmbeddingLearningTask(rawValue:operation) != nil {return try EmbeddingLearningSubmission(graph:graph).encoded()}
        if EmbeddingBagLearningTask(rawValue:operation) != nil {return try EmbeddingBagLearningSubmission(graph:graph).encoded()}
        return try submission().encoded()
    }
    public func submission()throws->TensorLearningSubmission {
        guard DataLoadingLearningTask(rawValue:operation)==nil,ProgramExportLearningTask(rawValue:operation)==nil else {throw LearningContractError("Recipe and artifact lessons use their typed submission, not a tensor graph.")}
        guard ModuleCompositionLearningTask(rawValue:operation)==nil else {throw LearningContractError("Composition exercises require their authored Block source, not a flat graph.")}
        guard GradientLearningTask(rawValue:operation) == nil else {throw LearningContractError("Gradient lessons require their protected training-control submission, not a forward graph.")}
        guard CheckpointLearningTask(rawValue:operation) == nil else {throw LearningContractError("Checkpoint lessons require lifecycle controls, not a forward graph.")}
        guard LossLearningTask(rawValue:operation) == nil else {throw LearningContractError("Loss lessons require their source-qualified objective-check submission, not a model-forward graph.")}
        guard AttentionLearningTask(rawValue:operation) == nil else {throw LearningContractError("Attention tasks require their named-mask submission.")}
        guard RecurrentLearningTask(rawValue:operation) == nil else {throw LearningContractError("Recurrent tasks require their named-state submission.")}
        guard SchedulerLearningTask(rawValue:operation) == nil else {throw LearningContractError("Scheduler lessons require their two typed training-control recipes, not a model-forward graph.")}
        guard OptimizerLearningTask(rawValue:operation) == nil else {throw LearningContractError("Optimizer lessons require their typed training-recipe submission, not a model-forward graph.")}
        guard StatefulNormalizationLearningTask(rawValue:operation) == nil else {throw LearningContractError("Normalization tasks require their reviewed parameter/state-aware submission.")}
        guard SpatialPatchLearningTask(rawValue:operation) == nil else {throw LearningContractError("Patch tasks require their reviewed spatial-patch submission.")}
        guard TensorRoutingLearningTask(rawValue:operation) == nil else {throw LearningContractError("Routing tasks require their named-output submission.")}
        guard RemainingActivationLearningTask(rawValue:operation) == nil else {throw LearningContractError("These activation tasks require their state-aware reviewed submission.")}
        guard TensorCompositionLearningTask(rawValue:operation) == nil else {throw LearningContractError("Composition tasks require their ordered-source submission.")}
        guard AdaptivePoolingLearningTask(rawValue:operation) == nil else {throw LearningContractError("Averaging/adaptive tasks require their reviewed pooling submission.")}
        guard ConfigurableActivationLearningTask(rawValue:operation) == nil else {throw LearningContractError("Activation tasks require their reviewed activation submission.")}
        guard PoolingLearningTask(rawValue:operation) == nil else {throw LearningContractError("Pooling tasks require their indexed-window submission.")}
        guard ConvolutionLearningTask(rawValue:operation) == nil else {throw LearningContractError("Shared-window lessons require supplied-kernel submissions.")}
        guard operation != LazyLinearAssessment.operation else {throw LearningContractError("Inferred-width tasks require their state-aware submission.")}
        guard BilinearLearningTask(rawValue:operation) == nil else {throw LearningContractError("Bilinear tasks require their fixture-bound submission.")}
        guard MatrixLearningTask(rawValue:operation) == nil else {throw LearningContractError("Matrix tasks require their two-input submission.")}
        guard EmbeddingBagLearningTask(rawValue:operation) == nil else {throw LearningContractError("Bag lessons require their named-role embeddingBagGraph submission.")}
        guard EmbeddingLearningTask(rawValue:operation) == nil else {throw LearningContractError("Lookup requires its embedding submission.")}
        guard LayerNormLearningTask(rawValue:operation) == nil else {throw LearningContractError("Normalization requires its explicit parameter submission.") }
        try validate()
        guard pendingFields.isEmpty else {throw LearningContractError("Apply or discard every pending field before submitting.")}
        return try TensorLearningSubmission(graph:graph,allowSoftmax:TensorLearningTask.softmaxAxis(for:operation) != nil)
    }
    public func encoded()throws->Data {
        try validate()
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Tensor draft exceeds 2 MB.")}
        return data
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Tensor draft exceeds 2 MB.")}
        guard case .object(let raw)=try JSONDecoder().decode(JSONValue.self,from:data),Set(raw.keys).subtracting(["canvasView","parameters","gradientNodeID","checkpointConfiguration","compositionModel","programExportRecipe"])==["schemaVersion","contentVersion","projectID","operation","mode","graph","notes","pendingFields"] else {throw LearningContractError("Tensor draft has missing or unknown fields.")}
        if let view=raw["canvasView"] {
            let fields=try TensorLearningSubmission.checkedObject(view,required:["zoom","scrollX","scrollY","selectedNodeIDs","selectedEdgeIDs","presentation"],optional:["overviewVisible"])
            _ = try TensorLearningSubmission.checkedObject(fields["presentation"]!,required:["flowDirection","nodeDetail"])
        }
        if let parameterValue=raw["parameters"] {
            guard case .object(let nodes)=parameterValue else {throw LearningContractError("Malformed saved parameter map.")}
            for node in nodes.values {
                guard case .object(let tensors)=node else {throw LearningContractError("Malformed saved node parameters.")}
                for tensor in tensors.values {_ = try TensorLearningSubmission.checkedObject(tensor,required:["shape","values"])}
            }
        }
        try TensorLearningSubmission.checkGraphFields(raw["graph"]!,maximumEdges:(TensorRoutingLearningTask(rawValue:raw["operation"]?.stringValue ?? "") != nil || TensorCompositionLearningTask(rawValue:raw["operation"]?.stringValue ?? "") != nil || MatrixLearningTask(rawValue:raw["operation"]?.stringValue ?? "") != nil || BilinearLearningTask(rawValue:raw["operation"]?.stringValue ?? "") != nil) ? 64:31)
        _ = try TensorLearningSubmission.checkedObject(raw["contentVersion"]!,required:["major","minor","patch"])
        if let model=raw["compositionModel"] {
            let bytes=try JSONEncoder().encode(model)
            guard case .blockModel=try ModelProjectCodec.decode(bytes) else {throw LearningContractError("Composition draft requires its tagged Block model.")}
            let typed=try JSONDecoder().decode(ModelAuthoringDocument.self,from:bytes)
            guard try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(typed))==model else {throw LearningContractError("Composition draft has unknown or noncanonical nested source fields.")}
        }
        var result=try JSONDecoder().decode(Self.self,from:data)
        try result.validate()
        if let task=DataLoadingLearningTask(rawValue:result.operation) {result.graph=try DataLoadingLearningSubmission(task:task,graph:result.graph,complete:false).graph}
        if let recipe=result.programExportRecipe {result.programExportRecipe=try ProgramExportConfiguration.decodeRecipe(recipe).recipeValue}
        if let task=GradientLearningTask(rawValue:result.operation) {
            result.graph=try GradientLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        } else if let task=InitializationLearningTask(rawValue:result.operation) {
            result.graph=try InitializationLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        } else if let task=LossLearningTask(rawValue:result.operation) {
            result.graph=try LossLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        } else if let task=AttentionLearningTask(rawValue:result.operation) {
            result.graph=try AttentionLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        } else if let task=RecurrentLearningTask(rawValue:result.operation) {
            result.graph=try RecurrentLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=SchedulerLearningTask(rawValue:result.operation) {
            result.graph=try SchedulerLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=OptimizerLearningTask(rawValue:result.operation) {
            result.graph=try OptimizerLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=StatefulNormalizationLearningTask(rawValue:result.operation) {
            result.graph=try StatefulNormalizationLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=SpatialPatchLearningTask(rawValue:result.operation) {
            result.graph=try SpatialPatchLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=TensorRoutingLearningTask(rawValue:result.operation) {
            result.graph=try TensorRoutingLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=RemainingActivationLearningTask(rawValue:result.operation) {
            result.graph=try RemainingActivationLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if let task=TensorCompositionLearningTask(rawValue:result.operation) {
            result.graph=try TensorCompositionLearningSubmission.validatedGraph(result.graph,task:task,complete:false)
        }
        if BilinearLearningTask(rawValue:result.operation) != nil {
            result.graph=try BilinearLearningSubmission.validatedGraph(result.graph,complete:false)
        }
        if MatrixLearningTask(rawValue:result.operation) != nil {
            result.graph=try MatrixLearningSubmission.validatedGraph(result.graph,complete:false)
        }
        if EmbeddingLearningTask(rawValue:result.operation) != nil {
            result.graph=try TensorLearningSubmission.validatedGraph(result.graph,complete:false,allowEmbedding:true)
        }
        if EmbeddingBagLearningTask(rawValue:result.operation) != nil {
            result.graph=try EmbeddingBagLearningSubmission.validatedGraph(result.graph,complete:false)
        }
        return result
    }
}
