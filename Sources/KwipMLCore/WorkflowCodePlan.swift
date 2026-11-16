import Foundation
import CryptoKit

/// Pure captured source for derived Code. File grants, local runtime paths,
/// source bytes, cache locations and checkpoint locations belong to invocation.
public struct WorkflowCodePlan:Codable,Equatable {
    public enum Scope:String,Codable,CaseIterable {case workflow,preparation,model,training}
    public struct Preparation:Codable,Equatable {
        public let graph:DataFlowDocument
        public let nodeLabels:[String:String]
        public let engine:String
        public let sampling: PreparationSampling?
        public let completeNodes: [String]?
    }
    public struct Model:Codable,Equatable {
        public let id:String
        public let graph:GraphDocument
        public let semanticSHA256:String
        public let origins:[String:CompositeNodeOrigin]
        public let provenance:CompositeSourceSnapshot?
    }
    public struct Prepared:Codable,Equatable {
        public let trainingNode:String
        public let trainingPort:String
        public let validationNode:String
        public let validationPort:String
        public let featureColumns:[String]
        public let targetColumn:String?
        public let bagInput:BagInputMapping?
        public let inputDtype:String?
        public let sampleShape:[Int]?
    }
    public struct Training:Codable,Equatable {
        public let id:String
        public let backendID:String
        public let graph:GraphDocument?
        public let configuration:[String:JSONValue]?
        public let prepared:Prepared
        public let settings:ExecutionSettings?
        /// Existing runner keyword names; values come from selected typed recipes.
        public let recipes:[String:JSONValue]
        public let recipeSources:[String:WorkflowNodeAddress]
        public let trainingMode:WorkflowTrainingMode
        public let preparationCachePolicy:WorkflowPreparationCachePolicy
    }
    public let schemaVersion:Int
    public let scope:Scope
    public let document:WorkflowDocument
    public let compilerContractVersion:Int
    public let modelRegistry:ComponentRegistry
    public let dataRegistry:DataFlowOperatorRegistry
    public let selectedTargets:[WorkflowNodeAddress]
    public let preparation:Preparation?
    public let models:[Model]
    public let trainings:[Training]
    public static let maximumBytes=2_000_000

    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let bytes=try encoder.encode(self)
        guard bytes.count<=Self.maximumBytes else {throw DataFlowContractError("Code snapshot exceeds its 2 MB bound; choose a smaller scope.")}
        return bytes
    }
    public var contentIdentity:String {get throws {
        // Moving blocks or saving a new revision must not spawn another worker.
        // Normalize typed layout fields only: data columns/configuration named
        // "x", "y" or "revision" retain their scientific meaning.
        func graph(_ value:GraphDocument)->GraphDocument {
            var result=value
            for index in result.nodes.indices {result.nodes[index].x=0;result.nodes[index].y=0}
            return result
        }
        func composite(_ value:CompositeGraph)->CompositeGraph {
            var result=value
            for index in result.nodes.indices {result.nodes[index].x=0;result.nodes[index].y=0}
            for index in result.instances.indices {result.instances[index].x=0;result.instances[index].y=0}
            return result
        }
        func definitions(_ values:[CompositeDefinition])->[CompositeDefinition] {
            values.map {var result=$0;result.body=composite(result.body);return result}
        }
        var source=document;source.revision=0
        for index in source.nodes.indices {source.nodes[index].x=0;source.nodes[index].y=0}
        source.definitions=definitions(source.definitions)
        let normalizedModels=models.map {value->Model in
            var provenance=value.provenance
            if var original=provenance {
                original.expandedGraph=graph(original.expandedGraph)
                original.sourceCompositeDocument.root=composite(original.sourceCompositeDocument.root)
                original.sourceCompositeDocument.definitions=definitions(original.sourceCompositeDocument.definitions)
                provenance=original
            }
            return .init(id:value.id,graph:graph(value.graph),semanticSHA256:value.semanticSHA256,origins:value.origins,provenance:provenance)
        }
        let normalizedTraining=trainings.map {value in
            Training(id:value.id,backendID:value.backendID,graph:value.graph.map(graph),configuration:value.configuration,prepared:value.prepared,settings:value.settings,recipes:value.recipes,recipeSources:value.recipeSources,trainingMode:value.trainingMode,preparationCachePolicy:value.preparationCachePolicy)
        }
        let normalized=Self(schemaVersion:schemaVersion,scope:scope,document:source,compilerContractVersion:compilerContractVersion,modelRegistry:modelRegistry,dataRegistry:dataRegistry,selectedTargets:selectedTargets,preparation:preparation,models:normalizedModels,trainings:normalizedTraining)
        return SHA256.hash(data:try normalized.encoded()).map {String(format:"%02x",$0)}.joined()
    }}
    /// Reproduce rather than trust a decoded projection. Does no file or source I/O.
    public func validate()throws {
        let reproduced=try Self.compilePlan(document,scope:scope,selectedTargets:selectedTargets,modelRegistry:modelRegistry,dataRegistry:dataRegistry,sampling:preparation?.sampling ?? .all,historicalEvidence:true)
        guard reproduced==self else {throw DataFlowContractError("Code snapshot differs from its captured workflow and compilers.")}
    }
    public static func compile(_ document:WorkflowDocument,scope:Scope = .workflow,
                               selectedTargets:[WorkflowNodeAddress] = [],
                               modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,sampling:PreparationSampling = .all)throws->Self {
        try RetiredTrainingRecipeMigration.requireCurrent(document)
        try RetiredDataOperatorMigration.requireCurrent(document)
        return try compilePlan(document,scope:scope,selectedTargets:selectedTargets,modelRegistry:modelRegistry,dataRegistry:dataRegistry,sampling:sampling,historicalEvidence:false)
    }
    private static func compilePlan(_ document:WorkflowDocument,scope:Scope,selectedTargets:[WorkflowNodeAddress],modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,sampling:PreparationSampling,historicalEvidence:Bool)throws->Self {
        if scope != .model,document.nodes.contains(where:{node in if case .train(let config)=node.payload,case .object(let fine)=config["fineTuning"] {return fine["task"] == .string("typedDecisions")};return false}) {throw DataFlowContractError("Typed-decision training Code has not qualified portable model-resource bindings. Use native Train and Save model; no causal training task is substituted.")}
        if document.nodes.contains(where:{node in if case .dataOperator(let id,_)=node.payload {return id==TypedDecisionConfiguration.operatorID};return false}),scope != .model {throw DataFlowContractError("Typed-decision Code export requires captured model resources. Use native Run Flow and Table Output for Decisions and Scores.")}
        if document.nodes.contains(where:{node in if case .dataOperator(let id,_)=node.payload {return id==AudioClassificationConfiguration.operatorID};return false}),scope != .model {throw DataFlowContractError("Audio classification Code export requires captured model resources and is not supported yet. Use native Run Flow and Table Output for Predictions or Scores.")}
        if document.nodes.contains(where:{node in if case .dataOperator(let id,_)=node.payload {return id==SpeechTranscriptionConfiguration.operatorID};return false}),scope != .model {throw DataFlowContractError("Speech workflow Code export requires captured model resources and is not supported yet. Use native Run Flow, and Table Output to export Transcripts or Segments.")}
        if document.nodes.contains(where:{node in if case .dataOperator(let id,_)=node.payload {return id==OCRConfiguration.operatorID};return false}),scope != .model {throw DataFlowContractError("OCR workflow Code export requires the verified optional pack capture and is not supported yet. Use the native workflow and export its scalar Documents/Regions through Table Output.")}

        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty else {throw DataFlowContractError("Apply or discard component drafts before generating Code.")}
        let selected=Set(selectedTargets)
        guard selected.count==selectedTargets.count,selected.isSubset(of:Set(document.nodes.map(\.address))) else {
            throw DataFlowContractError("Code selection must contain distinct current workflow nodes.")
        }
        if scope == .workflow && !selected.isEmpty {throw DataFlowContractError("Entire workflow includes every terminal; choose a narrower scope for selected targets.")}
        if scope == .preparation && !selected.isEmpty {throw DataFlowContractError("Preparation Code includes the complete data region.")}
        if document.nodes.contains(where:{node in if case .gameSession=node.payload{return scope == .workflow || selected.contains(node.address)};return false}) {throw DataFlowContractError("Whole-workflow Code cannot yet replay Game session. Use native Run Flow and its exact recorded Summary/Episodes/Moves tables; no terminal is silently omitted.")}
        if document.nodes.contains(where:{node in if case .generateText=node.payload {return scope == .workflow || selected.contains(node.address)};return false}) {throw DataFlowContractError("Whole-workflow Code cannot yet replay Generate text. Use the captured portable generation package or select a model/training scope.")}
        if document.nodes.contains(where:{node in if case .saveModel=node.payload {return scope == .workflow || selected.contains(node.address)};return false}) {throw DataFlowContractError("Code export does not yet replay Save model terminals. Select a supported model/training scope or Data preparation.")}
        if scope == .workflow && document.nodes.contains(where:{node in switch node.payload {case .evaluateTest:return true;case .frozenModel:return !document.edges.contains{edge in edge.from.address==node.address && document.nodes.contains{other in other.address==edge.to.address && FineTuningConfiguration.isExternal(other)}};default:return false}}) {
            throw DataFlowContractError("Entire workflow Code does not yet support frozen-model evaluation terminals; choose Data preparation or a supported model/training scope.")
        }
        var models:[Model]=[],trainings:[Training]=[]
        var capturedModels:[WorkflowNodeAddress:Model]=[:]
        var trainingDataIDs=Set<String>()
        func model(_ snapshot:ModelExecutionSnapshot,id:WorkflowNodeAddress)->Model {
            var graph=snapshot.graph;graph.execution=nil
            var provenance=snapshot.provenance
            provenance?.sourceCompositeDocument.execution=nil
            provenance?.expandedGraph.execution=nil
            return .init(id:id.localID,graph:graph,semanticSHA256:snapshot.semanticSHA256,origins:snapshot.origins,provenance:provenance)
        }
        func json<T:Encodable>(_ value:T)throws->JSONValue {try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(value))}
        let trainNodes=document.nodes.filter {node in
            guard selected.isEmpty || selected.contains(node.address) else {return false}
            switch node.payload {case .train,.estimator:return true;default:return false}
        }
        if scope == .training && (trainNodes.isEmpty || (!selected.isEmpty && selected != Set(trainNodes.map(\.address)))) {
            throw DataFlowContractError("Select supported Train or estimator terminals for Training Code.")
        }
        if scope == .workflow || scope == .training {
            for node in trainNodes {
                switch node.payload {
                case .train:
                    guard !SelfPlayConfiguration.isSelfPlay(node) else {throw DataFlowContractError("Self-play uses captured rules and required replay. Export the paired inference model; standalone Self-play Code export is not qualified.")}
                    if FineTuningConfiguration.isExternal(node) {
                        let p=try WorkflowFineTuningPlan.compile(document,train:node.address,dataRegistry:dataRegistry)
                        let data=try p.preparationDocument(document);trainingDataIDs.formUnion(data.nodes.map(\.address.localID))
                        guard case .train(let config)=node.payload else{throw DataFlowContractError("Missing fine-tuning configuration.")}
                        guard config["gradientPolicy"]==nil,config["initialization"]==nil,config["trainability"]==nil,config["objective"]==nil,config["datasetAccess"]==nil,config["batching"]==nil else{throw DataFlowContractError("Imported-model Code does not silently drop canvas-only recipe overrides.")}
                        let run=try WorkflowTrainRunSetup.resolve(in:document,train:node.address).configuration
                        guard run["experiment"]==nil,run["preparationCachePolicy"]==nil || run["preparationCachePolicy"] == .string("off"),config["trainingMode"]==nil || config["trainingMode"] == .string("fresh") else{throw DataFlowContractError("Causal Code requires its explicit task boundary without a canvas checkpoint, experiment or cache override.")}
                        let prepared=Prepared(trainingNode:p.training.address.localID,trainingPort:p.training.port,validationNode:p.validation.address.localID,validationPort:p.validation.port,featureColumns:[],targetColumn:nil,bagInput:nil,inputDtype:nil,sampleShape:nil)
                        trainings.append(.init(id:node.address.localID,backendID:"pretrainedCausal",graph:nil,configuration:["model":.string(p.model.localID),"train":.object(config)],prepared:prepared,settings:nil,recipes:[:],recipeSources:[:],trainingMode:.fresh,preparationCachePolicy:.off))
                        continue
                    }
                    let p=try historicalEvidence ? WorkflowCompiler.compileHistoricalEvidence(document,train:node.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:document.schemaVersion>=17 ? 15:document.schemaVersion>=16 ? 14:document.schemaVersion>=15 ? 13:12) : WorkflowCompiler.compile(document,train:node.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:document.schemaVersion>=17 ? 15:document.schemaVersion>=16 ? 14:document.schemaVersion>=15 ? 13:12)
                    trainingDataIDs.formUnion(p.dataflow.nodes.map(\.id))
                    var settings=p.settings;settings.datasetCSV=nil;settings.preparedTraining=nil;settings.compositeSource=nil
                    var recipes:[String:JSONValue]=[:],sources:[String:WorkflowNodeAddress]=[:]
                    func recipe(_ key:String,_ value:JSONValue,_ source:WorkflowNodeAddress) {recipes[key]=value;sources[key]=source}
                    if let plan=p.namedInputPlan {recipes["named_input_plan"] = .object(plan.values)}
                    if let r=p.optimizerRecipe {recipe("optimizer_recipe",r.configuration.recipeValue,r.source)}
                    if let r=p.schedulerRecipe {recipe("scheduler_recipe",r.configuration.recipeValue,r.source)}
                    if let r=p.lossRecipe {
                        recipe("objective_recipe",r.configuration.recipeValue,r.source)
                        recipes["target_mapping"]=try json(r.targetMapping)
                        recipes["objective_scalarization"] = .string(p.objectiveScalarization.rawValue)
                    }
                    if let r=p.initializationRecipe {recipe("initialization_recipe",r.configuration.recipeValue,r.source)}
                    if let r=p.gradientPolicyRecipe {recipe("gradient_policy",r.configuration.recipeValue,r.source)}
                    if let r=p.trainabilityRecipe {recipe("trainability",r.configuration.recipeValue,r.source)}
                    if let r=p.datasetRecipe {recipe("dataset_recipe",r.configuration.recipeValue,r.source)}
                    if let r=p.loaderRecipe {recipe("loader_recipe",r.configuration.recipeValue,r.source)}
                    if let capture=settings.gradientCapture {recipes["gradient_capture"]=try json(capture)}
                    let contract=try p.bagInput == nil ? WorkflowMappingRequirements.inputContract(p.model.graph,registry:modelRegistry,context:p.lossRecipe == nil ? .preparedWorkflow:.explicitObjective):nil
                    let prepared=Prepared(trainingNode:p.training.node,trainingPort:p.training.port,validationNode:p.validation.node,validationPort:p.validation.port,featureColumns:p.featureColumns,targetColumn:p.targetColumn,bagInput:p.bagInput,inputDtype:contract?.dtype.rawValue,sampleShape:contract?.layout.sample.dimensions)
                    var graph=p.model.graph;graph.execution=nil
                    trainings.append(.init(id:node.address.localID,backendID:"pytorch",graph:graph,configuration:nil,prepared:prepared,settings:settings,recipes:recipes,recipeSources:sources,trainingMode:p.trainingMode,preparationCachePolicy:p.preparationCachePolicy))
                    if let address=document.edges.first(where:{$0.to.address==node.address && $0.to.port=="model"})?.from.address {
                        capturedModels[address]=model(p.model,id:address)
                    }
                case .estimator:
                    let p=try historicalEvidence
                        ? WorkflowEstimatorCompiler.compileHistoricalEvidence(document,estimator:node.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
                        : WorkflowEstimatorCompiler.compile(document,estimator:node.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
                    if p.configuration.isPrediction {throw DataFlowContractError("Code export does not embed private classifier package bindings. Export the verified classifier inference package from Save model; its inference.py accepts unseen text.")}
                    trainingDataIDs.formUnion(p.dataflow.nodes.map(\.id))
                    let prepared=Prepared(trainingNode:p.training.node,trainingPort:p.training.port,validationNode:p.validation.node,validationPort:p.validation.port,featureColumns:p.configuration.featureColumns,targetColumn:p.configuration.targetColumn,bagInput:nil,inputDtype:p.configuration.isPrebuilt ? "float64":"float32",sampleShape:[p.configuration.isPrebuilt ? p.configuration.featureColumns.count:2])
                    var graph=p.model?.graph;graph?.execution=nil
                    trainings.append(.init(id:node.address.localID,backendID:p.configuration.backendID,graph:graph,configuration:p.configuration.isPrebuilt ? ["task":.string("prebuilt"),"algorithm":.string(p.configuration.algorithm ?? ""),"operation":.string(p.configuration.operation),"settings":.object(p.configuration.settings)]:p.configuration.isDocumentClassification ? ["task":.string(DocumentClassificationConfiguration.task),"settings":.object(p.configuration.settings)]:p.configuration.settings,prepared:prepared,settings:nil,recipes:[:],recipeSources:[:],trainingMode:.fresh,preparationCachePolicy:.off))
                    if let snapshot=p.model,let address=document.edges.first(where:{$0.to.address==node.address && $0.to.port=="model"})?.from.address {capturedModels[address]=model(snapshot,id:address)}
                default:break
                }
            }
        }
        if scope == .workflow || scope == .model {
            let definitions=document.nodes.filter {node in
                if WorkflowModelBoundary.isVisibleDefinition(node,in:document) {return selected.isEmpty || selected.contains(node.address)};return false
            }
            if scope == .model && (definitions.isEmpty || (!selected.isEmpty && selected != Set(definitions.map(\.address)))) {throw DataFlowContractError("Select Model definition components for Model Code.")}
            for node in definitions {
                if let existing=capturedModels[node.address] {models.append(existing)}
                else {models.append(model(try WorkflowCompiler.modelSnapshot(document,definition:node.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry),id:node.address))}
            }
        }
        guard Set(trainings.map { $0.preparationCachePolicy.rawValue }).count<=1 else {
            throw DataFlowContractError("These terminals use different preparation cache policies. Choose one training target for Code until separate preparation invocations are supported.")
        }
        let preparation:Preparation?
        if scope != .model,document.nodes.contains(where:{$0.address.domain == .data}) {
            var source=document
            if scope == .training {
                // This temporary preparation slice has no model owners. Keep their
                // retained metadata on the original Code document, not orphaned here.
                source.modelInputs=nil;source.modelSourceVersions=nil
                source.nodes=document.nodes.filter {$0.address.domain == .data && trainingDataIDs.contains($0.address.localID)}
                source.edges=document.edges.filter {$0.from.address.domain == .data && $0.to.address.domain == .data && trainingDataIDs.contains($0.from.address.localID) && trainingDataIDs.contains($0.to.address.localID)}
            }
            let graph=try historicalEvidence
                ? WorkflowDataPreparationPlan.compileHistoricalEvidence(source,registry:dataRegistry,allowTrainingTerminals:true)
                : WorkflowDataPreparationPlan.compile(source,registry:dataRegistry,allowTrainingTerminals:true)
            let labels=Dictionary(uniqueKeysWithValues:source.nodes.filter {$0.address.domain == .data}.map {node->(String,String) in
                let label:String
                if !node.name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {label=node.name}
                else if case .dataOperator(let id,_)=node.payload {label=dataRegistry.operatorDefinition(id:id)?.title ?? node.address.localID}
                else {label=node.address.localID}
                return (node.address.localID,label)
            })
            preparation = .init(graph:graph,nodeLabels:labels,engine:"polars",sampling:try sampling.validated() == .all ? nil : sampling,completeNodes:nil)
        } else {preparation=nil}
        guard preparation != nil || !models.isEmpty || !trainings.isEmpty else {throw DataFlowContractError("This Code scope has no supported preparation, model or training components.")}
        let result=Self(schemaVersion:1,scope:scope,document:document,compilerContractVersion:document.schemaVersion>=17 ? 15:document.schemaVersion>=16 ? 14:document.schemaVersion>=15 ? 13:12,modelRegistry:modelRegistry,dataRegistry:dataRegistry,selectedTargets:selectedTargets,preparation:preparation,models:models,trainings:trainings)
        _=try result.encoded();return result
    }
}
