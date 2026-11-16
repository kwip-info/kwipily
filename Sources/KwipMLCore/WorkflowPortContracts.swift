import Foundation

private func workflowPortFailure(_ code:WorkflowCompilerError.Code,_ message:String,_ node:WorkflowNodeAddress?=nil,_ port:String?=nil)->WorkflowCompilerError {
    .init(code:code,message:message,node:node,port:port)
}
/// Declared ports remain inspectable while configuration, wires or drafts are unfinished.
/// This resolver does not assert executable shapes or measured data compatibility.
public enum WorkflowPortContracts {
    /// New authoring omits sink outputs; executable resolution retains historical ports.
    public static func resolveForAuthoring(_ node:WorkflowNode,document:WorkflowDocument,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)throws->WorkflowNodePorts {
        let ports=try resolve(node,document:document,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        if case .train=node.payload,document.schemaVersion>=16,document.schemaVersion<17 {
            return .init(inputs:ports.inputs,outputs:[.init(id:"model",kind:.frozenModel,required:false,maximumConnections:nil),.init(id:"predictions",kind:.table,required:false,maximumConnections:nil)])
        }
        if case .dataOperator(let id,_)=node.payload,dataRegistry.operatorDefinition(id:id)?.isCanvasSink == true {
            return .init(inputs:ports.inputs,outputs:[])
        }
        return ports
    }
    public static func resolve(_ node:WorkflowNode,document:WorkflowDocument,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)throws->WorkflowNodePorts {
        try resolvePlanPorts(node,document:document,modelRegistry:modelRegistry,dataRegistry:dataRegistry,historicalEvidence:false)
    }
    static func resolvePlanPorts(_ node:WorkflowNode,document:WorkflowDocument,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,historicalEvidence:Bool)throws->WorkflowNodePorts {
        let routing:[WorkflowResolvedPort]
        if case .train(let config)=node.payload,config["fineTuning"]==nil,document.nodes.contains(where:{if case .modelPrimitive("mixture_experts",_,_)=$0.payload{return true};return false}) {
            routing=[.init(id:"routing",kind:.table,required:false,maximumConnections:nil),.init(id:"routingSample",kind:.table,required:false,maximumConnections:nil)]
        }else{routing=[]}
        if case .train(let config)=node.payload,config["selfPlay"] != nil {
            let selfPlay = try SelfPlayConfiguration(config)
            return .init(inputs:[.init(id:"model",kind:.modelDefinition,required:true,maximumConnections:1),.init(id:"definition",kind:.table,required:true,maximumConnections:1)],outputs:[.init(id:"model",kind:.frozenModel,required:false,maximumConnections:nil)]+SelfPlayConfiguration.tablePorts(selfPlay.values).map{.init(id:$0,kind:.table,required:false,maximumConnections:nil)}+routing)
        }
        if case .train=node.payload,document.schemaVersion>=18 {
            return .init(inputs:[.init(id:"model",kind:FineTuningConfiguration.isExternal(node) ? .frozenModel:.modelDefinition,required:true,maximumConnections:1),.init(id:"training",kind:.table,required:true,maximumConnections:1),.init(id:"validation",kind:.table,required:true,maximumConnections:1)],outputs:[.init(id:"model",kind:.frozenModel,required:false,maximumConnections:nil),.init(id:"predictions",kind:.table,required:false,maximumConnections:nil)]+routing)
        }
        if !historicalEvidence {
            if RetiredTrainingRecipeMigration.isRetired(node) { return .init(inputs:[],outputs:[]) }
            if case .train = node.payload {
                return .init(inputs:[.init(id:"model",kind:.modelDefinition,required:true,maximumConnections:1),.init(id:"data",kind:.trainingData,required:true,maximumConnections:1)] + (document.schemaVersion < 16 ? [.init(id:"configuration",kind:.runConfiguration,required:true,maximumConnections:1)]:[]),outputs:(document.schemaVersion>=17 ? [.init(id:"model",kind:.frozenModel,required:false,maximumConnections:nil)]:[]) + [.init(id:"predictions",kind:.table,required:false,maximumConnections:nil)]+routing)
            }
        }
        func input(_ id:String,_ kind:WorkflowEdgeKind)->WorkflowResolvedPort {.init(id:id,kind:kind,required:true,maximumConnections:1)}
        func output(_ id:String,_ kind:WorkflowEdgeKind)->WorkflowResolvedPort {.init(id:id,kind:kind,required:false,maximumConnections:nil)}
        switch node.payload {
        case .dataOperator(let id,let config):
            if id == "kwip.data.union" {_ = try UnionConfiguration(config)}
            guard let op=dataRegistry.operatorDefinition(id:id)?.configured(config) else {throw workflowPortFailure(.unsupportedOperator,"Unknown data operator \(id).",node.address)}
            func port(_ value:DataFlowOperatorPort)throws->WorkflowResolvedPort {
                let kind:WorkflowEdgeKind
                switch value.kind {
                case "model":kind = .frozenModel
                case "table":kind = .table
                case "dataset":kind = .dataset
                case "state" where value.stateKind=="fittedTransform":kind = .fittedState
                default:throw workflowPortFailure(.unsupportedOperator,"Unsupported data port kind \(value.kind).",node.address,value.id)
                }
                return .init(id:value.id,kind:kind,required:value.required,maximumConnections:value.maximumConnections)
            }
            return try .init(inputs:op.inputs.map(port),outputs:op.outputs.filter {$0.inspectionOnly != true}.map(port))
        case .modelPrimitive(let type,_,let configuration):
            guard let component=modelRegistry.component(type:type) else {throw workflowPortFailure(.unsupportedOperator,"Unknown model primitive \(type).",node.address)}
            let outputs=type=="output" ? [output("definition",.modelDefinition)] : component.configuredOutputs(configuration).map {output($0,.tensor)}
            return .init(inputs:component.configuredInputs(configuration).map {input($0,.tensor)},outputs:outputs)
        case .modelBlock(let reference,_):
            guard let definition=document.definitions.first(where:{$0.reference==reference}) else {throw workflowPortFailure(.invalidModel,"Pinned Block definition is unavailable.",node.address)}
            return .init(inputs:definition.inputs.map {input($0.id,.tensor)},outputs:definition.outputs.map {output($0.id,.tensor)})
        case .optimizerRecipe(let type,let version,let config):
            guard version==1,modelRegistry.component(type:type) != nil else {throw workflowPortFailure(.unsupportedOperator,"Optimizer recipe is unavailable or has an unsupported version.",node.address)}
            _ = try OptimizerConfiguration(componentID:type,values:config)
            return .init(inputs:[],outputs:[output("out",.optimizerRecipe)])
        case .lossRecipe(let type,let version,let config):
            guard document.schemaVersion>=8,version==1,modelRegistry.component(type:type) != nil else {throw workflowPortFailure(.unsupportedOperator,"Objective recipe is unavailable or has an unsupported version.",node.address)}
            _ = try LossConfiguration(componentID:type,values:config)
            return .init(inputs:[],outputs:[output("out",.objectiveRecipe)])
        case .initializationRecipe(let type,let version,let config):
            guard document.schemaVersion>=9,version==1 else {throw workflowPortFailure(.unsupportedOperator,"Starting settings require workflow schema9 and configuration version1.",node.address)}
            _ = try InitializationConfiguration(componentID:type,values:config)
            return .init(inputs:[],outputs:[output("out",.initializationRecipe)])
        case .gradientPolicyRecipe(let type,let version,let config):
            guard document.schemaVersion>=10,version==1 else {throw workflowPortFailure(.unsupportedOperator,"Gradient/trainability controls require workflow schema10 and configuration version1.",node.address)}
            _ = try GradientPolicyConfiguration(componentID:type,values:config)
            return .init(inputs:[],outputs:[output("out",.gradientPolicyRecipe)])
        case .trainabilityRecipe(let type,let version,let config):
            guard document.schemaVersion>=10,version==1 else {throw workflowPortFailure(.unsupportedOperator,"Gradient/trainability controls require workflow schema10 and configuration version1.",node.address)}
            _ = try TrainabilityConfiguration(componentID:type,values:config)
            return .init(inputs:[],outputs:[output("out",.trainabilityRecipe)])
        case .schedulerRecipe(let type,let version,let config):
            guard version==1,modelRegistry.component(type:type) != nil else {throw workflowPortFailure(.unsupportedOperator,"Scheduler recipe is unavailable or has an unsupported version.",node.address)}
            _ = try SchedulerConfiguration(componentID:type,values:config)
            return .init(inputs:[],outputs:[output("out",.schedulerRecipe)])
        case .datasetRecipe(let type,let version,let config):
            guard document.schemaVersion>=13,version==1,type==DatasetConfiguration.componentID else {throw workflowPortFailure(.unsupportedOperator,"Data reader controls require workflow schema13 and configuration version1.",node.address)}
            _ = try DatasetConfiguration(values:config)
            return .init(inputs:[],outputs:[output("out",.datasetRecipe)])
        case .batchRecipe(let type,let version,let config):
            guard document.schemaVersion>=13,version==1,type==DataLoaderConfiguration.componentID else {throw workflowPortFailure(.unsupportedOperator,"Data reader controls require workflow schema13 and configuration version1.",node.address)}
            _ = try DataLoaderConfiguration(values:config)
            return .init(inputs:[],outputs:[output("out",.batchRecipe)])
        case .modelDefinition:return .init(inputs:[input("source",.modelDefinition)],outputs:[output("model",.modelDefinition)])
        case .trainingData:return .init(inputs:[input("training",historicalEvidence ? .dataset:.table),input("validation",historicalEvidence ? .dataset:.table)],outputs:[output("data",.trainingData)])
        case .runSetup:return .init(inputs:[],outputs:[output("configuration",.runConfiguration)])
        case .frozenModel:return .init(inputs:[],outputs:[output("model",.frozenModel)])
        case .gameSession(let config):
            return .init(inputs:try GameSessionConfiguration(config).inputPorts,outputs:["summary","episodes","moves"].map{output($0,.table)})
        case .generateText(let config):
            let c=try TextGenerationConfiguration(config)
            return .init(inputs:[input("model",.frozenModel)] + (c.tableMode ? [input("prompts",.table)]:[]),outputs:[output("table",.table)])
        case .saveModel:
            guard document.schemaVersion>=17 else {throw workflowPortFailure(.unsupportedOperator,"Save model requires workflow schema17.",node.address)}
            return .init(inputs:[input("model",.frozenModel)],outputs:[])
        case .evaluateTest:return .init(inputs:[input("model",.frozenModel),input("test",historicalEvidence ? .dataset:.table)],outputs:historicalEvidence ? []:[output("predictions",.table)])
        case .estimator(let config):
            if PrebuiltModelConfiguration.isPrebuilt(config) {
                let decoded=try PrebuiltModelConfiguration.decode(config)
                let outputs=[output("predictions",.table)]+(PrebuiltModelConfiguration.isCluster(config) ? [output("profiles",.table)]:PrebuiltModelConfiguration.algorithms[decoded.algorithm ?? ""]?.task == "classification" && decoded.operation != "predict" ? [output("confusion",.table)]:[])
                if decoded.isPrediction {return .init(inputs:[input("model",.frozenModel),input("table",.table)],outputs:outputs)}
                return .init(inputs:[input("training",.table)]+(PrebuiltModelConfiguration.isCluster(config) ? []:[input("validation",.table)]),outputs:(PrebuiltModelConfiguration.hasModel(config) ? [output("model",.frozenModel)]:[])+outputs)
            }
            if DocumentClassificationConfiguration.isDocument(config) {
                if let op=config["operation"]?.stringValue,op != "fit" {return .init(inputs:[input("model",.frozenModel),input("table",.table)],outputs:[output("predictions",.table)])}
                return .init(inputs:[input("training",.table),input("validation",.table)],outputs:[output("model",.frozenModel),output("predictions",.table)])
            }
            return .init(inputs:[input("training",historicalEvidence ? .dataset:.table),input("validation",historicalEvidence ? .dataset:.table)] + (config["backendID"] == .string("tensorflow") ? [input("model",.modelDefinition)] : []),outputs:[])
        case .train:return .init(inputs:[input("model",.modelDefinition),input("data",.trainingData)] + (document.schemaVersion < 16 ? [input("configuration",.runConfiguration)]:[]) + (document.schemaVersion>=6 ? [.init(id:"optimizer",kind:.optimizerRecipe,required:false,maximumConnections:1)]:[]) + (document.schemaVersion>=7 ? [.init(id:"scheduler",kind:.schedulerRecipe,required:false,maximumConnections:1)]:[]) + (document.schemaVersion>=8 ? [.init(id:"objective",kind:.objectiveRecipe,required:false,maximumConnections:1)]:[]) + (document.schemaVersion>=9 ? [.init(id:"initialization",kind:.initializationRecipe,required:false,maximumConnections:1)]:[]) + (document.schemaVersion>=10 ? [.init(id:"gradientPolicy",kind:.gradientPolicyRecipe,required:false,maximumConnections:1),.init(id:"trainability",kind:.trainabilityRecipe,required:false,maximumConnections:1)]:[]) + (document.schemaVersion>=13 ? [.init(id:"dataset",kind:.datasetRecipe,required:false,maximumConnections:1),.init(id:"loader",kind:.batchRecipe,required:false,maximumConnections:1)]:[]),outputs:document.schemaVersion>=17 ? [output("model",.frozenModel)]:[])
        }
    }
}
