import Foundation

public enum WorkflowDomain:String,Codable,Hashable {case data,model,control}
public struct WorkflowNodeAddress:Codable,Equatable,Hashable {
    public var domain:WorkflowDomain
    public var localID:String
    public init(domain:WorkflowDomain,localID:String){self.domain=domain;self.localID=localID}
}
public struct WorkflowEndpoint:Codable,Equatable,Hashable {
    public var address:WorkflowNodeAddress
    public var port:String
    public init(address:WorkflowNodeAddress,port:String){self.address=address;self.port=port}
}
/// A saved claim, not proof of compatibility. The compiler validates actual registry ports.
public enum WorkflowEdgeKind:String,Codable,Hashable {case table,dataset,fittedState,tensor,modelDefinition,trainingData,runConfiguration,frozenModel,optimizerRecipe,schedulerRecipe,objectiveRecipe,initializationRecipe,gradientPolicyRecipe,trainabilityRecipe,datasetRecipe,batchRecipe}
public struct WorkflowEdge:Codable,Equatable,Identifiable {
    public var id:String
    public var from:WorkflowEndpoint
    public var to:WorkflowEndpoint
    public var kind:WorkflowEdgeKind
    public init(id:String=UUID().uuidString,from:WorkflowEndpoint,to:WorkflowEndpoint,kind:WorkflowEdgeKind){self.id=id;self.from=from;self.to=to;self.kind=kind}
}
public enum WorkflowNodePayload:Equatable,Codable {
    case dataOperator(operatorID:String,config:[String:JSONValue])
    case modelPrimitive(type:String,configurationVersion:Int,config:[String:JSONValue])
    case modelBlock(reference:CompositeDefinitionReference,overrides:[String:JSONValue])
    case optimizerRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case schedulerRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case lossRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case initializationRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case gradientPolicyRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case trainabilityRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case datasetRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case batchRecipe(type:String,configurationVersion:Int,config:[String:JSONValue])
    case modelDefinition
    case trainingData(config:[String:JSONValue])
    case runSetup(config:[String:JSONValue])
    case estimator(config:[String:JSONValue])
    case train(config:[String:JSONValue])
    case frozenModel(config:[String:JSONValue])
    case evaluateTest(config:[String:JSONValue])
    case generateText(config:[String:JSONValue])
    case gameSession(config:[String:JSONValue])
    case saveModel(config:[String:JSONValue])
    public var domain:WorkflowDomain {
        switch self {case .dataOperator:return .data;case .modelPrimitive,.modelBlock:return .model;default:return .control}
    }
    private enum Keys:String,CodingKey {case kind,operatorID,type,configurationVersion,config,reference,overrides}
    public init(from decoder:Decoder)throws {
        let c=try decoder.container(keyedBy:Keys.self)
        switch try c.decode(String.self,forKey:.kind) {
        case "dataOperator":self = .dataOperator(operatorID:try c.decode(String.self,forKey:.operatorID),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "modelPrimitive":self = .modelPrimitive(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "optimizerRecipe":self = .optimizerRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "lossRecipe":self = .lossRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "initializationRecipe":self = .initializationRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "gradientPolicyRecipe":self = .gradientPolicyRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "trainabilityRecipe":self = .trainabilityRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "schedulerRecipe":self = .schedulerRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "modelBlock":self = .modelBlock(reference:try c.decode(CompositeDefinitionReference.self,forKey:.reference),overrides:try c.decode([String:JSONValue].self,forKey:.overrides))
        case "datasetRecipe":self = .datasetRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try c.decode([String:JSONValue].self,forKey:.config))
        case "batchRecipe":self = .batchRecipe(type:try c.decode(String.self,forKey:.type),configurationVersion:try c.decode(Int.self,forKey:.configurationVersion),config:try DataLoaderConfiguration(values:c.decode([String:JSONValue].self,forKey:.config)).values)
        case "modelDefinition":self = .modelDefinition
        case "trainingData":self = .trainingData(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "runSetup":self = .runSetup(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "frozenModel":self = .frozenModel(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "gameSession":self = .gameSession(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "generateText":self = .generateText(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "saveModel":self = .saveModel(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "evaluateTest":self = .evaluateTest(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "estimator":self = .estimator(config:try c.decode([String:JSONValue].self,forKey:.config))
        case "train":
            var config=try c.decode([String:JSONValue].self,forKey:.config)
            if let raw=config["batching"] {
                guard case .object(let values)=raw else {throw DataFlowContractError("Train batching must be its exact configuration object.")}
                config["batching"] = .object(try DataLoaderConfiguration(values:values).values)
            }
            self = .train(config:try WorkflowTrainRecipes.canonicalize(config))
        default:throw DataFlowContractError("Unknown workflow payload kind.")
        }
    }
    public func encode(to encoder:Encoder)throws {
        var c=encoder.container(keyedBy:Keys.self)
        switch self {
        case .dataOperator(let id,let config):try c.encode("dataOperator",forKey:.kind);try c.encode(id,forKey:.operatorID);try c.encode(config,forKey:.config)
        case .modelPrimitive(let type,let version,let config):try c.encode("modelPrimitive",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .optimizerRecipe(let type,let version,let config):try c.encode("optimizerRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .lossRecipe(let type,let version,let config):try c.encode("lossRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .initializationRecipe(let type,let version,let config):try c.encode("initializationRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .gradientPolicyRecipe(let type,let version,let config):try c.encode("gradientPolicyRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .trainabilityRecipe(let type,let version,let config):try c.encode("trainabilityRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .schedulerRecipe(let type,let version,let config):try c.encode("schedulerRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .modelBlock(let reference,let overrides):try c.encode("modelBlock",forKey:.kind);try c.encode(reference,forKey:.reference);try c.encode(overrides,forKey:.overrides)
        case .datasetRecipe(let type,let version,let config):try c.encode("datasetRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .batchRecipe(let type,let version,let config):try c.encode("batchRecipe",forKey:.kind);try c.encode(type,forKey:.type);try c.encode(version,forKey:.configurationVersion);try c.encode(config,forKey:.config)
        case .modelDefinition:try c.encode("modelDefinition",forKey:.kind)
        case .trainingData(let config):try c.encode("trainingData",forKey:.kind);try c.encode(config,forKey:.config)
        case .runSetup(let config):try c.encode("runSetup",forKey:.kind);try c.encode(config,forKey:.config)
        case .frozenModel(let config):try c.encode("frozenModel",forKey:.kind);try c.encode(config,forKey:.config)
        case .gameSession(let config):try c.encode("gameSession",forKey:.kind);try c.encode(config,forKey:.config)
        case .generateText(let config):try c.encode("generateText",forKey:.kind);try c.encode(config,forKey:.config)
        case .saveModel(let config):try c.encode("saveModel",forKey:.kind);try c.encode(config,forKey:.config)
        case .evaluateTest(let config):try c.encode("evaluateTest",forKey:.kind);try c.encode(config,forKey:.config)
        case .estimator(let config):try c.encode("estimator",forKey:.kind);try c.encode(config,forKey:.config)
        case .train(let config):try c.encode("train",forKey:.kind);try c.encode(config,forKey:.config)
        }
    }
}
public struct WorkflowNode:Codable,Equatable,Identifiable {
    public var parameterSettings:[String:JSONValue]? = nil
    public var address:WorkflowNodeAddress
    public var name:String
    public var x:Double
    public var y:Double
    public var payload:WorkflowNodePayload
    public var id:WorkflowNodeAddress {address}
    public init(address:WorkflowNodeAddress,name:String="",x:Double=0,y:Double=0,payload:WorkflowNodePayload,parameterSettings:[String:JSONValue]? = nil){self.parameterSettings=parameterSettings;self.address=address;self.name=name;self.x=x;self.y=y;self.payload=payload}
}
/// One mixed canonical graph. Worker documents are derived, never independently stored here.
public struct WorkflowDocument:Codable,Equatable {
    public var schemaVersion=1
    public var kind="workflowProject"
    public var projectID:String
    public var revision:UInt64
    public var name:String
    public var nodes:[WorkflowNode]
    public var edges:[WorkflowEdge]
    public var definitions:[CompositeDefinition]
    public var drafts:[ModelBlockDraft]
    /// Model-definition local ID → retained input boundary local ID (schema14).
    public var namedModelInputs:[String:[String:String]]?
    public var modelInputs:[String:String]?
    /// Retain pre-existing composite source semantics across an ownership-only migration.
    public var modelSourceVersions:[String:Int]?
    public var templateOrigin: WorkflowTemplateOrigin?
    public var guidedBuild:WorkflowGuidedBuildProgress?
    public init(projectID:String,revision:UInt64=0,name:String="Untitled workflow",nodes:[WorkflowNode]=[],edges:[WorkflowEdge]=[],definitions:[CompositeDefinition]=[],drafts:[ModelBlockDraft]=[]){self.projectID=projectID;self.revision=revision;self.name=name;self.nodes=nodes;self.edges=edges;self.definitions=definitions;self.drafts=drafts}
}
/// The caller must resolve these private/unconverted values explicitly. This is deliberately not Codable.
public struct WorkflowMigrationResult {
    public let document:WorkflowDocument
    public let retainedExecution:ExecutionSettings?
    public let sourceBindings:[WorkflowNodeAddress:String]
}

/// Guided placement progress is not a correctness or mastery receipt.
public struct WorkflowGuidedBuildProgress:Codable,Equatable {
    public var courseID:String
    public var version:Int
    public var completedSteps:Int
    public init(courseID:String,version:Int,completedSteps:Int) {
        self.courseID=courseID;self.version=version;self.completedSteps=completedSteps
    }
}

public extension WorkflowDocument {
    /// Result freshness excludes saved presentation, never execution configuration or wiring.
    func matchesCapturedConstruction(_ captured: WorkflowDocument) -> Bool {
        guard nodes.count == captured.nodes.count else { return false }
        var construction = self
        construction.name = captured.name
        construction.revision = captured.revision
        for index in nodes.indices {
            guard nodes[index].address == captured.nodes[index].address else { return false }
            if case .modelPrimitive("mixture_experts",let version,let config)=nodes[index].payload,
               case .modelPrimitive("mixture_experts",let beforeVersion,let before)=captured.nodes[index].payload,version==beforeVersion,
               MixtureExpertsConfiguration.scientificValues(config)==MixtureExpertsConfiguration.scientificValues(before) {construction.nodes[index].payload=captured.nodes[index].payload}
            construction.nodes[index].name = captured.nodes[index].name
            construction.nodes[index].x = captured.nodes[index].x
            construction.nodes[index].y = captured.nodes[index].y
        }
        return construction == captured
    }
}
