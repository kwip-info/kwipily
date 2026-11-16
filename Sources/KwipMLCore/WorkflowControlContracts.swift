import Foundation

public enum WorkflowObjectiveScalarization:String,Codable,Equatable {case mean,sum}

/// The authoring inspector and execution compiler share these control contracts.
/// Reading defaults never changes the canonical workflow's raw configuration.
public enum WorkflowRunFieldKind:Equatable {
    case integer(ClosedRange<Int64>)
    case positiveDecimal(maximum:Double)
    case nonnegativeDecimal(maximum:Double)
    case choice([String])
}
public struct WorkflowRunField:Equatable {
    public let key:String
    public let title:String
    public let kind:WorkflowRunFieldKind
    public let defaultValue:JSONValue
    public func accepts(_ value:JSONValue)->Bool {
        switch kind {
        case .integer(let bounds):
            guard case .integer(let number)=value else {return false}
            return bounds.contains(number)
        case .positiveDecimal(let maximum):
            guard let number=value.doubleValue else {return false}
            return number.isFinite && number>0 && number<=maximum
        case .nonnegativeDecimal(let maximum):
            guard let number=value.doubleValue else {return false}
            return number.isFinite && number>=0 && number<=maximum
        case .choice(let values):
            guard case .string(let string)=value else {return false}
            return values.contains(string)
        }
    }
}
/// Preparation execution policy, deliberately independent of model settings and identity.
public enum WorkflowPreparationCachePolicy:String,Codable,CaseIterable {
    case off,materialize,reuse
}
public enum WorkflowControlContracts {
    public static func preparationCachePolicy(_ config:[String:JSONValue],schemaVersion:Int)throws->WorkflowPreparationCachePolicy {
        guard [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23].contains(schemaVersion) else {throw DataFlowContractError("Unsupported workflow cache policy schema.")}
        guard let value=config["preparationCachePolicy"] else {return .off}
        guard schemaVersion>=2 else {throw DataFlowContractError("Preparation cache policy requires workflow schema 2 or later.")}
        guard case .string(let raw)=value,let policy=WorkflowPreparationCachePolicy(rawValue:raw) else {
            throw DataFlowContractError("Preparation cache policy must be off, materialize or reuse.")
        }
        return policy
    }

    public static var runFields:[WorkflowRunField] {
        let defaults=ExecutionSettings()
        return [
            .init(key:"device",title:"Device",kind:.choice(["auto","cpu","mps"]),defaultValue:.string(defaults.device)),
            .init(key:"epochs",title:"Epochs",kind:.integer(1...10000),defaultValue:.integer(Int64(defaults.epochs))),
            .init(key:"batchSize",title:"Batch size",kind:.integer(1...100000),defaultValue:.integer(Int64(defaults.batchSize))),
            .init(key:"seed",title:"Seed",kind:.integer(0...4294967295),defaultValue:.integer(Int64(defaults.seed))),
            .init(key:"learningRate",title:"Learning rate",kind:.positiveDecimal(maximum:1),defaultValue:.number(defaults.learningRate))
        ]
    }
    public static var experimentFields:[WorkflowRunField] {runFields + [.init(key:"optimizer.lr",title:"Optimizer learning rate",kind:.nonnegativeDecimal(maximum:OptimizerConfiguration.scalarMaximum),defaultValue:0.001)]}
    private static func failure(_ message:String,_ address:WorkflowNodeAddress)->WorkflowCompilerError {
        .init(code:.configuration,message:message,node:address,port:nil)
    }
    public static func trainingMapping(_ config:[String:JSONValue],address:WorkflowNodeAddress)throws->([String],String) {
        guard Set(config.keys)==["featureColumns","targetColumn"],case .array(let values)=config["featureColumns"],case .string(let target)=config["targetColumn"] else {throw failure("Map ordered input columns and one target.",address)}
        let columns=values.compactMap {value->String? in if case .string(let v)=value{return v};return nil}
        guard columns.count==values.count else {throw failure("Input column names must be strings.",address)}
        do {try TrainingColumnMapping.validate(features:columns,target:target)}
        catch {throw failure(error.localizedDescription,address)}
        return(columns,target)
    }
    public static func bagInputMapping(_ config:[String:JSONValue],address:WorkflowNodeAddress)throws->BagInputMapping {
        guard Set(config.keys)==["bagInput","targetColumn"] || Set(config.keys)==["bagInput","lossTargetMapping"],
              let value=config["bagInput"] else {throw failure("Bag data needs exactly its packet mapping and either a binary target or explicit objective mapping.",address)}
        do {
            let mapping=try BagInputMapping(value:value)
            if let target=config["targetColumn"] {
                guard case .string(let name)=target else {throw GraphConfigurationError("Choose one binary target column.")}
                try mapping.validateBindings(targetColumn:name,objectiveMapping:nil)
            } else {
                let objective=try lossTargetMapping(config.filter {$0.key != "bagInput"},address:address)
                try mapping.validateBindings(targetColumn:nil,objectiveMapping:objective)
            }
            return mapping
        } catch {throw failure(error.localizedDescription,address)}
    }
    public static func objectiveScalarization(_ config:[String:JSONValue],schemaVersion:Int)throws->WorkflowObjectiveScalarization {
        guard let value=config["objectiveScalarization"] else {return .mean}
        guard schemaVersion>=8,case .string(let raw)=value,let mode=WorkflowObjectiveScalarization(rawValue:raw) else {throw DataFlowContractError("Objective scalarization requires schema8 and Mean or Sum.")}
        return mode
    }
    public static func lossTargetMapping(_ config:[String:JSONValue],address:WorkflowNodeAddress)throws->LossTargetMapping {
        guard Set(config.keys)==["lossTargetMapping"],case .object(let value)=config["lossTargetMapping"],
              Set(value.keys)==["schemaVersion","componentID","inputColumns","roles","tripletSourceGroups"],
              case .array(let roles)=value["roles"] else {throw failure("Bind explicit objective input columns and typed target roles.",address)}
        for role in roles {
            guard case .object(let fields)=role,Set(fields.keys)==["role","columns","sampleShape","dtype"] else {throw failure("Target roles contain missing or unknown fields.",address)}
        }
        do {
            let mapping=try JSONDecoder().decode(LossTargetMapping.self,from:JSONEncoder().encode(JSONValue.object(value)))
            try mapping.validate();return mapping
        } catch {throw failure(error.localizedDescription,address)}
    }
    public static func runSettings(_ config:[String:JSONValue],address:WorkflowNodeAddress)throws->ExecutionSettings {
        if let specification=config["experiment"] {_ = try WorkflowExperimentSpecification.decode(specification)}
        let fields=runFields
        let allowed=Set(fields.map(\.key)).union(["trainingControl","gradientCapture","preparationCachePolicy","experiment"])
        guard Set(config.keys).isSubset(of:allowed) else {throw failure("Unknown run setting.",address)}
        // Inspectors can validate settings for either schema. The document codec/compiler
        // enforce schema ownership; this helper never puts policy in model settings.
        _ = try preparationCachePolicy(config,schemaVersion:2)
        for field in fields {
            if let value=config[field.key], !field.accepts(value) {
                throw failure("\(field.title) has an invalid type or is outside its supported values.",address)
            }
        }
        var result=ExecutionSettings()
        if let value=config["trainingControl"] {
            do {result.trainingControl=try JSONDecoder().decode(TrainingControlPolicy.self,from:JSONEncoder().encode(value))}
            catch {throw failure(error.localizedDescription,address)}
        }
        if case .integer(let value)=config["epochs"] {result.epochs=Int(value)}
        if case .integer(let value)=config["batchSize"] {result.batchSize=Int(value)}
        if case .integer(let value)=config["seed"] {result.seed=Int(value)}
        if case .string(let value)=config["device"] {result.device=value}
        if let value=config["learningRate"]?.doubleValue {result.learningRate=value}
        if let value=config["gradientCapture"],value != .null {
            do {
                result.gradientCapture=try JSONDecoder().decode(GradientCaptureConfiguration.self,from:JSONEncoder().encode(value))
                try result.gradientCapture?.validate()
            } catch {throw failure(error.localizedDescription,address)}
        }
        return result
    }
}
