import Foundation

/// Ordered native batching policy; source snapshots and device transfers remain separate.
public struct DataLoaderConfiguration:Equatable,Sendable {
    public static let componentID="data_loader"
    public static let defaults:[String:JSONValue]=["batch_size":32,"shuffle":false,"drop_last":false,"seed":42,"num_workers":0,"prefetch_factor":.null,"timeout":.number(0),"collation":.string("auto")]
    public let values:[String:JSONValue]
    public var batchSize:Int {values["batch_size"]!.integerValue!}
    public var shuffle:Bool {values["shuffle"]!.boolValue!}
    public var dropLast:Bool {values["drop_last"]!.boolValue!}
    public var seed:Int64 {if case .integer(let value)=values["seed"] {return value};return 0}
    public var numberOfWorkers:Int {values["num_workers"]!.integerValue!}
    public var prefetchFactor:Int? {values["prefetch_factor"]?.integerValue}
    public var timeoutSeconds:Double {values["timeout"]!.doubleValue!}
    public var collation:String {values["collation"]!.stringValue!}
    public var recipeValue:JSONValue {.object(["schemaVersion":1,"componentID":.string(Self.componentID),"configurationVersion":1,"config":.object(values)])}
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==Set(Self.defaults.keys),case .integer(let batch)=values["batch_size"],(1...1024).contains(batch),
              values["shuffle"]?.boolValue != nil,values["drop_last"]?.boolValue != nil,case .integer(let seed)=values["seed"],seed>=0,
              case .integer(let workers)=values["num_workers"],(0...2).contains(workers),let timeout=values["timeout"]?.doubleValue,timeout.isFinite,
              let collation=values["collation"]?.stringValue,["auto","fixed_stack","padded_sequence","whole_bags"].contains(collation) else {
            throw GraphConfigurationError("Use the exact batching controls: 1–1024 rows, Boolean mixing/final-group switches, a nonnegative integer seed, 0–2 reader processes, finite timeout and a named collation policy.")
        }
        if workers == 0 {
            guard values["prefetch_factor"] == .null,timeout == 0 else {throw GraphConfigurationError("With zero reader processes, Prefetch is unset and Timeout is zero.")}
        } else {
            guard case .integer(let prefetch)=values["prefetch_factor"],(1...2).contains(prefetch),timeout>0,timeout<=30 else {throw GraphConfigurationError("Separate reader processes require Prefetch 1–2 and a positive timeout of at most 30 seconds.")}
        }
        var normalized=values;normalized["timeout"] = .number(timeout);self.values=normalized
    }
    public static func decodeRecipe(_ value:JSONValue)throws->Self {
        guard case .object(let envelope)=value,Set(envelope.keys)==["schemaVersion","componentID","configurationVersion","config"],envelope["schemaVersion"] == .integer(1),envelope["configurationVersion"] == .integer(1),envelope["componentID"] == .string(componentID),case .object(let config)=envelope["config"] else {throw GraphConfigurationError("Use the exact version1 DataLoader recipe and declared configuration.")}
        return try .init(values:config)
    }
}
public struct DataLoaderRecipe:Equatable {
    public let configuration:DataLoaderConfiguration
    public let source:WorkflowNodeAddress
    public init(configuration:DataLoaderConfiguration,source:WorkflowNodeAddress) {self.configuration=configuration;self.source=source}
    public func validate()throws {
        guard source.domain == .control,!source.localID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,source.localID.utf8.count<=128 else {throw GraphConfigurationError("DataLoader recipes require a distinct, bounded control-node origin.")}
        _=try DataLoaderConfiguration(values:configuration.values)
    }
}
