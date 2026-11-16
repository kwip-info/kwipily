import Foundation

/// Explicit inference-program export controls; an artifact action, never a tensor layer.
public struct ProgramExportConfiguration:Equatable,Sendable {
    public static let componentID="program_export"
    public static let defaults:[String:JSONValue]=["strict":true,"batch_mode":.string("static"),"example_batch_size":2,"minimum_batch_size":2,"maximum_batch_size":2,"device":.string("cpu"),"prefer_deferred_runtime_asserts":false]
    public let values:[String:JSONValue]
    public var batchMode:String {values["batch_mode"]!.stringValue!}
    public var exampleBatchSize:Int {values["example_batch_size"]!.integerValue!}
    public var minimumBatchSize:Int {values["minimum_batch_size"]!.integerValue!}
    public var maximumBatchSize:Int {values["maximum_batch_size"]!.integerValue!}
    public var device:String {values["device"]!.stringValue!}
    public var strict:Bool {values["strict"]!.boolValue!}
    public var preferDeferredRuntimeAsserts:Bool {values["prefer_deferred_runtime_asserts"]!.boolValue!}
    public var recipeValue:JSONValue {.object(["schemaVersion":1,"componentID":.string(Self.componentID),"configurationVersion":1,"config":.object(values)])}
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==Set(Self.defaults.keys),values["strict"]?.boolValue != nil,values["prefer_deferred_runtime_asserts"]?.boolValue != nil,
              let mode=values["batch_mode"]?.stringValue,["static","bounded"].contains(mode),let device=values["device"]?.stringValue,["cpu","mps"].contains(device),
              case .integer(let example)=values["example_batch_size"],(1...128).contains(example),case .integer(let minimum)=values["minimum_batch_size"],(1...4096).contains(minimum),case .integer(let maximum)=values["maximum_batch_size"],(1...4096).contains(maximum) else {
            throw GraphConfigurationError("Use the seven declared program-export controls, exact switches, CPU or MPS, example size 1–128 and batch bounds 1–4096.")
        }
        if mode == "static" {
            guard minimum == example,maximum == example else {throw GraphConfigurationError("Static export requires example, minimum and maximum batch sizes to match.")}
        } else {
            guard minimum<maximum,minimum<=example,example<=maximum,example>=2 else {throw GraphConfigurationError("Bounded export needs distinct ordered bounds, an example within them and an example batch of at least two observations.")}
        }
        self.values=values
    }
    public static func decodeRecipe(_ value:JSONValue)throws->Self {
        guard case .object(let envelope)=value,Set(envelope.keys)==["schemaVersion","componentID","configurationVersion","config"],envelope["schemaVersion"] == .integer(1),envelope["configurationVersion"] == .integer(1),envelope["componentID"] == .string(componentID),case .object(let config)=envelope["config"] else {throw GraphConfigurationError("Use the exact version1 executable-program export recipe and declared configuration.")}
        return try .init(values:config)
    }
}
