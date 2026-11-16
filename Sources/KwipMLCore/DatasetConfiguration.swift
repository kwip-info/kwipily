import Foundation

/// A training-only view over the verified prepared source, not a new source authority.
public struct DatasetConfiguration:Equatable,Sendable {
    public static let componentID="dataset_source"
    public static let defaults:[String:JSONValue]=["kind":.string("indexed"),"indices":.array([]),"max_rows":100000]
    public let values:[String:JSONValue]
    public var kind:String {values["kind"]!.stringValue!}
    public var indices:[Int] {if case .array(let values)=values["indices"] {return values.compactMap(\.integerValue)};return []}
    public var maximumRows:Int {values["max_rows"]!.integerValue!}
    public var recipeValue:JSONValue {.object(["schemaVersion":1,"componentID":.string(Self.componentID),"configurationVersion":1,"config":.object(values)])}
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==Set(Self.defaults.keys),let kind=values["kind"]?.stringValue,["indexed","boundedIterable","tensor","subset"].contains(kind),
              case .array(let raw)=values["indices"],case .integer(let maximum)=values["max_rows"],(1...100000).contains(maximum) else {
            throw GraphConfigurationError("Choose Indexed rows, Finite stream, Tensor tuple or Selected indices, with exact ordered indices and a whole-number row limit from 1 to 100,000.")
        }
        let indices=try raw.map {value->Int64 in
            guard case .integer(let index)=value,(0..<100000).contains(index) else {throw GraphConfigurationError("Selected indices are whole-number positions from 0 to 99,999 in the original training partition.")};return index
        }
        guard indices.count<=100000,Set(indices).count==indices.count else {throw GraphConfigurationError("Choose unique selected indices; repeated sampling is unavailable in this release.")}
        guard kind == "subset" ? !indices.isEmpty:indices.isEmpty else {throw GraphConfigurationError("Selected-index views require at least one position; other source kinds require an empty index list.")}
        guard kind == "boundedIterable" || maximum == 100000 else {throw GraphConfigurationError("The row limit configures Finite stream only; other source kinds retain the 100,000-row ceiling.")}
        self.values=values
        guard try JSONEncoder().encode(recipeValue).count<=32768 else {throw GraphConfigurationError("Dataset recipe exceeds its 32 KiB transport bound; select fewer explicit positions.")}
    }
    public static func decodeRecipe(_ value:JSONValue)throws->Self {
        guard case .object(let envelope)=value,Set(envelope.keys)==["schemaVersion","componentID","configurationVersion","config"],envelope["schemaVersion"] == .integer(1),envelope["configurationVersion"] == .integer(1),envelope["componentID"] == .string(componentID),case .object(let config)=envelope["config"] else {throw GraphConfigurationError("Use the exact version1 Dataset recipe and declared configuration.")}
        return try .init(values:config)
    }
}
public struct DatasetRecipe:Equatable {
    public let configuration:DatasetConfiguration
    public let source:WorkflowNodeAddress
    public init(configuration:DatasetConfiguration,source:WorkflowNodeAddress) {self.configuration=configuration;self.source=source}
    public func validate()throws {
        guard source.domain == .control,!source.localID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,source.localID.utf8.count<=128 else {throw GraphConfigurationError("Dataset recipes require a distinct, bounded control-node origin.")}
        _=try DatasetConfiguration(values:configuration.values)
    }
}
