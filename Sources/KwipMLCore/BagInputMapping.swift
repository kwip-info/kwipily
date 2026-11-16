import Foundation

/// Versioned table-to-packet bindings; the actual source owns packet geometry.
public struct BagInputMapping: Codable, Equatable {
    public let schemaVersion: Int
    public let indicesColumns: [String]
    public let weightColumns: [String]?
    public let lengthColumn: String?
    public init(value: JSONValue) throws {
        guard case .object(let fields)=value,
              Set(fields.keys).isSubset(of:["schemaVersion","indicesColumns","weightColumns","lengthColumn"]),
              fields["schemaVersion"] == .integer(1),let indices=fields["indicesColumns"] else {
            throw GraphConfigurationError("Bag mapping requires schema1, ordered item columns and only the applicable weight and length columns.")
        }
        func columns(_ value:JSONValue)throws->[String] {
            guard case .array(let values)=value,(1...128).contains(values.count) else {throw GraphConfigurationError("Map 1–128 ordered bag columns.")}
            let names=try values.map {value->String in
                guard case .string(let name)=value,Self.validName(name) else {throw GraphConfigurationError("Bag column names must be nonblank text of at most 200 characters.")};return name
            }
            guard Set(names).count==names.count else {throw GraphConfigurationError("Bag columns must be distinct.")};return names
        }
        indicesColumns=try columns(indices)
        if let weights=fields["weightColumns"] {
            let names=try columns(weights)
            guard names.count==indicesColumns.count else {throw GraphConfigurationError("Map one weight column for each ordered item column.")}
            weightColumns=names
        } else {weightColumns=nil}
        if let length=fields["lengthColumn"] {
            guard case .string(let name)=length,Self.validName(name) else {throw GraphConfigurationError("Packed bags require one nonblank length column.")};lengthColumn=name
        } else {lengthColumn=nil}
        schemaVersion=1
        let all=indicesColumns+(weightColumns ?? [])+(lengthColumn.map {[$0]} ?? [])
        guard Set(all).count==all.count else {throw GraphConfigurationError("Item, weight and length columns must be separate.")}
    }
    public init(from decoder:Decoder)throws {try self.init(value:JSONValue(from:decoder))}
    public var value: JSONValue {
        var fields:[String:JSONValue]=["schemaVersion":1,"indicesColumns":.array(indicesColumns.map(JSONValue.string))]
        if let weightColumns {fields["weightColumns"] = .array(weightColumns.map(JSONValue.string))}
        if let lengthColumn {fields["lengthColumn"] = .string(lengthColumn)}
        return .object(fields)
    }
    private static func validName(_ value:String)->Bool {!value.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && value.count<=200}
    func validateBindings(targetColumn:String?,objectiveMapping:LossTargetMapping?)throws {
        guard (targetColumn != nil) != (objectiveMapping != nil) else {throw GraphConfigurationError("Choose either one binary target or an explicit objective mapping for bag data.")}
        let packet=Set(indicesColumns+(weightColumns ?? [])+(lengthColumn.map {[$0]} ?? []))
        if let targetColumn {
            guard Self.validName(targetColumn),!packet.contains(targetColumn) else {throw GraphConfigurationError("The target column must be separate from item, weight and length columns.")}
        }
        if let objectiveMapping {
            try objectiveMapping.validate()
            guard objectiveMapping.inputColumns==indicesColumns,
                  packet.isDisjoint(with:objectiveMapping.roles.flatMap(\.columns)) else {
                throw GraphConfigurationError("Objective input columns must match ordered bag items; target roles must be separate from every packet column.")
            }
        }
    }
    public func validate(sourceConfiguration: BagSourceConfiguration, targetColumn: String? = nil,
                         objectiveMapping: LossTargetMapping? = nil) throws {
        guard indicesColumns.count==sourceConfiguration.maximumItemsPerBag,
              (weightColumns != nil)==sourceConfiguration.usesPerSampleWeights,
              (lengthColumn != nil)==(sourceConfiguration.layout == "packed") else {
            throw GraphConfigurationError("Bag mapping must match the captured source's item capacity, weights and packed-length controls.")
        }
        try validateBindings(targetColumn:targetColumn,objectiveMapping:objectiveMapping)
    }
}

/// Conservative prepared admission shared by all actual reachable bag consumers.
public struct BagWorkflowInputRequirement: Equatable {
    public let sourceNodeID: String
    public let configuration: BagSourceConfiguration
    public let numEmbeddings: Int
    public let embeddingDim: Int
    public var preparationMode: String { "sum" }
}
