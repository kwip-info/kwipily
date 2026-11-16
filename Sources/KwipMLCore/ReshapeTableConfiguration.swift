import Foundation

public struct ReshapeTableConfiguration:Equatable {
    public static let defaults:[String:JSONValue]=[
        "mode":.string("long"),"idColumns":.array([]),"valueColumns":.array([]),"fieldName":.string("field"),"valueName":.string("value"),"typePolicy":.string("strict"),"missingValues":.string("keep"),
        "rowKeys":.array([]),"columnKey":.string(""),"values":.array([]),"duplicates":.string("stop"),"nullColumnNames":.string("stop"),"nullLabel":.string("missing"),"missingCombinations":.string("missing"),"fillValue":.null,"columnOrder":.string("firstSeen"),"maxColumns":.integer(128)]
    public let values:[String:JSONValue]
    public static let reducers=["first","last","countRows","countNonNull","sum","mean","min","max"]
    public init(_ config:[String:JSONValue])throws {
        guard Set(config.keys)==Set(Self.defaults.keys) else {throw DataFlowContractError("Reshape Table needs explicit Long and Wide settings.")}
        func text(_ key:String)->String? {if case .string(let value)=config[key] {return value};return nil}
        func validName(_ name:String)->Bool {!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && name.unicodeScalars.count<=200}
        for key in ["idColumns","valueColumns","rowKeys"] {
            guard case .array(let list)=config[key],list.count<=128 else {throw DataFlowContractError("Choose unique ordered columns.")}
            let names=list.compactMap {if case .string(let s)=$0 {return s};return nil}
            guard names.count==list.count,Set(names).count==names.count,names.allSatisfy(validName) else {throw DataFlowContractError("Choose unique ordered columns.")}
        }
        for (key,allowed) in [("mode",["long","wide"]),("typePolicy",["strict","losslessNumeric"]),("missingValues",["keep","drop"]),("duplicates",["stop","aggregate"]),("nullColumnNames",["stop","drop","label"]),("missingCombinations",["missing","fill"]),("columnOrder",["firstSeen","sorted"])] {
            guard let value=text(key),allowed.contains(value) else {throw DataFlowContractError("Choose a valid \(key) policy.")}
        }
        guard ["fieldName","valueName","nullLabel"].allSatisfy({text($0).map(validName) == true}),let key=text("columnKey"),key.unicodeScalars.count<=200,case .integer(let limit)=config["maxColumns"],(1...128).contains(limit),case .array(let fields)=config["values"],fields.count<=128 else {throw DataFlowContractError("Name outputs and choose a column limit from 1 to 128.")}
        for field in fields {
            guard case .object(let row)=field,Set(row.keys)==Set(["column","aggregation","prefix","nullPolicy"]),case .string(let column)=row["column"],validName(column),case .string(let reducer)=row["aggregation"],Self.reducers.contains(reducer),case .string(let prefix)=row["prefix"],prefix.unicodeScalars.count<=200,case .string(let policy)=row["nullPolicy"],["skip","propagate","error"].contains(policy) else {throw DataFlowContractError("Choose each Wide value, reducer, prefix and missing policy.")}
        }
        values=config
    }
    public func string(_ key:String)->String {if case .string(let value)=values[key] {return value};return ""}
    public func columns(_ key:String)->[String] {if case .array(let a)=values[key] {return a.compactMap {if case .string(let s)=$0 {return s};return nil}};return []}
}
