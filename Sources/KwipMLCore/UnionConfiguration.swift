import Foundation

/// Stable port IDs belong to connections; their ordered list controls appended rows.
public struct UnionConfiguration:Equatable {
    public var inputs:[String]
    public var match:String
    public var coverage:String
    public var typePolicy:String
    public var sourceColumn:String
    public static let defaults:[String:JSONValue]=["inputs":.array([.string("table1"),.string("table2")]),"match":.string("name"),"coverage":.string("all"),"typePolicy":.string("strict"),"sourceColumn":.string("")]
    public init(_ config:[String:JSONValue])throws {
        guard Set(config.keys)==Set(Self.defaults.keys),case .array(let values)=config["inputs"],values.allSatisfy({if case .string=$0 {return true};return false}),case .string(let match)=config["match"],case .string(let coverage)=config["coverage"],case .string(let type)=config["typePolicy"],case .string(let source)=config["sourceColumn"] else {throw DataFlowContractError("Choose Union input order, matching, coverage, types and identifier settings.")}
        inputs=values.compactMap {if case .string(let id)=$0 {return id};return nil}
        guard (2...32).contains(inputs.count),Set(inputs).count==inputs.count,inputs.allSatisfy({$0.range(of:"\\Atable[1-9][0-9]{0,5}\\z",options:.regularExpression) != nil}),["name","position"].contains(match),["all","shared","exact"].contains(coverage),["strict","losslessNumeric"].contains(type),source.unicodeScalars.count<=200,source.isEmpty || !source.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {throw DataFlowContractError("Union needs 2–32 unique table ports and valid matching/type policies.")}
        self.match=match;self.coverage=coverage;typePolicy=type;sourceColumn=source
    }
    public var values:[String:JSONValue] {["inputs":.array(inputs.map(JSONValue.string)),"match":.string(match),"coverage":.string(coverage),"typePolicy":.string(typePolicy),"sourceColumn":.string(sourceColumn)]}
    public static func title(_ port:String)->String {"Table " + port.dropFirst(5)}
    public func nextPort()->String? {
        guard inputs.count<32,let maximum=inputs.compactMap({Int($0.dropFirst(5))}).max(),maximum<999999 else {return nil}
        return "table\(maximum+1)"
    }
}
