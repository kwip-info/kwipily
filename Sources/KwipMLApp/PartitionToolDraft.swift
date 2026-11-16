import Foundation
import KwipMLCore

struct PartitionToolDraft:Equatable {
    static let defaults:[String:JSONValue] = ["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20]
    private(set) var strategy:String
    private var drafts:[String:[String:JSONValue]]
    var embargoText:String
    private(set) var embargoUnit="milliseconds"
    private static let units:[String:Decimal] = ["milliseconds":1,"seconds":1000,"minutes":60000,"hours":3600000,"days":86400000]
    init(configuration:[String:JSONValue])throws {
        guard case .string(let mode)=configuration["strategy"],["random","entity","time"].contains(mode) else {throw DataFlowContractError("Choose a Partition strategy.")}
        strategy=mode
        drafts=["random":Self.defaults,"entity":["strategy":.string("entity"),"seed":42,"validationPercent":20,"testPercent":20,"groupBy":.array([])],"time":["strategy":.string("time"),"timeColumn":.string(""),"timeEncoding":.string("iso8601Offset"),"trainingEnd":.string(""),"validationEnd":.string(""),"embargoMilliseconds":0,"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")]]
        drafts[mode]=configuration
        if case .integer(let amount)=configuration["embargoMilliseconds"] {embargoText=String(amount)} else {embargoText="0"}
    }
    func value(_ key:String)->JSONValue? {drafts[strategy]?[key]}
    func text(_ key:String)->String {switch value(key) {case .string(let s):return s;case .integer(let n):return String(n);default:return ""}}
    func names(_ key:String)->[String] {guard case .array(let values)=value(key) else {return []};return values.compactMap {if case .string(let name)=$0 {return name};return nil}}
    mutating func set(_ key:String,_ value:JSONValue) {drafts[strategy]?[key]=value}
    mutating func remove(_ key:String) {drafts[strategy]?[key]=nil}
    mutating func switchStrategy(_ mode:String) {if drafts[mode] != nil {strategy=mode}}
    mutating func setEmbargoUnit(_ unit:String) {
        guard let next=Self.units[unit],let prior=Self.units[embargoUnit],let amount=Decimal(string:embargoText,locale:Locale(identifier:"en_US_POSIX")) else {return}
        embargoText=NSDecimalNumber(decimal:amount*prior/next).stringValue;embargoUnit=unit
    }
    func configuration(schema:[DataColumnSchema])throws->[String:JSONValue] {
        var config=drafts[strategy]!
        func integer(_ key:String)throws->Int64 {
            let text=text(key)
            guard text.range(of:"^-?[0-9]+$",options:.regularExpression) != nil,let value=Int64(text) else {throw DataFlowContractError("Enter a whole number for \(key).")};return value
        }
        func scalar(_ names:[String])throws {
            guard schema.isEmpty || names.allSatisfy({name in schema.contains {$0.name == name && $0.dtype != "image"}}) else {throw DataFlowContractError("Choose existing scalar columns; image values cannot be partition keys.")}
        }
        if strategy != "time" {
            for key in ["seed","validationPercent","testPercent"] {config[key] = .integer(try integer(key))}
            var legacy=config;legacy.removeValue(forKey:"strategy");legacy.removeValue(forKey:"stratifyColumn")
            if strategy == "random" {legacy["groupBy"] = .array([.string("allocation")])} else {try scalar(names("groupBy"))}
            legacy["nullKeys"] = .string("error");legacy["allocationUnit"] = .string("groups")
            _=try PartitionTableConfiguration.validate(legacy,operatorID:"kwip.data.groupSplit",schema:strategy == "random" ? []:schema)
            if let value=config["stratifyColumn"] {
                guard case .string(let name)=value,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count <= 200 else {throw DataFlowContractError("Choose an exact stratification column.")}
                try scalar([name])
            }
        } else {
            guard embargoText.range(of:"^[0-9]+(?:\\.[0-9]+)?$",options:.regularExpression) != nil,let amount=Decimal(string:embargoText,locale:Locale(identifier:"en_US_POSIX")),let unit=Self.units[embargoUnit] else {throw DataFlowContractError("Enter a nonnegative embargo duration.")}
            let milliseconds=amount*unit,number=NSDecimalNumber(decimal:milliseconds)
            guard milliseconds >= 0,milliseconds <= 86400000000,Decimal(number.int64Value) == milliseconds else {throw DataFlowContractError("Embargo must resolve to exact whole milliseconds, at most 1,000 days.")}
            config["embargoMilliseconds"] = .integer(number.int64Value)
            if text("timeEncoding") == "epochMilliseconds" {for key in ["trainingEnd","validationEnd"] {config[key] = .integer(try integer(key))}}
            var legacy=config;legacy.removeValue(forKey:"strategy")
            _=try PartitionTableConfiguration.validate(legacy,operatorID:"kwip.data.temporalSplit",schema:schema)
            try scalar(names("entityColumns"))
        }
        return config
    }
}
