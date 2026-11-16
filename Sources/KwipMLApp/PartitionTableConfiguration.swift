import Foundation
import KwipMLCore

/// Pure recorded-schema guidance. Worker verifies actual membership and nonempty partitions.
enum PartitionTableConfiguration {
    static func validate(_ config:[String:JSONValue],operatorID:String,schema:[DataColumnSchema])throws->[String:JSONValue] {
        func fail(_ message:String)->DataFlowContractError {DataFlowContractError(message)}
        func names(_ key:String,nonempty:Bool)throws->[String] {
            guard case .array(let values)=config[key],values.count <= 128,!nonempty || !values.isEmpty else {throw fail("Choose ordered \(key) columns.")}
            let names=values.compactMap {if case .string(let name)=$0 {return name};return nil}
            guard names.count == values.count,Set(names).count == names.count,names.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count <= 200}) else {throw fail("Use unique, exact nonblank column names.")}
            guard schema.isEmpty || names.allSatisfy({name in schema.contains {$0.name == name}}) else {throw fail("A selected column is absent from the recorded schema.")}
            return names
        }
        if operatorID == "kwip.data.groupSplit" {
            guard Set(config.keys) == ["groupBy","nullKeys","allocationUnit","seed","validationPercent","testPercent"],config["nullKeys"] == .string("error"),config["allocationUnit"] == .string("groups"),case .integer(let seed)=config["seed"],(0...4294967295).contains(seed),case .integer(let validation)=config["validationPercent"],case .integer(let test)=config["testPercent"],(1...98).contains(validation),(1...98).contains(test),validation+test < 100 else {throw fail("Choose integer group percentages from 1 to 98 totaling less than 100, and a seed from 0 to 4,294,967,295.")}
            _=try names("groupBy",nonempty:true)
        } else if operatorID == "kwip.data.temporalSplit" {
            guard Set(config.keys) == ["timeColumn","timeEncoding","trainingEnd","validationEnd","embargoMilliseconds","entityColumns","entityPolicy"],case .string(let column)=config["timeColumn"],!column.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,column.count <= 200,case .string(let encoding)=config["timeEncoding"],["iso8601Offset","epochMilliseconds"].contains(encoding),case .integer(let embargo)=config["embargoMilliseconds"],(0...86400000000).contains(embargo),case .string(let policy)=config["entityPolicy"],["allowAcrossTime","requireDisjoint"].contains(policy) else {throw fail("Choose a time column, explicit encoding, entity policy and an integer embargo from 0 to 86,400,000,000 milliseconds.")}
            let dtype=encoding == "epochMilliseconds" ? "int64":"string"
            guard schema.isEmpty || schema.contains(where:{$0.name == column && $0.dtype == dtype}) else {throw fail("Time column must use \(dtype) for this encoding. Add an explicit Cast if needed.")}
            _=try names("entityColumns",nonempty:policy == "requireDisjoint")
            guard try instant(config["trainingEnd"],encoding:encoding) < instant(config["validationEnd"],encoding:encoding) else {throw fail("Training cutoff must be strictly before validation cutoff, comparing exact instants.")}
        } else {throw fail("Unsupported partition operator.")}
        return config
    }
    /// Signed microseconds since Unix epoch; integer arithmetic preserves six fractional digits.
    static func instant(_ value:JSONValue?,encoding:String)throws->Int64 {
        let minimum:Int64 = -62135596800000000,maximum:Int64 = 253402300799999999
        if encoding == "epochMilliseconds",case .integer(let milliseconds)=value,milliseconds >= -62135596800000,milliseconds <= 253402300799999 {return milliseconds*1000}
        guard encoding == "iso8601Offset",case .string(let string)=value,string.utf8.count <= 40 else {throw DataFlowContractError("Enter an integer epoch millisecond value or an offset-explicit ISO time in years 0001–9999.")}
        let pattern="^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\\.([0-9]{1,6}))?(Z|[+-][0-9]{2}:[0-9]{2})$"
        let expression=try NSRegularExpression(pattern:pattern),range=NSRange(string.startIndex...,in:string)
        guard let match=expression.firstMatch(in:string,range:range),match.range == range else {throw DataFlowContractError("Use YYYY-MM-DDTHH:MM:SS[.ffffff]Z or an explicit ±HH:MM offset.")}
        func part(_ index:Int)->String {guard let range=Range(match.range(at:index),in:string) else {return ""};return String(string[range])}
        let year=Int64(part(1))!,month=Int(part(2))!,day=Int(part(3))!,hour=Int64(part(4))!,minute=Int64(part(5))!,second=Int64(part(6))!
        let leap=year%4 == 0 && (year%100 != 0 || year%400 == 0),days=[31,leap ? 29:28,31,30,31,30,31,31,30,31,30,31]
        guard (1...9999).contains(year),(1...12).contains(month),day >= 1,day <= days[month-1],hour < 24,minute < 60,second < 60 else {throw DataFlowContractError("Time contains an invalid calendar date or clock value.")}
        let zone=part(8)
        var offset:Int64=0
        if zone != "Z" {
            let components=zone.dropFirst().split(separator:":"),hours=Int64(components[0])!,minutes=Int64(components[1])!
            guard hours < 24,minutes < 60 else {throw DataFlowContractError("Time offset is invalid.")}
            offset=(hours*3600+minutes*60)*(zone.first == "-" ? -1:1)
        }
        let prior=year-1,ordinal=prior*365+prior/4-prior/100+prior/400+Int64(days.prefix(month-1).reduce(0,+)+day-1)-719162
        let fraction=part(7),microseconds=Int64(fraction+String(repeating:"0",count:6-fraction.count)) ?? 0
        let instant=(ordinal*86400+hour*3600+minute*60+second-offset)*1000000+microseconds
        guard (minimum...maximum).contains(instant) else {throw DataFlowContractError("UTC instant must fall within years 0001–9999.")}
        return instant
    }
}
