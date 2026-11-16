import Foundation

public struct SortConfiguration:Equatable {
    public struct Key:Equatable {
        public var column:String
        public var direction:String
        public var missing:String
        public var comparison:String
        public init(column:String,direction:String="ascending",missing:String="last",comparison:String="exact") {self.column=column;self.direction=direction;self.missing=missing;self.comparison=comparison}
        public var value:JSONValue {.object(["column":.string(column),"direction":.string(direction),"missing":.string(missing),"comparison":.string(comparison)])}
    }
    public var keys:[Key]
    public static let defaults:[String:JSONValue]=["keys":.array([])]
    public init(_ values:[String:JSONValue])throws {
        guard Set(values.keys)==["keys"],case .array(let rows)=values["keys"],(1...128).contains(rows.count) else {throw DataFlowContractError("Choose at least one Sort key.")}
        keys=try rows.map {row in
            guard case .object(let r)=row,Set(r.keys)==["column","direction","missing","comparison"],case .string(let column)=r["column"],case .string(let direction)=r["direction"],case .string(let missing)=r["missing"],case .string(let comparison)=r["comparison"],!column.isEmpty,column.unicodeScalars.count<=200,["ascending","descending"].contains(direction),["first","last"].contains(missing),["exact","casefold"].contains(comparison) else {throw DataFlowContractError("Invalid Sort key settings.")}
            return Key(column:column,direction:direction,missing:missing,comparison:comparison)
        }
        guard Set(keys.map(\.column)).count==keys.count else {throw DataFlowContractError("Sort keys must be unique.")}
    }
    public var values:[String:JSONValue] {["keys":.array(keys.map(\.value))]}
}

public struct UniqueConfiguration:Equatable {
    public var keys:[String]
    public var keep:String
    public var missing:String
    public var comparison:String
    public static let defaults:[String:JSONValue]=["keys":.array([]),"keep":.string("first"),"missing":.string("equal"),"comparison":.string("exact")]
    public init(_ values:[String:JSONValue])throws {
        guard Set(values.keys)==Set(Self.defaults.keys),case .array(let rows)=values["keys"],(1...128).contains(rows.count),case .string(let keep)=values["keep"],case .string(let missing)=values["missing"],case .string(let comparison)=values["comparison"] else {throw DataFlowContractError("Choose at least one Unique key and valid policies.")}
        keys=try rows.map {row in guard case .string(let key)=row,!key.isEmpty,key.unicodeScalars.count<=200 else {throw DataFlowContractError("Invalid Unique key.")};return key}
        guard Set(keys).count==keys.count,["first","last"].contains(keep),["equal","distinct","stop"].contains(missing),["exact","casefold"].contains(comparison) else {throw DataFlowContractError("Invalid Unique keys or policies.")}
        self.keep=keep;self.missing=missing;self.comparison=comparison
    }
    public var values:[String:JSONValue] {["keys":.array(keys.map(JSONValue.string)),"keep":.string(keep),"missing":.string(missing),"comparison":.string(comparison)]}
}
