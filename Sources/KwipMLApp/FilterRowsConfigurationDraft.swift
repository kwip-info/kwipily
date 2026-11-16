import Foundation
import KwipMLCore

/// Local atomic condition-editor draft. No source reads or implicit value conversion.
struct FilterRowsConfigurationDraft: Equatable {
    struct Condition: Equatable, Identifiable {
        var id = UUID()
        var column: String
        var comparison: String
        var valueType: String
        var valueText: String
        var caseSensitive: Bool
        var valueFormat = "lines"
        var hasOperand: Bool { !["isMissing","isNotMissing"].contains(comparison) }
        var isList: Bool { ["in","notIn"].contains(comparison) }
        func literal() throws -> JSONValue {
            if !hasOperand { return .null }
            func scalar(_ text: String) throws -> JSONValue {
                switch valueType {
                case "text": return .string(text)
                case "boolean":
                    guard ["true","false"].contains(text) else {throw DataFlowContractError("Enter true or false for a Boolean value.")}
                    return .bool(text == "true")
                case "number":
                    if let value = Int64(text.trimmingCharacters(in:.whitespaces)) {return .integer(value)}
                    guard let value = Double(text.trimmingCharacters(in:.whitespaces)),value.isFinite else {throw DataFlowContractError("Enter a finite number.")}
                    return .number(value)
                default: throw DataFlowContractError("Choose a supported value type.")
                }
            }
            if isList && valueFormat == "json" {
                guard let data=valueText.data(using:.utf8),case .array(let values)=try JSONDecoder().decode(JSONValue.self,from:data),(1...256).contains(values.count) else {throw DataFlowContractError("Enter a JSON array of 1–256 typed values.")}
                for value in values {
                    switch (valueType,value) {
                    case ("text",.string), ("boolean",.bool), ("number",.integer), ("number",.number):break
                    default:throw DataFlowContractError("Every list value must match the selected type.")
                    }
                }
                return .array(values)
            }
            if isList {
                let values = valueText.components(separatedBy:"\n")
                guard (1...256).contains(values.count) else {throw DataFlowContractError("Use at most 256 values, one per line.")}
                return .array(try values.map(scalar))
            }
            return try scalar(valueText)
        }
    }
    static let comparisons:[(String,String)] = [("eq","Equals"),("ne","Does not equal"),("lt","Less than"),("le","Less than or equal"),("gt","Greater than"),("ge","Greater than or equal"),("contains","Contains"),("notContains","Does not contain"),("startsWith","Starts with"),("endsWith","Ends with"),("in","Is one of"),("notIn","Is not one of"),("isMissing","Is missing"),("isNotMissing","Is not missing")]
    static let defaults:[String:JSONValue] = ["mode":.string("simple"),"conditions":.array([.object(["column":.string("feature"),"comparison":.string("gt"),"value":.integer(0),"caseSensitive":.bool(true)])]),"match":.string("all"),"expression":.string("[feature] > 0"),"column":.string("feature"),"pattern":.string(".+"),"matchMode":.string("search"),"flags":.array([]),"nullPolicy":.string("error")]
    var mode:String
    var conditions:[Condition]
    var match:String
    var expression:String
    var column:String
    var pattern:String
    var matchMode:String
    var flags:[String]
    var nullPolicy:String
    init(configuration:[String:JSONValue]) throws {
        guard Set(configuration.keys) == Set(Self.defaults.keys),
              case .string(let mode) = configuration["mode"],["simple","expression","pattern"].contains(mode),
              case .array(let rows) = configuration["conditions"],rows.count <= 32,
              case .string(let match) = configuration["match"],["all","any"].contains(match),
              case .string(let expression) = configuration["expression"],expression.count <= 4096,
              case .string(let column) = configuration["column"],!column.isEmpty,column.count <= 200,
              case .string(let pattern) = configuration["pattern"],pattern.count <= 512,
              case .string(let matchMode) = configuration["matchMode"],["search","full"].contains(matchMode),
              case .array(let flagValues) = configuration["flags"],
              case .string(let nullPolicy) = configuration["nullPolicy"],["error","nonmatching"].contains(nullPolicy) else {throw DataFlowContractError("Filter configuration has unsupported fields or choices.")}
        let flags = flagValues.compactMap(\.stringValue)
        guard flags.count == flagValues.count,Set(flags).count == flags.count,Set(flags).isSubset(of:["ignoreCase","multiline","dotAll"]) else {throw DataFlowContractError("Choose supported pattern flags once each.")}
        self.mode=mode;self.match=match;self.expression=expression;self.column=column;self.pattern=pattern;self.matchMode=matchMode;self.flags=flags;self.nullPolicy=nullPolicy
        conditions = try rows.map { value in
            guard case .object(let row) = value,Set(row.keys) == ["column","comparison","value","caseSensitive"],case .string(let column)=row["column"],!column.isEmpty,column.count<=200,case .string(let comparison)=row["comparison"],Self.comparisons.contains(where:{$0.0==comparison}),let literal=row["value"],case .bool(let sensitive)=row["caseSensitive"] else {throw DataFlowContractError("Each condition needs a column, operator, typed value and text-case choice.")}
            let values:[JSONValue];if case .array(let items)=literal {values=items}else {values=[literal]}
            let type:String
            switch values.first {case .string?:type="text";case .bool?:type="boolean";case .integer?,.number?:type="number";case .null?:type="text";default:throw DataFlowContractError("Use scalar condition values.")}
            let missing = ["isMissing","isNotMissing"].contains(comparison)
            guard missing ? literal == .null : !values.isEmpty else {throw DataFlowContractError("Missing-value conditions use null; membership needs values.")}
            if !missing {
                for item in values {
                    switch (type,item) {
                    case ("text",.string), ("boolean",.bool), ("number",.integer), ("number",.number):break
                    default:throw DataFlowContractError("Use a single consistent type for condition values.")
                    }
                }
                if ["in","notIn"].contains(comparison) {
                    guard case .array = literal,values.count <= 256 else {throw DataFlowContractError("Membership requires 1–256 typed values.")}
                } else if case .array = literal {throw DataFlowContractError("Choose one scalar comparison value.")}
            }
            let texts = try values.map { item -> String in
                switch item {case .string(let v):return v;case .bool(let v):return v ? "true":"false";case .integer(let v):return String(v);case .number(let v):return String(v);case .null:return "";default:throw DataFlowContractError("Use scalar condition values.")}
            }
            var condition=Condition(column:column,comparison:comparison,valueType:type,valueText:texts.joined(separator:"\n"),caseSensitive:sensitive)
            if ["in","notIn"].contains(comparison),texts.contains(where:{$0.contains("\n")}) {
                condition.valueFormat="json"
                condition.valueText=String(decoding:try JSONEncoder().encode(literal),as:UTF8.self)
            }
            return condition
        }
    }
    func validatedConfiguration(schema:[DataColumnSchema]) throws -> [String:JSONValue] {
        guard conditions.count <= 32,mode != "simple" || !conditions.isEmpty else {throw DataFlowContractError("Add at least one condition (maximum 32).")}
        if mode == "expression" && expression.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {throw DataFlowContractError("Enter a Boolean expression.")}
        if mode == "pattern" {
            guard !pattern.isEmpty else {throw DataFlowContractError("Enter a text pattern.")}
            if !schema.isEmpty {guard schema.contains(where:{$0.name==column && $0.dtype=="string"}) else {throw DataFlowContractError("Pattern matching needs an existing Text column.")}}
        }
        let rows = try conditions.map { row -> JSONValue in
            guard !row.column.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,row.column.count <= 200 else {throw DataFlowContractError("Choose an exact condition column.")}
            let literal = try row.literal()
            if mode == "simple",!schema.isEmpty {
                guard let source=schema.first(where:{$0.name==row.column}) else {throw DataFlowContractError("Condition column \(row.column) is missing upstream.")}
                if row.hasOperand {
                    let expected = ["int64","float64"].contains(source.dtype) ? "number":source.dtype == "bool" ? "boolean":source.dtype == "string" ? "text":"image"
                    guard expected == row.valueType else {throw DataFlowContractError("\(row.column) requires a \(expected) value. Filter does not cast columns.")}
                    if ["contains","notContains","startsWith","endsWith"].contains(row.comparison),expected != "text" {throw DataFlowContractError("This operator requires a Text column.")}
                    if expected == "boolean",!["eq","ne","in","notIn"].contains(row.comparison) {throw DataFlowContractError("Boolean columns support equality, membership or missing-value checks.")}
                }
            }
            return .object(["column":.string(row.column),"comparison":.string(row.comparison),"value":literal,"caseSensitive":.bool(row.caseSensitive)])
        }
        let result:[String:JSONValue] = ["mode":.string(mode),"conditions":.array(rows),"match":.string(match),"expression":.string(expression),"column":.string(column),"pattern":.string(pattern),"matchMode":.string(matchMode),"flags":.array(flags.map(JSONValue.string)),"nullPolicy":.string(nullPolicy)]
        _ = try Self(configuration:result)
        return result
    }
}
