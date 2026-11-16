import Foundation
import KwipMLCore

struct SummaryReducerDraft:Identifiable,Equatable {
    var id=UUID()
    var output="",op="countRows",column="",nullPolicy="skip"
    var separator="\n",separatorChoice="newline"
    var distinct=false,skipEmpty=false
    static let operations:[(String,String)]=[("countRows","Count rows"),("countNonNull","Count nonmissing"),("sum","Sum"),("sumDecimal","Sum exact decimal text"),("mean","Mean"),("min","Minimum"),("max","Maximum"),("joinText","Join text")]
    var configuration:JSONValue {
        var fields:[String:JSONValue]=["output":.string(output),"op":.string(op)]
        if op != "countRows" {fields["column"] = .string(column)}
        if !["countRows","countNonNull"].contains(op) {fields["nullPolicy"] = .string(nullPolicy)}
        if op == "joinText" {fields["separator"] = .string(separator);fields["distinct"] = .bool(distinct);fields["skipEmpty"] = .bool(skipEmpty)}
        return .object(fields)
    }
    func eligible(_ schema:[DataColumnSchema])->[DataColumnSchema] {
        schema.filter {$0.dtype != "image" && (!["sum","mean"].contains(op) || ["int64","float64"].contains($0.dtype)) && (!["joinText","sumDecimal"].contains(op) || $0.dtype == "string")}
    }
    mutating func setOperation(_ value:String,schema:[DataColumnSchema]) {
        op=value
        if value == "countRows" || (!schema.isEmpty && !eligible(schema).contains(where:{$0.name == column})) {column=""}
    }
    mutating func setSeparatorChoice(_ value:String) {
        separatorChoice=value
        if let separator=Self.separators[value] {self.separator=separator}
    }
    static let separators=["newline":"\n","comma":",","commaSpace":", ","space":" ","tab":"\t","none":""]
    static func decode(_ value:JSONValue)throws->Self {
        guard case .object(let fields)=value,case .string(let output)=fields["output"],case .string(let op)=fields["op"] else {throw DataFlowContractError("A saved summary has invalid settings.")}
        var draft=Self();draft.output=output;draft.op=op
        if case .string(let column)=fields["column"] {draft.column=column}
        if case .string(let policy)=fields["nullPolicy"] {draft.nullPolicy=policy}
        if case .string(let separator)=fields["separator"] {draft.separator=separator;draft.separatorChoice=separators.first(where:{$0.value == separator})?.key ?? "custom"}
        if case .bool(let value)=fields["distinct"] {draft.distinct=value}
        if case .bool(let value)=fields["skipEmpty"] {draft.skipEmpty=value}
        guard draft.configuration == value else {throw DataFlowContractError("A saved summary has unsupported or missing settings.")}
        return draft
    }
}
