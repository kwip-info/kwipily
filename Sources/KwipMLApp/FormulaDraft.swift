import Foundation
import KwipMLCore

struct FormulaGuidedRegex:Equatable {
    var kind:String,column:String,pattern="",replacement="",replaceMode="all",matchMode="search",noMatchPolicy="null"
    var flags:[String]=[],groups:[String]=["1"]
    init(kind:String,column:String) {self.kind=kind;self.column=column;pattern=kind == "extract" ? "([A-Za-z]+)":kind == "replace" ? "\\s+":"^ready$";replacement=kind == "replace" ? " ":""}
    static func quote(_ text:String)->String {String(data:try! JSONEncoder().encode(text),encoding:.utf8)!}
    var expression:String {
        let arguments:[String]
        switch kind {
        case "replace":arguments=[pattern,replacement,replaceMode,flags.joined(separator:"|")]
        case "extract":arguments=[pattern,matchMode,noMatchPolicy,flags.joined(separator:"|")]+groups
        default:arguments=[pattern,matchMode,flags.joined(separator:"|")]
        }
        return "regex_\(kind)([\(column)], "+arguments.map(Self.quote).joined(separator:", ")+")"
    }
    init?(expression:String) {
        let text=expression.trimmingCharacters(in:.whitespacesAndNewlines)
        guard let open=text.firstIndex(of:"("),text.hasSuffix(")"),["regex_replace","regex_extract","regex_match"].contains(String(text[..<open])) else {return nil}
        kind=String(text[..<open]).replacingOccurrences(of:"regex_",with:"")
        var tail=String(text[text.index(after:open)..<text.index(before:text.endIndex)]).trimmingCharacters(in:.whitespacesAndNewlines)
        guard tail.first == "[",let end=tail.firstIndex(of:"]") else {return nil}
        column=String(tail[tail.index(after:tail.startIndex)..<end]);tail=String(tail[tail.index(after:end)...]).trimmingCharacters(in:.whitespacesAndNewlines)
        guard tail.first == "," else {return nil};tail.removeFirst()
        var values:[String]=[]
        while !tail.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {
            tail=tail.trimmingCharacters(in:.whitespacesAndNewlines)
            guard tail.first == "\"" else {return nil}
            var escaped=false,closing:String.Index?
            for i in tail.indices.dropFirst() {let c=tail[i];if escaped {escaped=false} else if c == "\\" {escaped=true} else if c == "\"" {closing=i;break}}
            guard let closing,let bytes=String(tail[...closing]).data(using:.utf8),let value=try? JSONDecoder().decode(String.self,from:bytes) else {return nil}
            values.append(value);tail=String(tail[tail.index(after:closing)...]).trimmingCharacters(in:.whitespacesAndNewlines)
            if !tail.isEmpty {guard tail.first == "," else {return nil};tail.removeFirst();if tail.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {return nil}}
        }
        switch kind {
        case "replace":guard values.count == 4 else {return nil};pattern=values[0];replacement=values[1];replaceMode=values[2];flags=values[3].isEmpty ? []:values[3].components(separatedBy:"|")
        case "extract":guard (5...36).contains(values.count) else {return nil};pattern=values[0];matchMode=values[1];noMatchPolicy=values[2];flags=values[3].isEmpty ? []:values[3].components(separatedBy:"|");groups=Array(values.dropFirst(4))
        default:guard values.count == 3 else {return nil};pattern=values[0];matchMode=values[1];flags=values[2].isEmpty ? []:values[2].components(separatedBy:"|")
        }
    }
}

struct FormulaDraft:Equatable {
    struct Output:Equatable {var name:String,dtype:String,write="new"}
    struct Row:Equatable,Identifiable {
        var id:String,expression:String,outputs:[Output],nullPolicy="preserve"
        var guided:FormulaGuidedRegex? {get {FormulaGuidedRegex(expression:expression)} set {if let newValue {expression=newValue.expression}}}
        var temporal:FormulaGuidedTemporal? {get {FormulaGuidedTemporal(expression:expression)} set {if let newValue {expression=newValue.expression}}}
        var kind:String {temporal?.kind ?? guided?.kind ?? "expression"}
        var title:String {if let temporal {return temporal.title};switch kind {case "replace":return "Replace text";case "extract":return "Extract fields";case "match":return "Match pattern";default:return "Expression"}}
    }
    static let defaults:[String:JSONValue] = ["operations":.array([.object(["id":.string("formula-1"),"expression":.string("0"),"outputs":.array([.object(["name":.string("calculated"),"dtype":.string("int64"),"write":.string("new")])]),"nullPolicy":.string("preserve")])])]
    var rows:[Row]
    init(configuration:[String:JSONValue])throws {
        guard Set(configuration.keys)==["operations"],case .array(let operations)=configuration["operations"],(1...64).contains(operations.count) else {throw DataFlowContractError("Formula needs 1–64 operations.")}
        rows=try operations.map {value in
            guard case .object(let op)=value,Set(op.keys)==["id","expression","outputs","nullPolicy"],case .string(let id)=op["id"],case .string(let expression)=op["expression"],case .string(let policy)=op["nullPolicy"],case .array(let outputs)=op["outputs"],(1...32).contains(outputs.count) else {throw DataFlowContractError("Invalid Formula operation.")}
            let mapped=try outputs.map {value->Output in
                guard case .object(let out)=value,Set(out.keys)==["name","dtype","write"],case .string(let name)=out["name"],case .string(let dtype)=out["dtype"],case .string(let write)=out["write"] else {throw DataFlowContractError("Invalid Formula output mapping.")}
                return .init(name:name,dtype:dtype,write:write)
            }
            return Row(id:id,expression:expression,outputs:mapped,nullPolicy:policy)
        }
        try validateStructure()
    }
    var configuration:[String:JSONValue] {["operations":.array(rows.map {row in .object(["id":.string(row.id),"expression":.string(row.expression),"outputs":.array(row.outputs.map{.object(["name":.string($0.name),"dtype":.string($0.dtype),"write":.string($0.write)])}),"nullPolicy":.string(row.nullPolicy)])})]}
    static func validName(_ name:String)->Bool {!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && name.count<=200 && !name.contains("\0")}
    func validateStructure()throws {
        guard (1...64).contains(rows.count),Set(rows.map(\.id)).count==rows.count,rows.allSatisfy({Self.validName($0.id) && (1...4096).contains($0.expression.count) && !$0.expression.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && ["preserve","error"].contains($0.nullPolicy) && (1...32).contains($0.outputs.count) && $0.outputs.allSatisfy({Self.validName($0.name) && ["string","int64","float64","bool","date","datetime","time"].contains($0.dtype) && ["new","replace"].contains($0.write)})}) else {throw DataFlowContractError("Use bounded expressions, exact output names, scalar types and explicit write targets.")}
        let names=rows.flatMap{$0.outputs.map(\.name)}
        guard Set(names).count==names.count else {throw DataFlowContractError("Output names must be unique across operations.")}
        for row in rows {
            if let temporal=row.temporal {
                guard row.outputs.count == 1, row.outputs[0].dtype == temporal.outputType, Self.validName(temporal.column), temporal.kind != "temporal_difference" || Self.validName(temporal.otherColumn), ["date","datetime","time"].contains(temporal.dtype), ["stop","missing"].contains(temporal.invalid), ["stop","earliest","latest","missing"].contains(temporal.ambiguous), ["stop","missing"].contains(temporal.nonexistent), ["integer","decimal"].contains(temporal.representation), (1...200).contains(temporal.pattern.count) else {throw DataFlowContractError("Choose a temporal source, exact output type, pattern and explicit policies.")}
                if temporal.kind != "temporal_parse" || temporal.dtype == "datetime" {
                    guard TimeZone(identifier:temporal.timezone) != nil else {throw DataFlowContractError("Choose a valid IANA result timezone.")}
                }
                if temporal.kind == "temporal_parse", temporal.dtype == "datetime", !temporal.sourceTimezone.isEmpty, TimeZone(identifier:temporal.sourceTimezone) == nil {throw DataFlowContractError("Choose a valid IANA source timezone.")}
            } else if let guided=row.guided {
                guard Self.validName(guided.column),(1...512).contains(guided.pattern.count),Set(guided.flags).count==guided.flags.count,Set(guided.flags).isSubset(of:["ignoreCase","multiline","dotAll"]),["search","full"].contains(guided.matchMode),["first","all"].contains(guided.replaceMode),["null","error"].contains(guided.noMatchPolicy),guided.replacement.count<=4096 else {throw DataFlowContractError("Choose valid regex controls; syntax is checked by the isolated runner.")}
                guard row.outputs.count == (guided.kind == "extract" ? guided.groups.count:1),row.outputs.allSatisfy({$0.dtype == (guided.kind == "match" ? "bool":"string")}),guided.groups.allSatisfy(Self.validName) else {throw DataFlowContractError("Capture mappings and output types must match the regex operation.")}
            } else if row.outputs.count != 1 {throw DataFlowContractError("Only Extract fields can produce multiple columns.")}
        }
    }
    /// Guidance only: skip quoted literals when finding named column references.
    static func references(_ expression:String)->[String] {
        let pattern="\\\"(?:[^\\\"\\\\]|\\\\.)*\\\"|\\[([^\\]\\n]{1,200})\\]"
        guard let regex=try? NSRegularExpression(pattern:pattern) else {return []}
        return regex.matches(in:expression,range:NSRange(expression.startIndex...,in:expression)).compactMap {match in guard match.range(at:1).location != NSNotFound,let range=Range(match.range(at:1),in:expression) else {return nil};return String(expression[range])}
    }
    func outputSchema(_ incoming:[DataColumnSchema])throws->[DataColumnSchema] {
        try validateStructure();var schema=incoming
        for (index,row) in rows.enumerated() {
            if !incoming.isEmpty {for name in Self.references(row.expression) {guard let column=schema.first(where:{$0.name == name}),column.dtype != "image" else {throw DataFlowContractError("Operation \(index+1) references a missing or image column [\(name)].")}}}
            if let temporal=row.temporal, !incoming.isEmpty {
                guard let source=schema.first(where:{$0.name==temporal.column}) else {throw DataFlowContractError("Choose a known temporal source column.")}
                let supported=temporal.kind == "temporal_parse" ? source.dtype == "string":temporal.kind == "temporal_timezone" ? source.dtype == "datetime":TemporalColumnContract.types.contains(source.dtype)
                guard supported else {throw DataFlowContractError("The selected source type is not supported by this date/time operation.")}
                if temporal.kind == "temporal_difference" {
                    guard schema.first(where:{$0.name==temporal.otherColumn})?.dtype == source.dtype else {throw DataFlowContractError("Difference needs two columns of the same temporal type.")}
                }
                if temporal.kind == "temporal_part" {
                    let dateParts=["year","month","day","weekday","dayOfYear"],timeParts=["hour","minute","second","microsecond"]
                    let parts=source.dtype == "date" ? dateParts:source.dtype == "time" ? timeParts:dateParts+timeParts+["utcOffsetSeconds"]
                    guard parts.contains(temporal.part) else {throw DataFlowContractError("Choose a part supported by the source type.")}
                }
            }
            for output in row.outputs {
                let existing=schema.firstIndex{$0.name == output.name}
                if !incoming.isEmpty {
                    guard (existing != nil) == (output.write == "replace"),existing.map({schema[$0].dtype != "image"}) ?? true else {throw DataFlowContractError("Choose a new name or an existing scalar replacement for \(output.name).")}
                }
                var column=DataColumnSchema(name:output.name,dtype:output.dtype,nullable:row.nullPolicy == "preserve")
                if TemporalColumnContract.types.contains(output.dtype) {
                    column.temporal=try FormulaGuidedTemporal.metadata(for:row.expression,outputType:output.dtype,schema:schema)
                }
                if let existing {schema[existing]=column} else {schema.append(column)}
            }
            guard schema.count<=128 else {throw DataFlowContractError("Formula exceeds 128 output columns.")}
        }
        return schema
    }
    /// A column can be written only once per Formula, including other captures.
    func replacementColumns(for rowID:String,output index:Int,schema:[DataColumnSchema])->[DataColumnSchema] {
        guard let row=rows.first(where:{$0.id==rowID}),row.outputs.indices.contains(index) else {return []}
        let occupied=Set(rows.flatMap {row in row.outputs.enumerated().compactMap {offset,output in
            row.id==rowID && offset==index ? nil:output.name
        }})
        return schema.filter {$0.dtype != "image" && !occupied.contains($0.name)}
    }
    func validated(schema:[DataColumnSchema])throws->[String:JSONValue] {_ = try outputSchema(schema);return configuration}
    mutating func add(kind:String,column:String) {guard rows.count<64 else {return};let temporal=FormulaGuidedTemporal.kinds.contains(kind) ? FormulaGuidedTemporal(kind:kind,column:column):nil;let guided=kind == "expression" || temporal != nil ? nil:FormulaGuidedRegex(kind:kind,column:column);rows.append(.init(id:UUID().uuidString,expression:temporal?.expression ?? guided?.expression ?? "0",outputs:[.init(name:"calculated_\(rows.count+1)",dtype:temporal?.outputType ?? (kind == "match" ? "bool":kind == "expression" ? "int64":"string"))]))}
    mutating func remove(_ id:String) {rows.removeAll{$0.id==id}}
    mutating func duplicate(_ id:String) {guard rows.count<64,let index=rows.firstIndex(where:{$0.id==id}) else {return};var row=rows[index];row.id=UUID().uuidString;rows.insert(row,at:index+1)}
    mutating func move(_ id:String,by delta:Int) {guard let index=rows.firstIndex(where:{$0.id==id}),rows.indices.contains(index+delta) else {return};rows.swapAt(index,index+delta)}
    mutating func move(_ id:String,before target:String) {guard id != target,let index=rows.firstIndex(where:{$0.id==id}),rows.contains(where:{$0.id==target}) else {return};let row=rows.remove(at:index);rows.insert(row,at:rows.firstIndex(where:{$0.id==target})!)}
}
