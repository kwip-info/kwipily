import Foundation
import KwipMLCore

/// Recorded-schema guidance only. Execution checks actual rows, partitions and cardinality.
enum JoinTableConfiguration {
    static func validate(_ config:[String:JSONValue],left:[DataColumnSchema],right:[DataColumnSchema]) throws -> [String:JSONValue] {
        try JoinProjectionContract.validate(config)
        let required:Set<String>=["leftKeys","rightKeys","mode","nullPolicy","cardinality","leftPrefix","rightPrefix","maxRows"]
        guard required.isSubset(of:Set(config.keys)),Set(config.keys).subtracting(required).isSubset(of:["outputColumns","newColumns"]) else {throw DataFlowContractError("Supply all join settings without additional fields.")}
        if config["outputColumns"] != nil || config["newColumns"] != nil {
            _ = try JoinOutputColumn.decode(config["outputColumns"])
            guard case .string(let policy)=config["newColumns"],["keep","drop","error"].contains(policy) else {throw DataFlowContractError("Choose a new source column policy.")}
        }
        func keys(_ field:String)throws->[String] {
            guard case .array(let values)=config[field],!values.isEmpty,values.count <= 128 else {throw DataFlowContractError("Choose at least one ordered key pair.")}
            let names=values.compactMap {if case .string(let value)=$0 {return value};return nil}
            guard names.count == values.count,Set(names).count == names.count,names.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count <= 200}) else {throw DataFlowContractError("Join keys must be unique exact column names, at most 200 characters.")}
            return names
        }
        let lhs=try keys("leftKeys"),rhs=try keys("rightKeys")
        guard lhs.count == rhs.count else {throw DataFlowContractError("Each left key needs a corresponding right key.")}
        for (field,options) in [("mode",["inner","left","right","full"]),("nullPolicy",["neverMatch","match","error"]),("cardinality",["oneToOne","oneToMany","manyToOne","manyToMany"])] {
            guard case .string(let value)=config[field],options.contains(value) else {throw DataFlowContractError("Choose a supported \(field).")}
        }
        guard case .integer(let rows)=config["maxRows"],(1...100000).contains(rows) else {throw DataFlowContractError("Maximum rows must be an integer from 1 to 100,000.")}
        guard case .string(let lp)=config["leftPrefix"],case .string(let rp)=config["rightPrefix"],lp != rp,[lp,rp].allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count <= 100}) else {throw DataFlowContractError("Choose distinct, nonblank prefixes of at most 100 characters.")}
        for (l,r) in zip(lhs,rhs) {
            let lc=left.first {$0.name == l},rc=right.first {$0.name == r}
            guard left.isEmpty || lc != nil,right.isEmpty || rc != nil else {throw DataFlowContractError("A join key is absent from its recorded input schema.")}
            if lc?.dtype == "image" || rc?.dtype == "image" {throw DataFlowContractError("Image columns cannot be matching keys. Choose scalar metadata instead.")}
            if let lc,let rc,lc.dtype != rc.dtype {throw DataFlowContractError("Paired keys require matching types. Use Columns to cast types before joining.")}
        }
        if !left.isEmpty && !right.isEmpty {_ = try projected(config,left:left,right:right)}
        return config
    }
    static func outputSchema(_ config:[String:JSONValue],left:[DataColumnSchema],right:[DataColumnSchema])throws->[DataColumnSchema] {
        _=try validate(config,left:left,right:right)
        guard !left.isEmpty,!right.isEmpty else {return []}
        return try projected(config,left:left,right:right)
    }
    private static func projected(_ config:[String:JSONValue],left:[DataColumnSchema],right:[DataColumnSchema])throws->[DataColumnSchema] {
        guard case .string(let lp)=config["leftPrefix"],case .string(let rp)=config["rightPrefix"],case .string(let mode)=config["mode"] else {throw DataFlowContractError("Missing join settings.")}
        let overlap=Set(left.map(\.name)).intersection(right.map(\.name))
        let available=JoinOutputColumn.defaults(left:left,right:right,leftPrefix:lp,rightPrefix:rp)
        let configured=try config["outputColumns"].map {try JoinOutputColumn.decode($0)}
        var rows=configured ?? available
        if let configured {
            guard configured.allSatisfy({row in available.contains {$0.side == row.side && $0.source == row.source}}) else {throw DataFlowContractError("A configured output column is missing. Revise the grid.")}
            let new=available.filter {item in !configured.contains {$0.id == item.id}}
            if config["newColumns"] == .string("error"),!new.isEmpty {throw DataFlowContractError("New source columns require explicit grid choices.")}
            if config["newColumns"] == .string("keep") {rows += new}
        }
        let output=rows.filter(\.included).compactMap {row -> DataColumnSchema? in
            guard let source=(row.side == "left" ? left:right).first(where:{$0.name == row.source}) else {return nil}
            return DataColumnSchema(name:row.name,dtype:source.dtype,nullable:(row.side == "left" ? ["right","full"]:["left","full"]).contains(mode) ? true:source.nullable)
        }

        guard !output.isEmpty,output.count <= 128,Set(output.map(\.name)).count == output.count,output.allSatisfy({$0.name.count <= 200}) else {throw DataFlowContractError("Joined names collide or exceed 128 columns / 200 characters. Change prefixes or select and rename input columns.")}
        return output
    }
}


struct JoinOutputColumn:Equatable,Identifiable {
    var side:String
    var source:String
    var name:String
    var included:Bool
    var id:String {"\(side):\(source)"}
    var json:JSONValue {.object(["side":.string(side),"source":.string(source),"name":.string(name),"included":.bool(included)])}
    static func decode(_ value:JSONValue?)throws->[Self] {
        guard case .array(let values)=value,values.count <= 256 else {throw DataFlowContractError("Invalid output column grid.")}
        let rows=try values.map {value -> Self in
            guard case .object(let o)=value,Set(o.keys)==["side","source","name","included"],case .string(let side)=o["side"],["left","right"].contains(side),case .string(let source)=o["source"],case .string(let name)=o["name"],case .bool(let included)=o["included"],[source,name].allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200}) else {throw DataFlowContractError("Output columns need a side, exact source, output name and Keep choice.")}
            return .init(side:side,source:source,name:name,included:included)
        }
        guard Set(rows.map(\.id)).count == rows.count else {throw DataFlowContractError("Each source column can appear only once in the grid.")}
        return rows
    }
    static func defaults(left:[DataColumnSchema],right:[DataColumnSchema],leftPrefix:String,rightPrefix:String)->[Self] {
        let overlap=Set(left.map(\.name)).intersection(right.map(\.name))
        return [("left",left,leftPrefix),("right",right,rightPrefix)].flatMap {side,columns,prefix in
            columns.map {.init(side:side,source:$0.name,name:overlap.contains($0.name) ? prefix+$0.name:$0.name,included:true)}
        }
    }
}
