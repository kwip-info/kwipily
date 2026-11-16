import KwipMLCore

enum UnionTableConfiguration {
    struct Alignment {
        var output:DataColumnSchema
        var sources:[String?]
    }
    static func alignment(_ values:[String:JSONValue],schemas:[String:[DataColumnSchema]])throws->[Alignment] {
        let config=try UnionConfiguration(values)
        let tables=config.inputs.map {schemas[$0] ?? []}
        guard tables.allSatisfy({!$0.isEmpty}) else {return []}
        for table in tables {
            guard Set(table.map(\.name)).count==table.count,table.count<=128 else {throw DataFlowContractError("Union input schema has duplicate names or too many columns.")}
        }
        let keys=tables.map {table in table.enumerated().map {config.match == "name" ? $0.element.name:String($0.offset)}}
        if config.coverage == "exact",keys.contains(where:{Set($0) != Set(keys[0])}) {throw DataFlowContractError("Union requires identical column names or position counts.")}
        var ordered:[String]=[]
        for key in keys.flatMap({$0}) where !ordered.contains(key) {ordered.append(key)}
        if config.coverage == "shared" {ordered=ordered.filter {key in keys.allSatisfy {$0.contains(key)}}}
        var rows:[Alignment]=try ordered.map {key -> Alignment in
            let found=zip(tables,keys).map {table,keys in keys.firstIndex(of:key).map {table[$0]}}
            let known=found.compactMap {$0},types=Set(known.map(\.dtype))
            guard types.count==1 || (config.typePolicy == "losslessNumeric" && types.isSubset(of:["int64","float64"])) else {throw DataFlowContractError("Union column \(known[0].name) has incompatible types. Use Columns or checked numeric widening.")}
            return Alignment(output:.init(name:known[0].name,dtype:types.count>1 ? "float64":known[0].dtype,nullable:known.count<tables.count || known.contains {$0.nullable != false}),sources:found.map {$0?.name})
        }
        if !config.sourceColumn.isEmpty {
            guard !rows.contains(where:{$0.output.name==config.sourceColumn}) else {throw DataFlowContractError("Input identifier conflicts with an output column.")}
            rows.append(.init(output:.init(name:config.sourceColumn,dtype:"string",nullable:false),sources:config.inputs.map {Optional($0)}))
        }
        guard (1...128).contains(rows.count),Set(rows.map {$0.output.name}).count==rows.count else {throw DataFlowContractError("Union needs 1–128 unique output columns. Inspect alignment and coverage.")}
        return rows
    }
    static func validate(_ values:[String:JSONValue],schemas:[String:[DataColumnSchema]],connected:Set<String>=[])throws->[String:JSONValue] {
        let config=try UnionConfiguration(values)
        guard connected.isSubset(of:Set(config.inputs)) else {throw DataFlowContractError("Disconnect a Union input before removing its port. Existing connections are preserved.")}
        _ = try alignment(values,schemas:schemas)
        return config.values
    }
}
