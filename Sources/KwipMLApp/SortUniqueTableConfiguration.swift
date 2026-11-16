import KwipMLCore

enum SortTableConfiguration {
    static func validate(_ values:[String:JSONValue],schema:[DataColumnSchema])throws->[String:JSONValue] {
        let config=try SortConfiguration(values)
        try SortUniqueTableConfiguration.validateKeys(config.keys.map(\.column),schema:schema)
        return config.values
    }
}
enum UniqueTableConfiguration {
    static func validate(_ values:[String:JSONValue],schema:[DataColumnSchema])throws->[String:JSONValue] {
        let config=try UniqueConfiguration(values)
        try SortUniqueTableConfiguration.validateKeys(config.keys,schema:schema)
        return config.values
    }
}
enum SortUniqueTableConfiguration {
    static func validateKeys(_ keys:[String],schema:[DataColumnSchema])throws {
        guard !schema.isEmpty else {return}
        for key in keys {
            guard let column=schema.first(where:{$0.name==key}) else {throw DataFlowContractError("Unknown key column: \(key).")}
            guard ["string","int64","float64","bool","date","datetime","time"].contains(column.dtype) else {throw DataFlowContractError("Select scalar key columns; images cannot be keys.")}
        }
    }
}
