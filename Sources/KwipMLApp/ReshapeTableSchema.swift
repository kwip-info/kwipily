import KwipMLCore

enum ReshapeTableSchema {
    /// Wide vocabulary is runtime-observed, so it cannot be inferred from source types.
    static func projected(_ values:[String:JSONValue],schema:[DataColumnSchema])throws->[DataColumnSchema] {
        let c=try ReshapeTableConfiguration(values)
        guard c.string("mode")=="long" else {return []}
        let ids=c.columns("idColumns"),names=c.columns("valueColumns")
        guard !names.isEmpty,Set(ids).isDisjoint(with:Set(names)) else {throw DataFlowContractError("Select disjoint identifier and value columns.")}
        guard !schema.isEmpty else {return []}
        func column(_ name:String)throws->DataColumnSchema {guard let col=schema.first(where:{$0.name==name}) else {throw DataFlowContractError("Column \(name) is unavailable.")};return col}
        let fields=try names.map(column),types=Set(fields.map(\.dtype))
        guard types.count==1 || (c.string("typePolicy")=="losslessNumeric" && types.isSubset(of:["int64","float64"])) else {throw DataFlowContractError("Value types differ; choose checked numeric widening or Columns.")}
        guard fields.allSatisfy({$0.temporal==fields[0].temporal}) else {throw DataFlowContractError("Long temporal metadata must match exactly.")}
        let result=try ids.map(column)+[DataColumnSchema(name:c.string("fieldName"),dtype:"string",nullable:false),DataColumnSchema(name:c.string("valueName"),dtype:types.count>1 ? "float64":fields[0].dtype,nullable:c.string("missingValues")=="keep" && fields.contains {$0.nullable != false},temporal:fields[0].temporal)]
        guard result.count<=128,Set(result.map(\.name)).count==result.count else {throw DataFlowContractError("Reshape output names collide or exceed 128 columns.")}
        return result
    }
    static func validate(_ values:[String:JSONValue],schema:[DataColumnSchema])throws->[String:JSONValue] {
        let c=try ReshapeTableConfiguration(values)
        if c.string("mode")=="long" {_ = try projected(values,schema:schema)}
        else {
            guard !c.string("columnKey").isEmpty,case .array(let fields)=values["values"],!fields.isEmpty else {throw DataFlowContractError("Wide needs a column-name field and value rows.")}
            if c.string("missingCombinations")=="fill",values["fillValue"] == .null {throw DataFlowContractError("Enter a valid typed fixed fill.")}
            let keys=c.columns("rowKeys")+[c.string("columnKey")]
            for key in keys {if let col=schema.first(where:{$0.name==key}),col.dtype=="image" {throw DataFlowContractError("Wide keys must be scalar fields.")}}
            if !schema.isEmpty {
                for key in keys {guard schema.contains(where:{$0.name==key}) else {throw DataFlowContractError("Column \(key) is unavailable.")}}
                for field in fields {
                    guard case .object(let row)=field,case .string(let name)=row["column"],case .string(let op)=row["aggregation"],let col=schema.first(where:{$0.name==name}) else {throw DataFlowContractError("Wide value is unavailable.")}
                    if ["sum","mean"].contains(op),!["int64","float64"].contains(col.dtype) {throw DataFlowContractError("Sum and Mean require numeric values.")}
                    if col.dtype=="image",!["first","last","countRows","countNonNull"].contains(op) {throw DataFlowContractError("Images support First, Last and counts only.")}
                }
            }
        }
        return c.values
    }
}
