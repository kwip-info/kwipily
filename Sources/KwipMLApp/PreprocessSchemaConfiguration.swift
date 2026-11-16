import KwipMLCore

extension PreprocessConfiguration {
    func validate(schema:[DataColumnSchema])throws {
        try validate()
        func require(_ condition:Bool,_ message:String)throws {if !condition {throw DataFlowContractError(message)}}
        for row in columns {
            if !schema.isEmpty {
                guard let col=schema.first(where:{$0.name==row.column}) else {throw DataFlowContractError("Column \(row.column) is missing upstream.")}
                try require(col.dtype=="string" || !row.hasTextCleanup,"Text cleanup requires a Text column.")
                if ["image","date","datetime","time"].contains(col.dtype) {try require(["keep","error","drop"].contains(row.missing) && row.encoding=="none" && row.scaling=="none","Image and temporal columns support Preserve, Drop row or Stop run only.")}
                try require((!["mean","median"].contains(row.missing) && row.scaling=="none") || ["float64","int64"].contains(col.dtype),"Numeric imputation and scaling require numeric columns.")
                try require(row.encoding != "oneHot" || col.dtype=="string","One-hot encoding requires Text columns.")
                if row.missing=="constant" {try require(Self.matches(row.value,dtype:col.dtype),"Replacement constant must match \(col.dtype).")}
                if row.emptyPolicy=="constant" && ["mean","median","mostFrequent"].contains(row.missing) {try require(Self.matches(row.emptyValue,dtype:["mean","median"].contains(row.missing) ? "float64":col.dtype),"All-missing constant has the wrong type.")}
            }
        }
    }
    func outputSchema(_ schema:[DataColumnSchema])throws->[DataColumnSchema] {
        try validate(schema:schema)
        // Learned vocabularies cannot be inferred from a declaration or source sample.
        if columns.contains(where:{$0.encoding=="oneHot"}) {return []}
        return schema.map {col in
            guard let rule=columns.first(where:{$0.column==col.name}) else {return col}
            return .init(name:col.name,dtype:["mean","median"].contains(rule.missing) || rule.scaling=="standard" ? "float64":col.dtype,nullable:rule.missing=="keep" ? (rule.emptyAfterCleanup=="missing" ? true:col.nullable):false,temporal:col.temporal)
        }
    }
}
