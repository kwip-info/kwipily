import KwipMLCore

/// Shared typed draft validation; persistence and authority stay with each editor owner.
enum DataTableConfiguration {
    static func validate(_ config:[String:JSONValue],operatorID:String,schema:[DataColumnSchema],leftSchema:[DataColumnSchema]=[],rightSchema:[DataColumnSchema]=[],definition:DataFlowOperatorDefinition?=nil) throws -> [String:JSONValue] {
    switch operatorID {
    case GameRuleConfiguration.operatorID: try GameRuleConfiguration.validate(config);return config
    case "kwip.data.decide":try TypedDecisionConfiguration.validate(config);return config
    case "kwip.data.classifyAudio":
        guard let definition else {throw DataFlowContractError("Audio classification is unavailable.")}
        try AudioClassificationConfiguration.validate(config,definition:definition);return config
    case "kwip.data.transcribeAudio":
        guard let definition else {throw DataFlowContractError("Transcription is unavailable.")}
        try SpeechTranscriptionConfiguration.validate(config,definition:definition);return config
    case "kwip.data.extractText":
        guard let definition else {throw DataFlowContractError("OCR definition unavailable.")}
        try OCRConfiguration.validate(config,definition:definition);return config
    case "kwip.data.prepareMedia":
        try PrepareMediaConfiguration.validate(config)
        guard let input=config["inputColumn"]?.stringValue,let output=config["outputColumn"]?.stringValue else {throw DataFlowContractError("Choose source and prepared output columns.")}
        if !schema.isEmpty {guard schema.contains(where:{$0.name==input && $0.dtype==(config["kind"] == .string("audio") ? "audio":"image")}),!schema.contains(where:{$0.name==output}) else {throw DataFlowContractError("Choose a captured media column and an unused output name.")}}
        return config
    case "kwip.data.sequenceWindows": try SequenceWindowsConfiguration.validate(config,schema:schema.map {.init(name:$0.name,dtype:$0.dtype,nullable:$0.nullable ?? true)});return config
    case "kwip.data.textEncode":
        guard let definition else {throw DataFlowContractError("Text encoding is unavailable.")}
        try TextEncodingConfiguration.validate(config,definition:definition)
        return config
    case "kwip.data.outputDelimited":
        guard let definition else {throw DataFlowContractError("The installed Table Output definition is unavailable.")}
        return try TableOutputConfiguration.validate(config,definition:definition,schema:schema)
    case "kwip.data.sort": return try SortTableConfiguration.validate(config,schema:schema)
    case "kwip.data.unique": return try UniqueTableConfiguration.validate(config,schema:schema)
    case "kwip.data.reshapeTable": return try ReshapeTableSchema.validate(config,schema:schema)
    case "kwip.data.splitText": return try SplitTextConfigurationDraft(configuration:config).validatedConfiguration(schema:schema)
    case "kwip.data.preprocess": let value=try PreprocessConfiguration(values:config);try value.validate(schema:schema);return config
    case "kwip.data.formula": return try FormulaDraft(configuration:config).validated(schema:schema)
    case "kwip.data.partition": return try PartitionToolDraft(configuration:config).configuration(schema:schema)
    case "kwip.data.aggregate": return try AggregateTableConfiguration.validate(config,schema:schema)
    case "kwip.data.join": return try JoinTableConfiguration.validate(config,left:leftSchema,right:rightSchema)
    case "kwip.data.filterRows": return try FilterRowsConfigurationDraft(configuration: config).validatedConfiguration(schema: schema)
    case "kwip.data.columns": return try ColumnsConfigurationDraft(configuration: config).validatedConfiguration(schema: schema)
    default: throw DataFlowContractError("This editor cannot configure that operator.")
    }
    }
}
