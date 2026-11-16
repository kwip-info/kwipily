import KwipMLCore

/// Pure editing guidance, not execution validation. The caller owns source freshness,
/// fixture authority and captured-evidence binding. This type never opens a source or
/// infers category vocabularies that have not been explicitly supplied as evidence.
enum DataNodeSchemaProjection {
    static func upstreamSchema(
        for nodeID: String,
        in graph: DataFlowDocument,
        inputPort: String? = nil,
        sourceSchema: (String) -> [DataColumnSchema],
        capturedSchema: ((DataFlowEndpoint) -> [DataColumnSchema]?)? = nil
    ) -> [DataColumnSchema] {
        guard graph.nodes.count <= 128, graph.edges.count <= 256,
              Set(graph.nodes.map(\.id)).count == graph.nodes.count,
              let node = graph.nodes.first(where: { $0.id == nodeID }) else { return [] }
        let nodes = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        func singleInput(_ id: String, ports: Set<String>) -> DataFlowEdge? {
            let incoming = graph.edges.filter { $0.to.node == id && ports.contains($0.to.port) }
            return incoming.count == 1 ? incoming.first : nil
        }
        func project(_ endpoint: DataFlowEndpoint, visited: Set<String>) -> [DataColumnSchema] {
            guard !visited.contains(endpoint.node), let node = nodes[endpoint.node] else { return [] }
            // nil means unavailable; an explicitly empty captured schema must not fall
            // back to declarations or source headers.
            if let captured = capturedSchema?(endpoint) { return captured }
            if node.operatorID==TypedDecisionConfiguration.operatorID {
                if endpoint.port=="decisions" {return [.init(name:"source_observation",dtype:"string",nullable:false),.init(name:"source_key",dtype:"string",nullable:false),.init(name:"question",dtype:"string",nullable:false),.init(name:"type",dtype:"string",nullable:false),.init(name:"answer",dtype:"string",nullable:false),.init(name:"confidence",dtype:"float64",nullable:false),.init(name:"expected_index",dtype:"float64",nullable:true),.init(name:"yes_probability",dtype:"float64",nullable:true),.init(name:"status",dtype:"string",nullable:false),.init(name:"needs_review",dtype:"bool",nullable:false),.init(name:"model_identity",dtype:"string",nullable:false),.init(name:"question_set",dtype:"string",nullable:false)]}
                if endpoint.port=="scores" {return [.init(name:"source_observation",dtype:"string",nullable:false),.init(name:"source_key",dtype:"string",nullable:false),.init(name:"question",dtype:"string",nullable:false),.init(name:"option_key",dtype:"string",nullable:false),.init(name:"option_label",dtype:"string",nullable:false),.init(name:"option_index",dtype:"int64",nullable:false),.init(name:"probability",dtype:"float64",nullable:false),.init(name:"temperature",dtype:"float64",nullable:false),.init(name:"model_identity",dtype:"string",nullable:false),.init(name:"question_set",dtype:"string",nullable:false)]}
                return []
            }
            if node.operatorID==GameRuleConfiguration.operatorID {
                let columns:[(String,String)]
                switch endpoint.port {
                case "tests":columns=[("test","string"),("status","string"),("detail","string"),("moves","int64"),("definition","string")]
                case "summary":columns=[("definition","string"),("passed","int64"),("failed","int64"),("moves","int64")]
                case "definition":columns=[("schema","string"),("sha256","string"),("json","string")]
                case "trace":columns=[("test","string"),("move","int64"),("before","string"),("action","string"),("after","string"),("status","string"),("rule","string"),("terminal","bool"),("winner","string"),("rewards","string")]
                default:return []
                };return columns.map {.init(name:$0.0,dtype:$0.1,nullable:false)}
            }
            if DataFlowSourceOperators.contains(node.operatorID) { return sourceSchema(node.id) }
            if node.operatorID == "kwip.data.union",let config=try? UnionConfiguration(node.config) {
                var schemas:[String:[DataColumnSchema]]=[:]
                for port in config.inputs {if let input=singleInput(node.id,ports:[port]) {schemas[port]=project(input.from,visited:visited.union([node.id]))}}
                return (try? UnionTableConfiguration.alignment(node.config,schemas:schemas).map(\.output)) ?? []
            }
            if node.operatorID == "kwip.data.join" {
                guard let left=singleInput(node.id,ports:["left"]),let right=singleInput(node.id,ports:["right"]) else {return []}
                return (try? JoinTableConfiguration.outputSchema(node.config,left:project(left.from,visited:visited.union([node.id])),right:project(right.from,visited:visited.union([node.id])))) ?? []
            }
            guard let input = singleInput(node.id, ports: (node.operatorID == "kwip.data.preprocess" && node.config["mode"] != .string("table") || node.operatorID == TextEncodingConfiguration.operatorID && node.config["vocabularySource"] == .string("training") || node.operatorID == SequenceWindowsConfiguration.operatorID && node.config["mode"] == .string("partitions")) ? ["training"]:["table"]) else { return [] }
            var schema = project(input.from, visited: visited.union([node.id]))
            guard !schema.isEmpty else { return [] }
            switch node.operatorID {
            case "kwip.data.sequenceWindows": return SequenceWindowsConfiguration.columns(node.config).map {.init(name:$0.name,dtype:$0.dtype,nullable:$0.nullable)}
            case "kwip.data.textEncode":
                guard let output=node.config["outputColumn"]?.stringValue,let source=node.config["textColumn"]?.stringValue,schema.contains(where:{$0.name==source && $0.dtype=="string"}) else {return []}
                if node.config["keepSourceText"] == .bool(false) {schema.removeAll {$0.name==source}}
                guard !schema.contains(where:{$0.name==output || $0.name==output+"_length"}) else {return []}
                schema += [.init(name:output,dtype:"int64Sequence",nullable:false),.init(name:output+"_length",dtype:"int64",nullable:false)]
            case "kwip.data.classifyAudio":
                let kind=schema.first(where:{$0.name==node.config["inputColumn"]?.stringValue})?.dtype ?? "tensor"
                if endpoint.port=="predictions",let prefix=node.config["outputPrefix"]?.stringValue {
                    schema += [("window","int64"),("start_seconds","float64"),("end_seconds","float64"),("label","string"),("score","float64"),("reported","int64"),("status","string"),("model","string")].map{.init(name:prefix+"_"+$0.0,dtype:$0.1,nullable:false)}
                } else if endpoint.port=="scores" {
                    schema=[("source_observation","string"),("window","int64"),("start_seconds","float64"),("end_seconds","float64"),("label","string"),("logit","float64"),("score","float64"),("rank","int64"),("meets_threshold","bool"),("reported","bool"),("source_audio",kind),("model_identity","string"),("timestamp_coverage","string")].map{.init(name:$0.0,dtype:$0.1,nullable:false)}
                } else {return []}
            case "kwip.data.transcribeAudio":
                if endpoint.port=="transcripts",let name=node.config["outputColumn"]?.stringValue {schema += [(name,"string"),("speech_status","string"),("speech_model","string"),("speech_chunks","int64")].map{.init(name:$0.0,dtype:$0.1,nullable:false)}}
                else if endpoint.port=="segments" {let kind=schema.first(where:{$0.name==node.config["inputColumn"]?.stringValue})?.dtype ?? "tensor";schema=[("source_observation","string"),("segment","int64"),("start_seconds","float64"),("end_seconds","float64"),("text","string"),("source_audio",kind),("speech_model","string"),("timestamp_coverage","string")].map{.init(name:$0.0,dtype:$0.1,nullable:false)}} else {return []}
            case "kwip.data.extractText":
                if endpoint.port=="documents",let name=node.config["outputColumn"]?.stringValue {
                    let names=Set(schema.map(\.name))
                    let geometry=Set(schema.filter {$0.dtype=="image" && names.contains($0.name+"_source")}.map {$0.name+"_to_source"})
                    schema.removeAll {["image","audio","tensor"].contains($0.dtype) || ["unit_index","page_number","frame_index"].contains($0.name) || $0.name.hasPrefix("pdf_map_") || geometry.contains($0.name) || $0.name==node.config["pageColumn"]?.stringValue}
                    schema += [.init(name:name,dtype:"string",nullable:false),.init(name:"ocr_status",dtype:"string",nullable:false),.init(name:"ocr_page_count",dtype:"int64",nullable:false),.init(name:"ocr_region_count",dtype:"int64",nullable:false),.init(name:"ocr_review_count",dtype:"int64",nullable:false),.init(name:"ocr_pack",dtype:"string",nullable:false)]
                } else if endpoint.port=="regions" {
                    schema=[("document_id","string"),("page","int64"),("region","int64"),("text","string"),("confidence","float64"),("review","string"),("source_image","image")].map {.init(name:$0.0,dtype:$0.1,nullable:false)}
                    schema += (0..<4).flatMap {i in [DataColumnSchema(name:"x"+String(i),dtype:"float64",nullable:false),DataColumnSchema(name:"y"+String(i),dtype:"float64",nullable:false)]}
                    schema += [.init(name:"coordinate_space",dtype:"string",nullable:false),.init(name:"ocr_pack",dtype:"string",nullable:false)]
                    schema += (0..<4).flatMap {i in [DataColumnSchema(name:"pdf_x"+String(i),dtype:"float64",nullable:true),DataColumnSchema(name:"pdf_y"+String(i),dtype:"float64",nullable:true)]}
                } else {return []}
            case "kwip.data.prepareMedia":
                guard (try? PrepareMediaConfiguration.validate(node.config)) != nil,let name=node.config["outputColumn"]?.stringValue else {return []}
                schema.append(.init(name:name,dtype:node.config["kind"] == .string("image") && node.config["outputMode"] == .string("image") ? "image":"tensor",nullable:false))
                if node.config["kind"] == .string("image"),node.config["outputMode"] == .string("image") {schema += [.init(name:name+"_source",dtype:"image",nullable:false),.init(name:name+"_to_source",dtype:"string",nullable:false)]}
                if node.config["kind"] == .string("audio") {schema += [.init(name:name+"_mask",dtype:"tensor",nullable:false),.init(name:name+(node.config["representation"] == nil || node.config["representation"] == .string("samples") ? "_valid_samples":"_valid_frames"),dtype:"int64",nullable:false),.init(name:name+"_parent",dtype:"string",nullable:false)]}
            case "kwip.data.reshapeTable": return (try? ReshapeTableSchema.projected(node.config,schema:schema)) ?? []
            case "kwip.data.splitText": return (try? SplitTextConfigurationDraft(configuration:node.config).outputSchema(schema)) ?? []
            case "kwip.data.sort", "kwip.data.unique", "kwip.data.outputDelimited": break
            case "kwip.data.preprocess": return (try? PreprocessConfiguration(values:node.config).outputSchema(schema)) ?? []
            case "kwip.data.formula": return (try? FormulaDraft(configuration:node.config).outputSchema(schema)) ?? []
            case "kwip.data.columns": return (try? ColumnsConfigurationDraft(configuration: node.config).outputSchema(schema)) ?? []
            case "kwip.data.aggregate": return (try? AggregateTableConfiguration.outputSchema(node.config,schema:schema)) ?? []
            case "kwip.data.filterRows", "kwip.data.view", "kwip.data.partition", "kwip.data.sampleRows": break
            default: return []
            }
            return schema
        }
        if inputPort == nil, DataFlowSourceOperators.contains(node.operatorID) { return project(.init(node: nodeID, port: "table"), visited: []) }
        guard let input = singleInput(nodeID, ports: inputPort.map {Set([$0])} ?? (node.operatorID == "kwip.data.preprocess" && node.config["mode"] != .string("table") ? ["training"]:["table", "training"])) else { return [] }
        return project(input.from, visited: [])
    }
}
