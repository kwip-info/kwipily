import Foundation

public enum WorkflowCodec {
    public static let maximumBytes=2_000_000
    public static func decode(_ data:Data)throws->WorkflowDocument {
        try check(data.count<=maximumBytes,"Workflow exceeds 2 MB.")
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        try structure(raw)
        let document=try JSONDecoder().decode(WorkflowDocument.self,from:data)
        try validate(document);return document
    }
    public static func encode(_ document:WorkflowDocument)throws->Data {
        try validate(document)
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
        let data=try encoder.encode(document)
        try check(data.count<=maximumBytes,"Workflow exceeds 2 MB.")
        return data
    }
    public static func validate(_ document:WorkflowDocument)throws {
        try check(document.kind=="workflowProject" && [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24].contains(document.schemaVersion) && UUID(uuidString:document.projectID) != nil,"Unknown workflow kind, version or project identity.")
        if let origin = document.templateOrigin {
            try check(document.schemaVersion >= 24 && origin.version > 0 && origin.version <= 10000 &&
                origin.id.count <= 80 && !origin.id.isEmpty && origin.id.allSatisfy { $0.isASCII && ($0.isLowercase || $0.isNumber || $0 == "-") },
                "Invalid versioned template origin.")
        }
        try WorkflowModelInput.validateOwnership(document)
        if let progress=document.guidedBuild {
            try check(named(progress.courseID) && (1...10000).contains(progress.version) && (0...256).contains(progress.completedSteps),"Invalid guided-build progress metadata.")
        }
        try check(document.name.count<=500 && document.nodes.count<=256 && document.edges.count<=512 && document.definitions.count<=64 && document.drafts.count<=1,"Workflow name, graph or definition/draft bounds exceeded.")
        try check(document.nodes.filter{$0.address.domain == .data}.count<=128 && document.nodes.filter{$0.address.domain == .model}.count<=128,"A data/model execution domain supports at most 128 nodes.")
        try check(Set(document.nodes.map(\.address)).count==document.nodes.count,"Duplicate qualified workflow node address.")
        for node in document.nodes {
            if let settings=node.parameterSettings {
                try check(document.schemaVersion>=15,"Layer parameter settings require workflow schema15.")
                guard case .modelPrimitive=node.payload else {throw failure("Only primitive layers can own parameter settings.")}
                try LayerParameterSettings.validate(settings)
            }
            try address(node.address)
            try check(node.address.domain==node.payload.domain && node.name.count<=500 && node.x.isFinite && node.y.isFinite,"Workflow node domain or presentation metadata is invalid.")
            switch node.payload {
            case .dataOperator(let operatorID,let config):
                try check(named(operatorID),"Invalid data operator identity.");try json(.object(config))
                if operatorID=="kwip.data.sourceCSV" {
                    if try TableInputConfiguration.format(of: config) == .csv { try keys(config, CSVParsingConfiguration.allowedKeys.union(["format"])) }
                    else { _ = try TableInputConfiguration.validate(config) }
                } else if operatorID==DataFlowSourceOperators.fileInput { _ = try FileInputConfiguration.validate(config) }
            case .modelPrimitive(let type,let version,let config):
                if type=="output",config["task"] != nil {try check(document.schemaVersion>=23,"Policy/value outputs require workflow schema23.");_ = try ModelOutputConfiguration(config)}
                try check(named(type) && [1,2].contains(version),"Invalid model primitive or configuration version.");try json(.object(config))
            case .modelBlock(_,let overrides):try json(.object(overrides))
            case .frozenModel(let config):
                try check(document.schemaVersion>=5,"Frozen model requires workflow schema 5.")
                if !config.isEmpty {try check(document.schemaVersion>=20,"External model sources require workflow schema20.");_ = try LoadModelConfiguration.decode(config)}
            case .gameSession(let config):
                try check(document.schemaVersion>=22,"Game sessions require workflow schema22.")
                _ = try GameSessionConfiguration(config)
            case .generateText(let config):
                try check(document.schemaVersion>=19,"Generate text requires workflow schema19.")
                _ = try TextGenerationConfiguration(config)
            case .saveModel(let config):
                try check(document.schemaVersion>=17,"Save model requires workflow schema17.")
                try json(.object(config));_ = try WorkflowSaveModelConfiguration.decode(config)
            case .evaluateTest(let config):
                try check(document.schemaVersion>=5,"Evaluate test requires workflow schema 5.")
                _ = try WorkflowEvaluationSettings.decode(config)
            case .optimizerRecipe(let type,let version,let config):
                try check(document.schemaVersion>=6 && version==1,"Optimizer recipes require workflow schema6 and configuration version1.")
                _ = try OptimizerConfiguration(componentID:type,values:config)
            case .lossRecipe(let type,let version,let config):
                try check(document.schemaVersion>=8 && version==1,"Objective recipes require workflow schema8 and configuration version1.")
                _ = try LossConfiguration(componentID:type,values:config)
            case .initializationRecipe(let type,let version,let config):
                try check(document.schemaVersion>=9 && version==1,"Initialization recipes require workflow schema9 and configuration version1.")
                _ = try InitializationConfiguration(componentID:type,values:config)
            case .gradientPolicyRecipe(let type,let version,let config):
                try check(document.schemaVersion>=10 && version==1,"Gradient/trainability recipes require workflow schema10 and configuration version1.")
                _ = try GradientPolicyConfiguration(componentID:type,values:config)
            case .trainabilityRecipe(let type,let version,let config):
                try check(document.schemaVersion>=10 && version==1,"Gradient/trainability recipes require workflow schema10 and configuration version1.")
                _ = try TrainabilityConfiguration(componentID:type,values:config)
            case .schedulerRecipe(let type,let version,let config):
                try check(document.schemaVersion>=7 && version==1,"Scheduler recipes require workflow schema7 and configuration version1.")
                _ = try SchedulerConfiguration(componentID:type,values:config)
            case .datasetRecipe(let type,let version,let config):
                try check(document.schemaVersion>=13 && version==1 && type==DatasetConfiguration.componentID,"Data reader recipes require workflow schema13 and configuration version1.")
                _ = try DatasetConfiguration(values:config)
            case .batchRecipe(let type,let version,let config):
                try check(document.schemaVersion>=13 && version==1 && type==DataLoaderConfiguration.componentID,"Data reader recipes require workflow schema13 and configuration version1.")
                _ = try DataLoaderConfiguration(values:config)
            case .modelDefinition:break
            case .trainingData(let config):
                if config["bagInput"] != nil {
                    try check(document.schemaVersion>=12,"Bag input mapping requires explicit workflow schema12 migration.")
                    _ = try WorkflowControlContracts.bagInputMapping(config,address:node.address)
                } else if config["lossTargetMapping"] != nil {
                    try check(document.schemaVersion>=8,"Explicit objective mapping requires workflow schema8.")
                    _ = try WorkflowControlContracts.lossTargetMapping(config,address:node.address)
                } else {try keys(config,["featureColumns","targetColumn"])}
                try json(.object(config))
            case .runSetup(let config):
                var allowed:Set<String>=["device","epochs","batchSize","seed","learningRate","gradientCapture","trainingControl"]
                if document.schemaVersion>=2 {allowed.insert("preparationCachePolicy")}
                if document.schemaVersion>=4 {allowed.insert("experiment")}
                if let specification=config["experiment"] {_ = try WorkflowExperimentSpecification.decode(specification)}
                try keys(config,allowed);try json(.object(config))
                if let value=config["trainingControl"] {_ = try JSONDecoder().decode(TrainingControlPolicy.self,from:JSONEncoder().encode(value))}
                _ = try WorkflowControlContracts.preparationCachePolicy(config,schemaVersion:document.schemaVersion)
                if let value=config["gradientCapture"],value != .null {
                    guard case .object(let capture)=value else {throw failure("Gradient capture configuration must be an object or null.")}
                    try keys(capture,["schemaVersion","everyEpochs","maxBatchesPerEpoch","maxCaptures"])
                }
            case .estimator(let config):
                try check(document.schemaVersion >= 5,"Estimator nodes require workflow schema 5.")
                try keys(config,["backendID","featureColumns","targetColumn","settings","task","algorithm","operation","reviewThreshold"]);try json(.object(config))
            case .train(let config):
                if config["selfPlay"] != nil {try check(document.schemaVersion>=23,"Self-play requires workflow schema23.");_ = try SelfPlayConfiguration(config);continue}
                var fields:Set<String> = document.schemaVersion >= 3 ? ["trainingMode"]:[]
                if document.schemaVersion>=8 {fields.insert("objectiveScalarization")}
                if document.schemaVersion>=13 {fields.formUnion(["datasetAccess","batching"]);fields.formUnion(WorkflowTrainRecipes.roleKeys)}
                if document.schemaVersion>=16 {fields.insert("runSetup")}
                if document.schemaVersion>=18 {fields.insert("dataMapping")}
                if document.schemaVersion>=21 {fields.insert("fineTuning")}
                _ = try FineTuningConfiguration.decode(config)
                try json(.object(config))
                try keys(config,fields)
                _ = try WorkflowTrainMapping.configuration(config,schemaVersion:document.schemaVersion)
                _ = try WorkflowTrainRunSetup.configuration(config,schemaVersion:document.schemaVersion)
                _ = try WorkflowControlContracts.objectiveScalarization(config,schemaVersion:document.schemaVersion)
                _ = try WorkflowControlContracts.trainingMode(config, schemaVersion: document.schemaVersion)
                try WorkflowTrainDataLoading.validate(config,schemaVersion:document.schemaVersion)
                _ = try WorkflowTrainRecipes(configuration:config,schemaVersion:document.schemaVersion)
            }
        }
        try check(document.schemaVersion>=5 || !document.edges.contains {$0.kind == .frozenModel},"Frozen model edges require workflow schema 5.")
        try check(document.schemaVersion>=6 || !document.edges.contains {$0.kind == .optimizerRecipe},"Optimizer recipe edges require workflow schema6.")
        try check(document.schemaVersion>=7 || !document.edges.contains {$0.kind == .schedulerRecipe},"Scheduler recipe edges require workflow schema7.")
        try check(document.schemaVersion>=10 || !document.edges.contains {$0.kind == .gradientPolicyRecipe || $0.kind == .trainabilityRecipe},"Gradient/trainability recipe edges require workflow schema10.")
        try check(document.schemaVersion>=9 || !document.edges.contains {$0.kind == .initializationRecipe},"Initialization recipe edges require workflow schema9.")
        try check(document.schemaVersion>=8 || !document.edges.contains {$0.kind == .objectiveRecipe},"Objective recipe edges require workflow schema8.")
        try check(document.schemaVersion>=13 || !document.edges.contains {$0.kind == .datasetRecipe || $0.kind == .batchRecipe},"Data reader recipe edges require workflow schema13.")
        try check(Set(document.edges.map(\.id)).count==document.edges.count && Set(document.edges.map(\.to)).count==document.edges.count,"Duplicate workflow edge identity or input occupancy.")
        for edge in document.edges {
            try check(named(edge.id,maximum:1024),"Invalid workflow edge identity.")
            try endpoint(edge.from);try endpoint(edge.to)
        }
        for domain in [WorkflowDomain.data,.model] {
            try check(document.edges.filter{$0.from.address.domain==domain && $0.to.address.domain==domain}.count<=256,"A data/model execution domain supports at most 256 edges.")
        }
        // Reuse existing bounds/pin/draft rules, without storing another editable root graph.
        // Configuration versions are still stored per primitive; executable mixed-version checks belong to the compiler.
        let root=CompositeGraph(nodes:document.nodes.compactMap {node in
            if case .modelPrimitive(let type,_,let config)=node.payload {return GraphNode(id:node.address.localID,type:type,name:node.name,x:node.x,y:node.y,config:config)}
            return nil
        },instances:document.nodes.compactMap {node in
            if case .modelBlock(let reference,let overrides)=node.payload {return CompositeInstance(id:node.address.localID,definition:reference,name:node.name,x:node.x,y:node.y,overrides:overrides)}
            return nil
        },edges:document.edges.filter{$0.kind == .tensor && $0.from.address.domain == .model && $0.to.address.domain == .model}.map {
            GraphEdge(id:$0.id,source:$0.from.address.localID,sourcePort:$0.from.port,target:$0.to.address.localID,targetPort:$0.to.port)
        })
        try check(document.schemaVersion>=11 || (document.definitions.allSatisfy { $0.composition == nil } && document.drafts.allSatisfy { $0.proposed.composition == nil }),"Native container policies require explicit workflow schema11 migration.")
        for definition in document.definitions {
            try check(document.schemaVersion>=15 || definition.body.nodes.allSatisfy {$0.parameterSettings == nil},"Layer settings in Blocks require workflow schema15.")
            try definitionJSONBounds(definition)
        }
        for draft in document.drafts {try check(document.schemaVersion>=15 || draft.proposed.body.nodes.allSatisfy {$0.parameterSettings == nil},"Layer settings in drafts require workflow schema15.");try definitionJSONBounds(draft.proposed)}
        let source=CompositeProjectDocument(schemaVersion:document.schemaVersion>=11 ? 3 : 2,name:document.name,root:root,definitions:document.definitions)
        var modelDrafts = document.drafts
        for index in modelDrafts.indices {
            if let origin = modelDrafts[index].migrationOrigin {
                try origin.validate(workflowSchemaVersion: document.schemaVersion)
                // Only the validated borrowed model projection uses a model origin.
                // The persisted workflow retains its exact original workflow version.
                modelDrafts[index].migrationOrigin = .model(modelSchemaVersion: 1, sourceSchemaVersion: 2)
            }
        }
        try ModelProjectCodec.validate(.blockModel(.init(projectID:document.projectID,revision:document.revision,source:source,drafts:modelDrafts,schemaVersion:document.schemaVersion>=11 ? 2 : 1)))
    }
    /// A reversible authoring transaction must persist this explicit migration.
    /// It adds no parent selection or inferred training configuration.
    public static func migrateToVersion3(_ document: WorkflowDocument) throws -> WorkflowDocument {
        try validate(document)
        var migrated = document
        migrated.schemaVersion = max(3,document.schemaVersion)
        try validate(migrated)
        return migrated
    }

    public static func migrateToVersion13(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(13,document.schemaVersion);try validate(migrated);return migrated
    }
    public static func migrateToBagInput(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(12,document.schemaVersion);try validate(migrated);return migrated
    }
    public static func migrateToVersion11(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(11,document.schemaVersion);try validate(migrated);return migrated
    }
    public static func migrateToVersion10(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(10,document.schemaVersion);try validate(migrated);return migrated
    }
    public static func migrateToVersion9(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(9,document.schemaVersion);try validate(migrated);return migrated
    }
    public static func migrateToVersion8(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(8,document.schemaVersion);try validate(migrated);return migrated
    }
    public static func migrateToVersion7(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(7,document.schemaVersion);try validate(migrated);return migrated
    }

    public static func migrateToVersion6(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document);var migrated=document;migrated.schemaVersion=max(6,document.schemaVersion);try validate(migrated);return migrated
    }

    public static func migrateToVersion5(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document)
        var migrated=document;migrated.schemaVersion=max(5,document.schemaVersion)
        try validate(migrated);return migrated
    }

    public static func migrateToVersion4(_ document: WorkflowDocument) throws -> WorkflowDocument {
        try validate(document)
        var migrated=document;migrated.schemaVersion=max(4,document.schemaVersion)
        try validate(migrated);return migrated
    }

    /// Explicit source migration only. The owner supplies the undo transaction and
    /// revision increment; decoding and historical replay never invoke this method.
    public static func migrateToVersion2(_ document:WorkflowDocument)throws->WorkflowDocument {
        try validate(document)
        guard document.schemaVersion==1 else {return document}
        var migrated=document
        migrated.schemaVersion=2
        for index in migrated.nodes.indices {
            if case .runSetup(var config)=migrated.nodes[index].payload {
                config["preparationCachePolicy"] = .string("off")
                migrated.nodes[index].payload = .runSetup(config:config)
            }
        }
        try validate(migrated)
        return migrated
    }
    public static func migrateModel(_ source:ModelProjectDocument,projectID:String)throws->WorkflowMigrationResult {
        try ModelProjectCodec.validate(source)
        let model:ModelAuthoringDocument
        switch source {
        case .legacyFlat(let graph):model=try ModelProjectCodec.promote(graph,projectID:projectID)
        case .blockModel(let existing):
            try check(existing.projectID==projectID,"Model migration preserves its parent identity; copying requires a separate explicit operation.")
            model=existing
        }
        let scope=model.source.root
        var nodes=scope.nodes.map {node in WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:scope.configurationVersion ?? 1,config:node.config),parameterSettings:node.parameterSettings)}
        nodes += scope.instances.map {node in WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelBlock(reference:node.definition,overrides:node.overrides))}
        let edges=scope.edges.map {edge in WorkflowEdge(id:edge.id,from:.init(address:.init(domain:.model,localID:edge.source),port:edge.sourcePort),to:.init(address:.init(domain:.model,localID:edge.target),port:edge.targetPort),kind:.tensor)}
        var document=WorkflowDocument(projectID:projectID,revision:model.revision,name:model.source.name,nodes:nodes,edges:edges,definitions:model.source.definitions,drafts:model.drafts)
        if scope.nodes.contains(where:{$0.parameterSettings != nil}) || model.source.definitions.contains(where:{$0.body.nodes.contains(where:{$0.parameterSettings != nil})}) {document.schemaVersion=15}
        try validate(document)
        return WorkflowMigrationResult(document:document,retainedExecution:model.source.execution,sourceBindings:[:])
    }
    public static func migrateData(_ source:DataFlowProject,projectID:String,registry:DataFlowOperatorRegistry?=nil)throws->WorkflowMigrationResult {
        try DataFlowCodec.validate(source)
        if !source.graph.edges.isEmpty {
            guard let registry,registry.schemaVersion==1,Set(registry.operators.map(\.id)).count==registry.operators.count else {
                throw failure("Connected data migration requires an authoritative operator registry with unique identities.")
            }
        }
        let nodes=source.graph.nodes.map {node in
            let editor=source.editor[node.id] ?? DataFlowEditorNode()
            return WorkflowNode(address:.init(domain:.data,localID:node.id),name:editor.name,x:editor.x,y:editor.y,payload:.dataOperator(operatorID:node.operatorID,config:node.config))
        }
        let edges=try source.graph.edges.map {edge in
            guard let producer=source.graph.nodes.first(where:{$0.id==edge.from.node}),
                  let definition=registry?.operatorDefinition(id:producer.operatorID),
                  Set(definition.outputs.map(\.id)).count==definition.outputs.count,
                  let port=definition.outputs.first(where:{$0.id==edge.from.port}) else {
                throw failure("Cannot classify a data connection without its declared producer output; provide the matching registry or resolve the connection first.")
            }
            let kind:WorkflowEdgeKind
            switch port.kind {
            case "table":kind = .table
            case "dataset":kind = .dataset
            case "state" where port.stateKind=="fittedTransform":kind = .fittedState
            default:throw failure("The producer output kind has no supported workflow connection declaration.")
            }
            return WorkflowEdge(id:edge.id,from:.init(address:.init(domain:.data,localID:edge.from.node),port:edge.from.port),to:.init(address:.init(domain:.data,localID:edge.to.node),port:edge.to.port),kind:kind)
        }
        let document=WorkflowDocument(projectID:projectID,name:source.name,nodes:nodes,edges:edges)
        try validate(document)
        let bindings=Dictionary(uniqueKeysWithValues:source.sourceBindings.map{(WorkflowNodeAddress(domain:.data,localID:$0.key),$0.value)})
        return WorkflowMigrationResult(document:document,retainedExecution:nil,sourceBindings:bindings)
    }
    private static func definitionJSONBounds(_ definition:CompositeDefinition)throws {
        for node in definition.body.nodes {try json(.object(node.config))}
        for instance in definition.body.instances {try json(.object(instance.overrides))}
        for parameter in definition.parameters {try json(parameter.field.defaultValue)}
    }
    private static func keys(_ config:[String:JSONValue],_ allowed:Set<String>)throws {try check(Set(config.keys).isSubset(of:allowed),"Unknown or nonportable control/source configuration field.")}
    private static func address(_ value:WorkflowNodeAddress)throws {try check(named(value.localID),"Invalid qualified local node identity.")}
    private static func endpoint(_ value:WorkflowEndpoint)throws {try address(value.address);try check(named(value.port),"Invalid workflow port identity.")}
    private static func named(_ value:String,maximum:Int=128)->Bool {!value.isEmpty && value.unicodeScalars.count<=maximum}
    private static func check(_ value:Bool,_ message:String)throws {if !value {throw failure(message)}}
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
    private static func json(_ value:JSONValue,depth:Int=0)throws {
        try check(depth<=32,"Workflow configuration exceeds nesting depth 32.")
        switch value {
        case .number(let value):try check(value.isFinite,"Nonfinite workflow configuration.")
        case .string(let value):try check(value.utf8.count<=65_536,"Workflow configuration string exceeds bounds.")
        case .array(let values):try check(values.count<=4096,"Workflow configuration array exceeds bounds.");for value in values {try json(value,depth:depth+1)}
        case .object(let values):try check(values.count<=4096,"Workflow configuration object exceeds bounds.");for (key,value) in values {try check(key.utf8.count<=16_384,"Workflow configuration key exceeds bounds.");try json(value,depth:depth+1)}
        default:break
        }
    }
    private static func object(_ value:JSONValue,required:Set<String>,optional:Set<String> = [])throws->[String:JSONValue] {
        guard case .object(let result)=value,required.isSubset(of:Set(result.keys)),Set(result.keys).isSubset(of:required.union(optional)) else {throw failure("Unknown or missing workflow structure field.")}
        return result
    }
    private static func array(_ value:JSONValue)throws->[JSONValue] {guard case .array(let values)=value else {throw failure("Expected workflow array.")};return values}
    private static func structure(_ value:JSONValue)throws {
        let root=try object(value,required:["schemaVersion","kind","projectID","revision","name","nodes","edges","definitions","drafts"],optional:["guidedBuild","modelInputs","modelSourceVersions","namedModelInputs","templateOrigin"])
        if let origin=root["templateOrigin"], origin != .null { _ = try object(origin,required:["id","version"]) }
        if let progress=root["guidedBuild"],progress != .null {
            _ = try object(progress,required:["courseID","version","completedSteps"])
        }
        try check(root["kind"] == .string("workflowProject") && [JSONValue.integer(1),.integer(2),.integer(3),.integer(4),.integer(5),.integer(6),.integer(7),.integer(8),.integer(9),.integer(10),.integer(11),.integer(12),.integer(13),.integer(14),.integer(15),.integer(16),.integer(17),.integer(18),.integer(19),.integer(20),.integer(21),.integer(22),.integer(23),.integer(24)].contains(root["schemaVersion"] ?? .null),"Unknown workflow kind/version; no legacy fallback.")
        for value in try array(root["nodes"]!) {
            let node=try object(value,required:["address","name","x","y","payload"],optional:["parameterSettings"])
            try addressStructure(node["address"]!)
            guard case .object(let payload)=node["payload"],case .string(let kind)=payload["kind"] else {throw failure("Missing workflow payload tag.")}
            let required:Set<String>
            switch kind {
            case "dataOperator":required=["kind","operatorID","config"]
            case "modelPrimitive","optimizerRecipe","schedulerRecipe","lossRecipe","initializationRecipe","datasetRecipe","batchRecipe","gradientPolicyRecipe","trainabilityRecipe":required=["kind","type","configurationVersion","config"]
            case "modelBlock":required=["kind","reference","overrides"]
            case "modelDefinition":required=["kind"]
            case "trainingData","runSetup","train","frozenModel","evaluateTest","saveModel","generateText","gameSession","estimator":required=["kind","config"]
            default:throw failure("Unknown workflow payload kind.")
            }
            _=try object(node["payload"]!,required:required)
            if let reference=payload["reference"] {try referenceStructure(reference)}
        }
        for value in try array(root["edges"]!) {
            let edge=try object(value,required:["id","from","to","kind"])
            for key in ["from","to"] {let endpoint=try object(edge[key]!,required:["address","port"]);try addressStructure(endpoint["address"]!)}
        }
        for value in try array(root["definitions"]!) {try definitionStructure(value)}
        for value in try array(root["drafts"]!) {
            let draft=try object(value,required:["projectID","instancePath","baseReference","baseRevision","proposed"],optional:["migrationOrigin"])
            if let origin = draft["migrationOrigin"] {
                try CompositeDraftMigrationOrigin.validateWire(origin)
            }
            try referenceStructure(draft["baseReference"]!);try definitionStructure(draft["proposed"]!)
        }
    }
    private static func addressStructure(_ value:JSONValue)throws {_=try object(value,required:["domain","localID"])}
    private static func referenceStructure(_ value:JSONValue)throws {let ref=try object(value,required:["id","version"]);_=try object(ref["version"]!,required:["major","minor","patch"])}
    private static func definitionStructure(_ value:JSONValue)throws {
        let definition=try object(value,required:["reference","title","body","inputs","outputs","parameters"],optional:["integerProducts","composition","appearance"])
        if let appearance=definition["appearance"] {try check(appearance.stringValue.flatMap(CompositeAppearance.init(rawValue:)) != nil,"Unknown container appearance identity.")}
        if let policy=definition["composition"] {
            _ = try JSONDecoder().decode(CompositionPolicy.self,from:JSONEncoder().encode(policy))
        }
        try referenceStructure(definition["reference"]!)
        let body=try object(definition["body"]!,required:["nodes","instances","edges"],optional:["configurationVersion"])
        for value in try array(body["nodes"]!) {_=try object(value,required:["id","type","name","x","y","config"],optional:["parameterSettings"])}
        for value in try array(body["instances"]!) {let instance=try object(value,required:["id","definition","name","x","y","overrides"]);try referenceStructure(instance["definition"]!)}
        for value in try array(body["edges"]!) {_=try object(value,required:["id","source","sourcePort","target","targetPort"])}
        for value in try array(definition["inputs"]!) {let input=try object(value,required:["id","targets"]);for target in try array(input["targets"]!) {_=try object(target,required:["componentID","portID"])}}
        for value in try array(definition["outputs"]!) {let output=try object(value,required:["id","source"]);_=try object(output["source"]!,required:["componentID","portID"])}
        for value in try array(definition["parameters"]!) {
            let parameter=try object(value,required:["field","targets"])
            let field=try object(parameter["field"]!,required:["key","title","defaultValue"],optional:["kind","min","max","minItems","maxItems","allowsNull","choices"])
            if let choices=field["choices"] {
                for choice in try array(choices) {_=try object(choice,required:["value","title"])}
            }
            let decoded=try JSONDecoder().decode(ComponentField.self,from:JSONEncoder().encode(parameter["field"]!))
            try check(decoded.isValidDefinition,"Invalid exposed field definition.")
            for target in try array(parameter["targets"]!) {_=try object(target,required:["componentID","field"])}
        }
        if let bindings=definition["integerProducts"] {
            let products=try array(bindings)
            try check(products.count<=128,"At most 128 integer product bindings are supported.")
            for value in products {
                let product=try object(value,required:["lhs","rhs","targets"])
                guard let lhs=product["lhs"]?.stringValue,let rhs=product["rhs"]?.stringValue else {throw failure("Integer product operands must name exposed parameters.")}
                try check(named(lhs) && named(rhs),"Invalid integer product operand identity.")
                let targets=try array(product["targets"]!)
                try check((1...256).contains(targets.count),"Integer products require 1–256 targets.")
                for target in targets {_=try object(target,required:["componentID","field"])}
            }
        }
    }
}
