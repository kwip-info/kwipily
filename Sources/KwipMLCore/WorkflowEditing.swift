import Foundation

public struct WorkflowSelection: Equatable {
    public var nodeAddresses: Set<WorkflowNodeAddress>
    public var edgeIDs: Set<String>
    public init(nodeAddresses: Set<WorkflowNodeAddress> = [], edgeIDs: Set<String> = []) {
        self.nodeAddresses = nodeAddresses
        self.edgeIDs = edgeIDs
    }
}

public enum WorkflowCommand {
    case advanceGuidedBuild(expected:WorkflowGuidedBuildProgress)
    case inlineContainer(root:WorkflowNodeAddress,command:WorkflowInlineContainer.Command,retirePreviousPrivateReference:CompositeDefinitionReference? = nil)
    case insert(WorkflowNode)
    case insertBlock(WorkflowNode, definition: CompositeDefinition)
    case insertConnected(WorkflowNode, incoming:[WorkflowEdge])
    case rename(address: WorkflowNodeAddress, name: String)
    case configure(address: WorkflowNodeAddress, config: [String: JSONValue])
    case setLayerParameters(address:WorkflowNodeAddress,settings:[String:JSONValue]?)
    case setActivation(address:WorkflowNodeAddress,function:ActivationConfiguration.Function,configuration:[String:JSONValue]? = nil)
    case setPooling(address:WorkflowNodeAddress,variant:PoolingConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setConvolution(address:WorkflowNodeAddress,variant:ConvolutionConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setBilinear(address:WorkflowNodeAddress,configuration:[String:JSONValue])
    case setEmbedding(address:WorkflowNodeAddress,configuration:[String:JSONValue])
    case setAttention(address:WorkflowNodeAddress,configuration:[String:JSONValue])
    case setEmbeddingBag(address:WorkflowNodeAddress,configuration:[String:JSONValue])
    case setRecurrent(address:WorkflowNodeAddress,method:RecurrentSequenceConfiguration.Method,configuration:[String:JSONValue]? = nil)
    case setTensorOperation(address:WorkflowNodeAddress,type:String,configuration:[String:JSONValue])
    case setReshape(address:WorkflowNodeAddress,variant:ReshapeAuthoringConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setLinear(address:WorkflowNodeAddress,variant:LinearConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setNormalization(address:WorkflowNodeAddress,variant:NormalizationConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case removeModelInput(definition:WorkflowNodeAddress,inputName:String)
    case setModelInput(definition:WorkflowNodeAddress,type:String,configuration:[String:JSONValue],destinations:[WorkflowEndpoint],inputName:String? = nil)
    case move(addresses: Set<WorkflowNodeAddress>, offset: GraphOffset)
    case delete(WorkflowSelection)
    case duplicate(addresses: Set<WorkflowNodeAddress>, offset: GraphOffset)
    case connect(from: WorkflowEndpoint, to: WorkflowEndpoint)
    case reconnect(edgeID: String, from: WorkflowEndpoint, to: WorkflowEndpoint)
}

public struct WorkflowCopyMap: Equatable {
    public var nodes: [WorkflowNodeAddress: WorkflowNodeAddress]
    public var edges: [String: String]
    public init(nodes: [WorkflowNodeAddress: WorkflowNodeAddress] = [:], edges: [String: String] = [:]) {
        self.nodes = nodes
        self.edges = edges
    }
}

public struct WorkflowEditResult: Equatable {
    public let document: WorkflowDocument
    public let selection: WorkflowSelection
    public let copies: WorkflowCopyMap
    public init(document:WorkflowDocument,selection:WorkflowSelection,copies:WorkflowCopyMap){self.document=document;self.selection=selection;self.copies=copies}
}

/// Atomic authoring edits, not an execution validator. Configuration may remain incomplete.
/// Connections prove declared kind/cardinality and acyclicity only; the compiler owns
/// tensor shapes, partition lineage, required inputs and executable configuration.
public enum WorkflowEditing {
    public static func apply(
        _ command: WorkflowCommand,
        to document: WorkflowDocument,
        selection: WorkflowSelection = .init(),
        modelRegistry: ComponentRegistry,
        dataRegistry: DataFlowOperatorRegistry,
        makeID: () -> String = { UUID().uuidString }
    ) throws -> WorkflowEditResult {
        _ = try WorkflowCodec.encode(document)
        var next = document
        var selected = selection
        var copies = WorkflowCopyMap()
        // Reserve raw IDs across domains as well as edges. Existing qualified duplicates
        // are valid, but generated identities must not introduce ambiguous raw copies.
        var reserved = Set(document.nodes.map { $0.address.localID }).union(document.edges.map(\.id))
        func freshID(maximum: Int) throws -> String {
            let id = makeID()
            guard !id.isEmpty, id.unicodeScalars.count <= maximum, reserved.insert(id).inserted else {
                throw failure(.invalidDocument, "A generated workflow identity is invalid or already in use.")
            }
            return id
        }
        func index(_ address: WorkflowNodeAddress) throws -> Int {
            guard let index = next.nodes.firstIndex(where: { $0.address == address }) else {
                throw failure(.invalidSelection, "The selected workflow node no longer exists.", address)
            }
            return index
        }
        func requireNodes(_ addresses: Set<WorkflowNodeAddress>) throws {
            for address in addresses { _ = try index(address) }
        }
        func requireOffset(_ offset: GraphOffset) throws {
            guard offset.x.isFinite, offset.y.isFinite else {
                throw failure(.configuration, "Movement must use finite coordinates.")
            }
        }
        func edge(from: WorkflowEndpoint, to: WorkflowEndpoint, id: String, in source: WorkflowDocument,allowLegacySinkOutput:Bool=false) throws -> WorkflowEdge {
            guard let producer = source.nodes.first(where: { $0.address == from.address }),
                  let consumer = source.nodes.first(where: { $0.address == to.address }) else {
                throw failure(.invalidPort, "Connect existing workflow nodes.")
            }
            if !allowLegacySinkOutput,case .dataOperator(let operatorID,_)=producer.payload,
               dataRegistry.operatorDefinition(id:operatorID)?.isCanvasSink == true {
                throw failure(.invalidPort,"View ends this branch. Connect the upstream table directly to another component.",from.address,from.port)
            }
            let outputs = try WorkflowPortContracts.resolve(producer, document: source, modelRegistry: modelRegistry, dataRegistry: dataRegistry).outputs
            let inputs = try WorkflowPortContracts.resolve(consumer, document: source, modelRegistry: modelRegistry, dataRegistry: dataRegistry).inputs
            guard let output = outputs.first(where: { $0.id == from.port }) else {
                throw failure(.invalidPort, "The selected output is not declared by this component.", from.address, from.port)
            }
            guard let input = inputs.first(where: { $0.id == to.port }) else {
                throw failure(.invalidPort, "The selected input is not declared by this component.", to.address, to.port)
            }
            let tableKinds: Set<WorkflowEdgeKind> = [.table, .dataset]
            guard output.kind == input.kind || (tableKinds.contains(output.kind) && tableKinds.contains(input.kind)) else {
                throw failure(.portKind, "A \(output.kind.rawValue) output cannot connect to a \(input.kind.rawValue) input.", to.address, to.port)
            }
            if let maximum = input.maximumConnections, source.edges.filter({ $0.to == to }).count >= maximum {
                throw failure(.invalidPort, "This input has reached its connection limit; reconnect its existing wire.", to.address, to.port)
            }
            if let maximum = output.maximumConnections, source.edges.filter({ $0.from == from }).count >= maximum {
                throw failure(.invalidPort, "This output has reached its connection limit.", from.address, from.port)
            }
            // A new source→target wire closes a cycle exactly when target reaches source.
            let outgoing = Dictionary(grouping: source.edges, by: { $0.from.address })
            var pending = [to.address], visited = Set<WorkflowNodeAddress>()
            while let address = pending.popLast() {
                if address == from.address { throw failure(.cycle, "A workflow connection cannot create a cycle.", to.address, to.port) }
                if visited.insert(address).inserted { pending += (outgoing[address] ?? []).map { $0.to.address } }
            }
            return .init(id: id, from: from, to: to, kind: output.kind)
        }

        switch command {
        case .inlineContainer(let root,let command,let retirePreviousPrivateReference):
            let edited=try WorkflowInlineContainer.apply(command,to:document,root:root,registry:modelRegistry,dataRegistry:dataRegistry,retirePreviousPrivateReference:retirePreviousPrivateReference,idProvider:makeID)
            return .init(document:edited,selection:selection,copies:.init())
        case .advanceGuidedBuild(let expected):
            let plan=try WorkflowGuidedBuildPlan.parallelClassifier(projectID:document.projectID,version:expected.version)
            guard document.guidedBuild == expected,expected.courseID == plan.id,expected.version == plan.version else {
                throw failure(.invalidDocument,"The guided course changed or is unsupported. Reopen its current instructions before advancing.")
            }
            let result=try apply(plan.insertion(at:expected.completedSteps),to:document,selection:selection,modelRegistry:modelRegistry,dataRegistry:dataRegistry,makeID:makeID)
            var advanced=result.document
            // A guided Model carries the pinned source semantics of its course seed.
            // Generic model placement remains current-version authoring. Copy metadata
            // only for the owner inserted now; never overwrite a learner's existing owner.
            for owner in advanced.nodes where result.selection.nodeAddresses.contains(owner.address) {
                guard WorkflowModelBoundary.isDefinition(owner) else {continue}
                let id=owner.address.localID
                if let version=plan.completeDocument.modelSourceVersions?[id] {
                    var versions=advanced.modelSourceVersions ?? [:];versions[id]=version
                    advanced.modelSourceVersions=versions
                }
                if let input=plan.completeDocument.modelInputs?[id] {
                    var owners=advanced.modelInputs ?? [:];owners[id]=input
                    advanced.modelInputs=owners
                }
            }
            advanced.guidedBuild = .init(courseID:plan.id,version:plan.version,completedSteps:expected.completedSteps+1)
            _ = try WorkflowCodec.encode(advanced)
            return .init(document:advanced,selection:result.selection,copies:result.copies)
        case .insert(let node), .insertConnected(let node, _), .insertBlock(let node, _):
            if case .trainingData=node.payload,next.guidedBuild == nil || next.guidedBuild!.version>=5 {throw failure(.configuration,"Data mapping now belongs to Train. Configure its input and target columns there.",node.address)}
            guard !WorkflowTrainRunSetup.isLegacyNode(node) else {throw failure(.configuration,"Run settings now belong to Train. Run setup cannot be placed as a separate tool.",node.address)}
            guard !RetiredTrainingRecipeMigration.isRetired(node) else {throw failure(.configuration,"Training settings now belong to Train or their layer. Retired recipe tools cannot be placed.",node.address)}
            if case .insertBlock(_, let definition) = command {
                if definition.composition != nil,next.schemaVersion<11 {
                    WorkflowModelInput.migrateOwnershipMetadata(&next)
                    next.schemaVersion=max(11,next.schemaVersion)
                }
                guard case .modelBlock(let reference, _) = node.payload,
                      reference == definition.reference else {
                    throw failure(.invalidModel, "The inserted Block must reference its supplied definition exactly.", node.address)
                }
                if let pinned = next.definitions.first(where: { $0.reference == reference }) {
                    guard pinned == definition else {
                        throw failure(.invalidModel, "A different Block definition already uses this pinned reference.", node.address)
                    }
                } else {
                    next.definitions.append(definition)
                }
                // Validate against the same local collection that will be published with
                // the node. Any failure discards both changes without altering its pins.
                try CompositeDefinitionValidation.validate(definition, definitions: next.definitions, registry: modelRegistry)
            }
            guard !next.nodes.contains(where: { $0.address == node.address }), node.address.domain == node.payload.domain else {
                throw failure(.invalidDocument, "The inserted node has a duplicate address or an incompatible payload domain.", node.address)
            }
            if case .gameSession=node.payload {next.schemaVersion=max(22,next.schemaVersion)}
            if case .generateText=node.payload {next=try WorkflowTrainedModelSource.promoteAuthoring(next);next.schemaVersion=max(19,next.schemaVersion)}
            if case .saveModel=node.payload {next=try WorkflowTrainedModelSource.promoteAuthoring(next)}
            if case .train(let config)=node.payload,config["runSetup"] != nil {next=try WorkflowTrainRunSetup.migrateAuthoringSchema(next)}
            if case .train(let config)=node.payload,config["dataMapping"] != nil {next=try WorkflowTrainMapping.migrate(next);next.schemaVersion=max(18,next.schemaVersion)}
            switch node.payload {
            case .train(let config) where config["datasetAccess"] != nil || config["batching"] != nil || !WorkflowTrainRecipes.roleKeys.isDisjoint(with:config.keys): next = try WorkflowTrainRecipes.migrateAuthoringSchema(next)
            case .frozenModel, .evaluateTest, .estimator:
                next = try WorkflowCodec.migrateToVersion5(next)
            default: break
            }
            // Known contracts (including exact Block pins) are required for new nodes;
            // unknown imported draft nodes remain editable/removable through other commands.
            _ = try WorkflowPortContracts.resolve(node, document: next, modelRegistry: modelRegistry, dataRegistry: dataRegistry)
            if case .frozenModel(let config)=node.payload,!config.isEmpty {next.schemaVersion=max(20,next.schemaVersion)}
            next.nodes.append(node)
            if case .insertConnected(_,let incoming)=command {
                for proposed in incoming {
                    guard proposed.to.address == node.address,
                          !next.edges.contains(where:{$0.id == proposed.id}) else {
                        throw failure(.invalidDocument,"A staged wire must have a fresh identity and target the inserted component.",node.address)
                    }
                    let checked=try edge(from:proposed.from,to:proposed.to,id:proposed.id,in:next)
                    guard checked.kind == proposed.kind else {
                        throw failure(.portKind,"The staged wire kind does not match its declared source.",proposed.from.address,proposed.from.port)
                    }
                    next.edges.append(checked)
                }
            }
            selected = .init(nodeAddresses: [node.address])
        case .rename(let address, let name):
            next.nodes[try index(address)].name = name
        case .setAttention(let address,let configuration):
            let position=try index(address)
            next.nodes[position]=try AttentionConfiguration.applying(configuration,to:next.nodes[position],connectedInputPorts:Set(next.edges.filter {$0.to.address==address}.map {$0.to.port}),connectedOutputPorts:Set(next.edges.filter {$0.from.address==address}.map {$0.from.port}),registry:modelRegistry)
        case .setEmbeddingBag(let address,let configuration):
            let position=try index(address)
            next.nodes[position]=try EmbeddingBagConfiguration.applying(configuration,to:next.nodes[position],connectedInputPorts:Set(next.edges.filter {$0.to.address==address}.map {$0.to.port}),registry:modelRegistry)
        case .setEmbedding(let address,let configuration):
            let position=try index(address)
            next.nodes[position]=try EmbeddingConfiguration.applying(configuration,to:next.nodes[position],registry:modelRegistry)
        case .setBilinear(let address,let configuration):
            let position=try index(address)
            next.nodes[position]=try BilinearConfiguration.applying(configuration,to:next.nodes[position],registry:modelRegistry)
        case .setRecurrent(let address,let method,let configuration):
            let position=try index(address)
            let inputs=Set(next.edges.filter {$0.to.address==address}.map {$0.to.port})
            let outputs=Set(next.edges.filter {$0.from.address==address}.map {$0.from.port})
            next.nodes[position]=try RecurrentAuthoringConfiguration.switching(next.nodes[position],to:method,configuration:configuration,connectedInputPorts:inputs,connectedOutputPorts:outputs,registry:modelRegistry)
        case .setTensorOperation(let address,let type,let configuration):
            let position=try index(address)
            let contracts=TensorOperationAuthoring.incomingContracts(address:address,document:next,registry:modelRegistry,dataRegistry:dataRegistry)
            next.nodes[position]=try TensorOperationAuthoring.switching(next.nodes[position],to:type,configuration:configuration,
                connectedInputs:Set(next.edges.filter {$0.to.address==address}.map {$0.to.port}),connectedOutputs:Set(next.edges.filter {$0.from.address==address}.map {$0.from.port}),
                inputs:contracts.mapValues(\.layout),registry:modelRegistry,inputContracts:contracts)
        case .setReshape(let address,let variant,let configuration):
            let position=try index(address)
            next.nodes[position]=try ReshapeAuthoringConfiguration.switching(next.nodes[position],to:variant,configuration:configuration,registry:modelRegistry)
        case .setLinear(let address,let variant,let configuration):
            let position=try index(address)
            let outputs=Set(next.edges.filter {$0.from.address==address}.map {$0.from.port})
            next.nodes[position]=try LinearConfiguration.switching(next.nodes[position],to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:modelRegistry)
        case .setNormalization(let address,let variant,let configuration):
            let position=try index(address)
            let outputs=Set(next.edges.filter {$0.from.address==address}.map {$0.from.port})
            next.nodes[position]=try NormalizationConfiguration.switching(next.nodes[position],to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:modelRegistry)
        case .setConvolution(let address,let variant,let configuration):
            let position=try index(address)
            let outputs=Set(next.edges.filter {$0.from.address==address}.map {$0.from.port})
            next.nodes[position]=try ConvolutionConfiguration.switching(next.nodes[position],to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:modelRegistry)
        case .setPooling(let address,let variant,let configuration):
            let position=try index(address)
            let outputs=Set(next.edges.filter {$0.from.address==address}.map {$0.from.port})
            next.nodes[position]=try PoolingConfiguration.switching(next.nodes[position],to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:modelRegistry)
        case .setActivation(let address,let function,let configuration):
            let position=try index(address)
            next.nodes[position]=try ActivationConfiguration.switching(next.nodes[position],to:function,configuration:configuration,registry:modelRegistry)
        case .setLayerParameters(let address,let settings):
            let position=try index(address)
            guard case .modelPrimitive=next.nodes[position].payload else {throw failure(.configuration,"Select a primitive layer.",address)}
            if let settings {try LayerParameterSettings.validate(settings)}
            WorkflowModelInput.migrateOwnershipMetadata(&next)
            next.schemaVersion=max(15,next.schemaVersion)
            next.nodes[position].parameterSettings=settings
        case .configure(let address, let config):
            let position = try index(address)
            guard !RetiredTrainingRecipeMigration.isRetired(next.nodes[position]) else {throw failure(.configuration,"This retired tool is preserved for repair. Upgrade its settings into Train; its original configuration cannot be edited here.",address)}
            switch next.nodes[position].payload {
            case .dataOperator(let id, _):
                // Validate accepted mutations using the same inspector/runtime authority.
                // Loading older incomplete documents remains available for repair.
                if id=="kwip.data.join" {try JoinProjectionContract.validate(config)}
                if id==PrepareMediaConfiguration.operatorID {try PrepareMediaConfiguration.validate(config)}
                if id=="kwip.data.outputDelimited",APITableSettings.isAPI(config) {
                    guard case .object(let settings)=config["api"] else {throw DataFlowContractError("API Table Output requires an api settings object.",apiIssue:.outputEnvelope)}
                    let normalized=try APITableSettings.normalized(settings)
                    guard normalized["direction"] == .string("output") else {throw DataFlowContractError("Table Output requires API output direction.",apiIssue:.outputEnvelope)}
                }
                next.nodes[position].payload = .dataOperator(operatorID: id, config: config)
            case .modelPrimitive(let type, let version, _):
                if type=="output",config["task"] != nil {next.schemaVersion=max(23,next.schemaVersion)}
                next.nodes[position].payload = .modelPrimitive(type: type, configurationVersion: version, config: config)
            case .modelBlock(let reference, _): next.nodes[position].payload = .modelBlock(reference: reference, overrides: config)
            case .modelDefinition:
                guard config.isEmpty else { throw failure(.configuration, "Model definition carriers have no configuration fields.", address) }
            case .optimizerRecipe,.lossRecipe,.initializationRecipe,.gradientPolicyRecipe,.trainabilityRecipe,.schedulerRecipe,.datasetRecipe,.batchRecipe:
                throw failure(.configuration,"Retired training recipe tools require project repair.",address)
            case .trainingData: next.nodes[position].payload = .trainingData(config: config)
            case .runSetup: next.nodes[position].payload = .runSetup(config: config)
            case .estimator:
                if PrebuiltModelConfiguration.isPrebuilt(config) {_ = try PrebuiltModelConfiguration.decode(config)}
                var proposed=next.nodes[position];proposed.payload = .estimator(config:config)
                let ports=try WorkflowPortContracts.resolve(proposed,document:next,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
                for edge in next.edges where edge.to.address==address || edge.from.address==address {
                    let port=edge.to.address==address ? edge.to.port:edge.from.port
                    let candidates=edge.to.address==address ? ports.inputs:ports.outputs
                    guard candidates.contains(where:{$0.id==port && $0.kind==edge.kind}) else {throw failure(.invalidPort,"Disconnect the incompatible \(port) connection before changing the Prebuilt model task or operation.",address,port)}
                }
                next.nodes[position]=proposed
            case .train:
                if config["selfPlay"] != nil {next.schemaVersion=max(23,next.schemaVersion)}
                if config["fineTuning"] != nil {next.schemaVersion=max(21,next.schemaVersion)}
                if case .object(let mapping)=config["dataMapping"],mapping["namedInputPlan"] != nil {next.schemaVersion=max(19,next.schemaVersion)}
                if config["runSetup"] != nil {next=try WorkflowTrainRunSetup.migrateAuthoringSchema(next)}
                if config["datasetAccess"] != nil || config["batching"] != nil || !WorkflowTrainRecipes.roleKeys.isDisjoint(with:config.keys) {next = try WorkflowTrainRecipes.migrateAuthoringSchema(next)}
                next.nodes[try index(address)].payload = .train(config: config)
            case .frozenModel:
                _ = try LoadModelConfiguration.decode(config)
                if !config.isEmpty {next.schemaVersion=max(20,next.schemaVersion)}
                next.nodes[position].payload = .frozenModel(config: config)
            case .evaluateTest: next.nodes[position].payload = .evaluateTest(config: config)
            case .gameSession:
                let settings = try GameSessionConfiguration(config)
                for edge in next.edges where edge.to.address==address {
                    guard settings.inputPorts.contains(where:{$0.id==edge.to.port && $0.kind==edge.kind}) else {throw failure(.portKind,"Disconnect the incompatible policy wire before changing its source.",address,edge.to.port)}
                }
                next.schemaVersion=max(22,next.schemaVersion)
                next.nodes[position].payload = .gameSession(config:config)
            case .generateText:
                let settings=try TextGenerationConfiguration(config)
                guard settings.tableMode || !next.edges.contains(where:{$0.to.address==address && $0.to.port=="prompts"}) else {throw failure(.invalidPort,"Disconnect the Prompts table before choosing One prompt.",address,"prompts")}
                next.nodes[position].payload = .generateText(config: config)
            case .saveModel: next.nodes[position].payload = .saveModel(config: config)
            }
        case .setModelInput(let definition,let type,let configuration,let destinations,let inputName):
            guard WorkflowModelInput.componentIDs.contains(type),Set(destinations).count==destinations.count else {
                throw failure(.configuration,"Choose a supported model input and unique entry ports.",definition)
            }
            if let inputName {guard inputName.range(of:"^[A-Za-z][A-Za-z0-9_]{0,63}$",options:.regularExpression) != nil else {throw failure(.configuration,"Use an input name starting with a letter, followed by letters, digits or underscores.",definition)}}
            var existing=try WorkflowModelInput.boundary(for:definition,in:next,name:inputName)
            if inputName != nil,next.namedModelInputs?[definition.localID] == nil,let previous=existing {
                next.namedModelInputs=(next.namedModelInputs ?? [:]).merging([definition.localID:["input":previous.address.localID]]) {_,new in new}
                if inputName != "input" {existing=nil}
            }
            if let existing,!WorkflowModelInput.isBoundary(existing) {
                throw failure(.configuration,"Bag Input keeps its named packet connections on the canvas.",definition)
            }
            let version:Int
            if let existing,case .modelPrimitive(let oldType,let oldVersion,_)=existing.payload {
                guard oldType==type || oldVersion==2 else {throw failure(.configuration,"Migrate the legacy model configuration before changing its input kind.",definition)}
                version=oldVersion
            } else {
                let region=WorkflowModelInput.ancestors(of:definition,in:next)
                let versions=Set(next.nodes.compactMap {node->Int? in
                    guard region.contains(node.address),case .modelPrimitive(_,let version,_)=node.payload else {return nil};return version
                })
                guard versions.count<=1 else {throw failure(.mixedModelVersions,"Repair mixed model configuration versions before creating its input.",definition)}
                version=versions.first ?? 2
            }
            guard let component=modelRegistry.component(type:type),Set(configuration.keys)==Set(component.fields(configuration:configuration).map(\.key)),component.fields(configuration:configuration).allSatisfy({field in configuration[field.key].map(field.accepts) ?? false}) else {
                throw failure(.configuration,"Input dimensions are outside the supported model contract.",definition)
            }
            if type=="input",configuration["shape"] != nil {_ = try FloatingInputConfiguration(configuration)}
            if type=="index_input" {_ = try IndexInputConfiguration(values:configuration)}
            let address=try existing?.address ?? WorkflowNodeAddress(domain:.model,localID:freshID(maximum:128))
            if let position=next.nodes.firstIndex(where:{$0.address==address}) {
                next.nodes[position].payload = .modelPrimitive(type:type,configurationVersion:version,config:configuration)
            } else {
                next.nodes.append(.init(address:address,payload:.modelPrimitive(type:type,configurationVersion:version,config:configuration)))
            }
            // Ownership survives an empty destination list, so settings stay editable
            // while the user repairs or builds the model. Worker graph IDs stay intact.
            WorkflowModelInput.migrateOwnershipMetadata(&next)
            var owners=next.modelInputs ?? [:]
            if inputName == nil || owners[definition.localID] == nil {owners[definition.localID]=next.namedModelInputs?[definition.localID]?["input"] ?? address.localID};next.modelInputs=owners
            if let inputName {
                var all=next.namedModelInputs ?? [:];var names=all[definition.localID] ?? [:]
                guard names[inputName] != nil || names.count<8 else {throw failure(.configuration,"A model supports at most eight named inputs.",definition)}
                names[inputName]=address.localID;all[definition.localID]=names;next.namedModelInputs=all;next.schemaVersion=max(19,next.schemaVersion)
            }
            let original=Dictionary(uniqueKeysWithValues:next.edges.filter {$0.from.address==address && $0.from.port=="out"}.map {($0.to,$0)})
            next.edges.removeAll {$0.from.address==address && $0.from.port=="out" && !destinations.contains($0.to)}
            for target in destinations {
                guard target.address.domain == .model,target.address != address else {throw failure(.invalidPort,"Choose a model component input port.",definition)}
                if let retained=original[target] {
                    // Validate retained edges too; stale port names must remain actionable.
                    var check=next;check.edges.removeAll {$0.id==retained.id}
                    _ = try edge(from:retained.from,to:target,id:retained.id,in:check)
                } else {
                    next.edges.append(try edge(from:.init(address:address,port:"out"),to:target,id:freshID(maximum:1024),in:next))
                }
            }
            selected = .init(nodeAddresses:[definition])
        case .removeModelInput(let definition,let inputName):
            guard var all=next.namedModelInputs,var names=all[definition.localID],names.count>1,let id=names[inputName] else {throw failure(.configuration,"Keep at least one model input.",definition)}
            let address=WorkflowNodeAddress(domain:.model,localID:id)
            guard !next.edges.contains(where:{$0.from.address==address || $0.to.address==address}) else {throw failure(.configuration,"Disconnect this input’s entry ports before removing it.",definition)}
            names.removeValue(forKey:inputName);all[definition.localID]=names;next.namedModelInputs=all
            if next.modelInputs?[definition.localID]==id {next.modelInputs?[definition.localID]=names[names.keys.sorted()[0]]}
            let retained=all.values.contains {$0.values.contains(id)} || (next.modelInputs ?? [:]).values.contains(id)
            if !retained {next.nodes.removeAll {$0.address==address}}
            selected = .init(nodeAddresses:[definition])
        case .move(let addresses, let offset):
            try requireNodes(addresses)
            try requireOffset(offset)
            for position in next.nodes.indices where addresses.contains(next.nodes[position].address) {
                next.nodes[position].x += offset.x
                next.nodes[position].y += offset.y
            }
        case .delete(let removal):
            try requireNodes(removal.nodeAddresses)
            guard removal.edgeIDs.isSubset(of: Set(next.edges.map(\.id))) else {
                throw failure(.invalidSelection, "A selected workflow wire no longer exists.")
            }
            next.nodes.removeAll { removal.nodeAddresses.contains($0.address) }
            if let owners=next.modelInputs {
                let surviving=Set(next.nodes.map(\.address))
                let kept=owners.filter {(surviving.contains(.init(domain:.control,localID:$0.key)) || surviving.contains(.init(domain:.model,localID:$0.key))) && surviving.contains(.init(domain:.model,localID:$0.value))}
                next.modelInputs=kept.isEmpty ? nil : kept
            }
            if let models=next.namedModelInputs {
                let kept=models.filter {id,_ in next.nodes.contains {$0.address.localID==id && WorkflowModelBoundary.isDefinition($0)}}.mapValues {names in names.filter {_,input in next.nodes.contains {$0.address == .init(domain:.model,localID:input)}}}.filter {!$0.value.isEmpty}
                next.namedModelInputs=kept.isEmpty ? nil:kept
            }
            if let versions=next.modelSourceVersions {next.modelSourceVersions=versions.filter {id,_ in next.nodes.contains {$0.address.localID==id && WorkflowModelBoundary.isDefinition($0)}}}
            next.edges.removeAll { removal.edgeIDs.contains($0.id) || removal.nodeAddresses.contains($0.from.address) || removal.nodeAddresses.contains($0.to.address) }
        case .duplicate(let requested, let offset):
            guard !document.nodes.contains(where:{requested.contains($0.address) && RetiredTrainingRecipeMigration.isRetired($0)}) else {throw failure(.configuration,"Upgrade retired training settings before duplicating them.")}
            var addresses=requested
            for definition in document.nodes where requested.contains(definition.address) {
                guard WorkflowModelBoundary.isDefinition(definition),let boundary=try? WorkflowModelInput.boundary(for:definition.address,in:document),WorkflowModelInput.isBoundary(boundary),
                      (WorkflowModelInput.destinations(of:boundary,in:document).isEmpty || WorkflowModelInput.destinations(of:boundary,in:document).contains(where:{requested.contains($0.address)})) else {continue}
                addresses.insert(boundary.address)
            }
            for definition in document.nodes where requested.contains(definition.address) && WorkflowModelBoundary.isDefinition(definition) {
                for id in (document.namedModelInputs?[definition.address.localID] ?? [:]).values {
                    let boundary=WorkflowNodeAddress(domain:.model,localID:id)
                    let destinations=document.edges.filter {$0.from.address==boundary}.map { $0.to.address }
                    if destinations.isEmpty || destinations.contains(where:requested.contains) {addresses.insert(boundary)}
                }
            }
            try requireNodes(addresses)
            try requireOffset(offset)
            for original in document.nodes where addresses.contains(original.address) {
                let address = WorkflowNodeAddress(domain: original.address.domain, localID: try freshID(maximum: 128))
                var copy = original
                copy.address = address
                copy.x += offset.x
                copy.y += offset.y
                next.nodes.append(copy)
                copies.nodes[original.address] = address
            }
            for original in document.edges {
                guard let from = copies.nodes[original.from.address], let to = copies.nodes[original.to.address] else { continue }
                let id = try freshID(maximum: 1024)
                let copy = try edge(from: .init(address: from, port: original.from.port), to: .init(address: to, port: original.to.port), id: id, in: next,allowLegacySinkOutput:true)
                guard copy.kind == original.kind else { throw failure(.portKind, "The copied wire's recorded kind does not match its declared output.", original.from.address, original.from.port) }
                next.edges.append(copy)
                copies.edges[original.id] = id
            }
            if var owners=next.modelInputs {
                for (original,copy) in copies.nodes where document.nodes.first(where:{$0.address==original}).map(WorkflowModelBoundary.isDefinition)==true {
                    if let input=owners[original.localID] {owners[copy.localID]=copies.nodes[.init(domain:.model,localID:input)]?.localID ?? input}
                }
                next.modelInputs=owners
            }
            if var models=next.namedModelInputs {
                for (original,copy) in copies.nodes where document.nodes.first(where:{$0.address==original}).map(WorkflowModelBoundary.isDefinition)==true {
                    if let names=models[original.localID] {models[copy.localID]=names.mapValues {copies.nodes[.init(domain:.model,localID:$0)]?.localID ?? $0}}
                }
                next.namedModelInputs=models
            }
            if var versions=next.modelSourceVersions {
                for (original,copy) in copies.nodes where document.nodes.first(where:{$0.address==original}).map(WorkflowModelBoundary.isDefinition)==true {
                    if let version=versions[original.localID] {versions[copy.localID]=version}
                }
                next.modelSourceVersions=versions
            }
            if !addresses.isEmpty { selected = .init(nodeAddresses: Set(copies.nodes.values), edgeIDs: Set(copies.edges.values)) }
        case .connect(let from, let to):
            if from.port=="model",let provider=next.nodes.first(where:{$0.address==from.address}),case .train=provider.payload {next=try WorkflowTrainedModelSource.promoteAuthoring(next)}
            next.edges.append(try edge(from: from, to: to, id: try freshID(maximum: 1024), in: next))
            selected = .init(edgeIDs: [next.edges.last!.id])
        case .reconnect(let id, let from, let to):
            if from.port=="model",let provider=next.nodes.first(where:{$0.address==from.address}),case .train=provider.payload {next=try WorkflowTrainedModelSource.promoteAuthoring(next)}
            guard let position = next.edges.firstIndex(where: { $0.id == id }) else {
                throw failure(.invalidSelection, "The wire selected for reconnection no longer exists.")
            }
            var withoutOriginal = next
            withoutOriginal.edges.remove(at: position)
            let checked = try edge(from: from, to: to, id: id, in: withoutOriginal,allowLegacySinkOutput:from == next.edges[position].from)
            next=withoutOriginal;next.edges.insert(checked,at:min(position,next.edges.count))
            selected = .init(edgeIDs: [id])
        }
        if next != document {
            guard document.drafts.isEmpty else { throw failure(.unresolvedDraft, "Apply or discard the pending Block draft before changing its parent workflow.") }
            guard document.revision < UInt64.max else { throw failure(.invalidDocument, "Workflow revision limit reached.") }
            next.revision += 1
            _ = try WorkflowCodec.encode(next)
        }
        selected.nodeAddresses.formIntersection(Set(next.nodes.map(\.address)))
        selected.edgeIDs.formIntersection(Set(next.edges.map(\.id)))
        return .init(document: next, selection: selected, copies: copies)
    }

    private static func failure(_ code: WorkflowCompilerError.Code, _ message: String, _ node: WorkflowNodeAddress? = nil, _ port: String? = nil) -> WorkflowCompilerError {
        .init(code: code, message: message, node: node, port: port)
    }
}
