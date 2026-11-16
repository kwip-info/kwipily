import Foundation

/// A node-local visual type prevents unresolved wire ports from contaminating sibling contracts.
/// This is presentation metadata, not an executable operator identity.
public struct WorkflowCanvasType: Codable, Equatable {
    public let address: WorkflowNodeAddress
    public let kind: String
    public let identifier: String
    public let configurationVersion: Int?
    public let reference: CompositeDefinitionReference?
}

/// Canonical, reversible display identities. These identifiers must never be serialized to workers.
public enum WorkflowCanvasIDs {
    public static func nodeID(_ address: WorkflowNodeAddress) throws -> String { try encode(address, prefix: "wf-n-") }
    public static func address(from displayID: String) throws -> WorkflowNodeAddress {
        let value: WorkflowNodeAddress = try decode(displayID, prefix: "wf-n-")
        guard !value.localID.isEmpty, value.localID.unicodeScalars.count <= 128 else { throw invalidID() }
        return value
    }
    public static func displayEdgeID(_ sourceID: String) throws -> String { try encode(sourceID, prefix: "wf-e-") }
    public static func edgeID(from displayID: String) throws -> String {
        let value: String = try decode(displayID, prefix: "wf-e-")
        guard !value.isEmpty, value.unicodeScalars.count <= 1024 else { throw invalidID() }
        return value
    }
    public static func displayTypeID(_ type: WorkflowCanvasType) throws -> String { try encode(type, prefix: "wf-t-") }
    public static func type(from displayID: String) throws -> WorkflowCanvasType { try decode(displayID, prefix: "wf-t-") }
    private static func encode<T: Encodable>(_ value: T, prefix: String) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return prefix + (try encoder.encode(value)).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-").replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
    }
    private static func decode<T: Codable>(_ value: String, prefix: String) throws -> T {
        // A valid 1,024-scalar edge can require six JSON bytes per escaped control scalar,
        // then base64 expansion. Keep tokens bounded without excluding codec-valid identities.
        guard value.hasPrefix(prefix), value.utf8.count <= 16384 else { throw invalidID() }
        let suffix = String(value.dropFirst(prefix.count))
        guard !suffix.isEmpty, suffix.utf8.allSatisfy({ (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || $0 == 45 || $0 == 95 }) else { throw invalidID() }
        var encoded = suffix.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        encoded += String(repeating: "=", count: (4 - encoded.count % 4) % 4)
        guard let bytes = Data(base64Encoded: encoded) else { throw invalidID() }
        let decoded = try JSONDecoder().decode(T.self, from: bytes)
        guard try encode(decoded, prefix: prefix) == value else { throw invalidID() }
        return decoded
    }
    private static func invalidID() -> DataFlowContractError { .init("Invalid or noncanonical workflow display identity.") }
}

public struct WorkflowCanvasPortDescriptor: Equatable {
    public let endpoint: WorkflowEndpoint
    public let direction: PortDirection
    public let contract: WorkflowResolvedPort?
    public let unresolvedReason: String?
    public var semanticKind: WorkflowEdgeKind? { contract?.kind }
    public var label: String {
        switch semanticKind {
        case .table: return "Table"
        case .dataset: return "Dataset"
        case .fittedState: return "Fitted state"
        case .tensor: return "Tensor template"
        case .optimizerRecipe: return "Optimizer recipe"
        case .schedulerRecipe: return "Scheduler recipe"
        case .objectiveRecipe: return "Objective recipe"
        case .initializationRecipe: return "Starting settings recipe"
        case .gradientPolicyRecipe: return "Gradient policy recipe"
        case .datasetRecipe: return "Dataset view recipe"
        case .batchRecipe: return "Batch reader recipe"
        case .trainabilityRecipe: return "Trainability recipe"
        case .modelDefinition: return "Model definition"
        case .trainingData: return "Training data mapping"
        case .runConfiguration: return "Run configuration"
        case .frozenModel: return "Trained model"
        case nil: return "Unresolved port"
        }
    }
    public var accessibilityDescription: String {
        let prefix = "\(direction == .input ? "Input" : "Output") · \(endpoint.port) · \(label)"
        if let unresolvedReason { return prefix + " · " + unresolvedReason }
        return prefix + " · Declared contract; not measured execution data"
    }
}

public struct WorkflowCanvasIssue: Equatable {
    public let address: WorkflowNodeAddress
    public let port: String?
    public let message: String
}

/// Immutable render lookup. Direction and qualified domain identity are independent keys.
/// Preserve first-descriptor semantics even for a malformed duplicate supplied by a caller.
struct WorkflowCanvasPortIndex {
    private struct Key: Hashable {
        let endpoint: WorkflowEndpoint
        let direction: PortDirection
    }
    private let descriptors: [Key: WorkflowCanvasPortDescriptor]
    init(_ ports: [WorkflowCanvasPortDescriptor]) {
        descriptors = Dictionary(ports.map { (Key(endpoint: $0.endpoint, direction: $0.direction), $0) },
                                 uniquingKeysWith: { first, _ in first })
    }
    func descriptor(endpoint: WorkflowEndpoint, direction: PortDirection) -> WorkflowCanvasPortDescriptor? {
        descriptors[Key(endpoint: endpoint, direction: direction)]
    }
}

public struct WorkflowCanvasProjection {
    private var sourceDocument: WorkflowDocument
    public private(set) var graph: GraphDocument
    public let registry: ComponentRegistry
    public let addressByDisplayID: [String: WorkflowNodeAddress]
    public let displayIDByAddress: [WorkflowNodeAddress: String]
    public let edgeIDByDisplayID: [String: String]
    public let displayIDByEdgeID: [String: String]
    public let typeByDisplayID: [String: WorkflowCanvasType]
    public let ports: [WorkflowCanvasPortDescriptor]
    public let issues: [WorkflowCanvasIssue]
    private let portIndex: WorkflowCanvasPortIndex

    init(graph: GraphDocument, registry: ComponentRegistry,
         addressByDisplayID: [String: WorkflowNodeAddress], displayIDByAddress: [WorkflowNodeAddress: String],
         edgeIDByDisplayID: [String: String], displayIDByEdgeID: [String: String],
         typeByDisplayID: [String: WorkflowCanvasType], ports: [WorkflowCanvasPortDescriptor],
         issues: [WorkflowCanvasIssue], sourceDocument: WorkflowDocument) {
        self.sourceDocument = sourceDocument
        self.graph = graph
        self.registry = registry
        self.addressByDisplayID = addressByDisplayID
        self.displayIDByAddress = displayIDByAddress
        self.edgeIDByDisplayID = edgeIDByDisplayID
        self.displayIDByEdgeID = displayIDByEdgeID
        self.typeByDisplayID = typeByDisplayID
        self.ports = ports
        self.issues = issues
        self.portIndex = WorkflowCanvasPortIndex(ports)
    }

    /// A conservative presentation-only reuse path. The source retained by this
    /// projection owns all contract identity; callers cannot supply an unrelated key.
    /// Semantic changes require the full projector and current inferred definitions.
    public func updatingPresentation(_ document:WorkflowDocument) throws -> WorkflowCanvasProjection? {
        guard document.nodes.count == sourceDocument.nodes.count else {return nil}
        var semantic=document
        semantic.name=sourceDocument.name;semantic.revision=sourceDocument.revision
        for index in semantic.nodes.indices {
            semantic.nodes[index].name=sourceDocument.nodes[index].name
            semantic.nodes[index].x=sourceDocument.nodes[index].x
            semantic.nodes[index].y=sourceDocument.nodes[index].y
        }
        guard semantic == sourceDocument else {return nil}
        // Retain the same validation and encoded-size boundary as a full projection.
        // The optimization skips contract resolution and ID/index reconstruction only.
        _ = try WorkflowCodec.encode(document)
        var updated=self
        updated.sourceDocument=document
        updated.graph.name=document.name
        let sourceByAddress=Dictionary(uniqueKeysWithValues:document.nodes.map {($0.address,$0)})
        for index in updated.graph.nodes.indices {
            guard let address=addressByDisplayID[updated.graph.nodes[index].id],let node=sourceByAddress[address] else {return nil}
            updated.graph.nodes[index].name=node.name
            updated.graph.nodes[index].x=node.x
            updated.graph.nodes[index].y=node.y
        }
        return updated
    }

    public func portDescriptor(address: WorkflowNodeAddress, port: String, direction: PortDirection) -> WorkflowCanvasPortDescriptor? {
        portIndex.descriptor(endpoint: .init(address: address, port: port), direction: direction)
    }
}

/// Projects all domains on the same canvas. No compilation, defaults injection, migration, source
/// mutation or compatibility inference occurs here. Optional routing declarations come from the
/// editor’s revision-bound model inference cache. Unknown contracts remain inspectable drafts.
public enum WorkflowCanvasProjector {
    public static func project(_ document: WorkflowDocument, modelRegistry: ComponentRegistry,
                               dataRegistry: DataFlowOperatorRegistry, configuredModelDefinitions: [WorkflowNodeAddress: ComponentDefinition] = [:], inlineTrainControls: Bool = false) throws -> WorkflowCanvasProjection {
        _ = try WorkflowCodec.encode(document)
        var nodes: [GraphNode] = [], components: [ComponentDefinition] = [], ports: [WorkflowCanvasPortDescriptor] = []
        var addresses: [String: WorkflowNodeAddress] = [:], nodeIDs: [WorkflowNodeAddress: String] = [:]
        var types: [String: WorkflowCanvasType] = [:], issues: [WorkflowCanvasIssue] = []
        for node in document.nodes {
            let metadata = metadata(node, document: document, modelRegistry: modelRegistry, dataRegistry: dataRegistry)
            let id = try WorkflowCanvasIDs.nodeID(node.address), type = try WorkflowCanvasIDs.displayTypeID(metadata.type)
            addresses[id] = node.address; nodeIDs[node.address] = id; types[type] = metadata.type
            let configured: ComponentDefinition?
            if case .modelPrimitive(let type, _, _) = node.payload,
               TensorRoutingConfiguration.componentIDs.contains(type),
               let candidate = configuredModelDefinitions[node.address], candidate.id == type {
                configured = candidate
            } else { configured = nil }
            var resolved: WorkflowNodePorts?
            var failure: String?
            do {
                if let configured {
                    resolved = .init(inputs: configured.inputs.map { .init(id: $0, kind: .tensor, required: true, maximumConnections: 1) },
                                     outputs: configured.outputs.map { .init(id: $0, kind: .tensor, required: false, maximumConnections: nil) })
                } else {
                    if inlineTrainControls,case .train=node.payload {
                        resolved = try WorkflowPortContracts.resolveForAuthoring(node,document:document,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
                    } else {resolved = try WorkflowPortContracts.resolve(node, document: document, modelRegistry: modelRegistry, dataRegistry: dataRegistry)}
                }
            }
            catch { failure = error.localizedDescription; issues.append(.init(address: node.address, port: nil, message: failure!)) }
            if case .dataOperator(let operatorID, _) = node.payload,
               dataRegistry.operatorDefinition(id: operatorID)?.isCanvasSink == true, let contract = resolved {
                // Retain only already-saved outbound endpoints; a new View is a terminal.
                let retained = Set(document.edges.filter { $0.from.address == node.address }.map { $0.from.port })
                resolved = .init(inputs: contract.inputs, outputs: contract.outputs.filter { retained.contains($0.id) })
            }
            // Current authoring owns data-access settings inside Train. Historical
            // projections retain the legacy contracts when explicitly requested.
            if inlineTrainControls, case .train = node.payload, let contract = resolved {
                resolved = .init(inputs: contract.inputs.filter { !["optimizer", "scheduler", "objective", "initialization", "gradientPolicy", "trainability", "dataset", "loader"].contains($0.id) }, outputs: contract.outputs)
            }
            var inputIDs = resolved?.inputs.map(\.id) ?? [], outputIDs = resolved?.outputs.map(\.id) ?? []
            for (direction, values) in [(PortDirection.input, resolved?.inputs ?? []), (.output, resolved?.outputs ?? [])] {
                ports += values.map { .init(endpoint: .init(address: node.address, port: $0.id), direction: direction, contract: $0, unresolvedReason: nil) }
            }
            // Preserve visible wire endpoints even when their saved port is unavailable. These
            // fallback ports carry no kind, cardinality or implied acceptance from the edge claim.
            for direction in [PortDirection.input, .output] {
                let referenced = Set(document.edges.compactMap { edge -> String? in
                    let endpoint = direction == .input ? edge.to : edge.from
                    return endpoint.address == node.address ? endpoint.port : nil
                }).sorted()
                for port in referenced where !(direction == .input ? inputIDs : outputIDs).contains(port) {
                    let reason = failure ?? "This saved connection names a port not declared by the current component contract."
                    ports.append(.init(endpoint: .init(address: node.address, port: port), direction: direction, contract: nil, unresolvedReason: reason))
                    issues.append(.init(address: node.address, port: port, message: reason))
                    // Routing repair wires remain visible through saved graph endpoints and
                    // nil-contract descriptors, but must not become live connection choices.
                    if configured == nil {
                        if direction == .input { inputIDs.append(port) } else { outputIDs.append(port) }
                    }
                }
            }
            let displayName:String
            switch node.payload {case .frozenModel,.generateText:displayName=node.name.isEmpty ? metadata.title:node.name;default:displayName=node.name}
            var displayed = GraphNode(id: id, type: type, name: displayName, x: node.x, y: node.y, config: metadata.config)
            displayed.config = metadata.config // Raw authoring values, even for legacy Linear without bias.
            nodes.append(displayed)
            components.append(.init(id: type, title: metadata.title, category: metadata.group,
                                    inputs: inputIDs, outputs: outputIDs, fields: metadata.fields))
        }
        var edgeIDs: [String: String] = [:], displayEdgeIDs: [String: String] = [:]
        let edges = try document.edges.map { edge -> GraphEdge in
            let id = try WorkflowCanvasIDs.displayEdgeID(edge.id)
            edgeIDs[id] = edge.id; displayEdgeIDs[edge.id] = id
            return GraphEdge(id: id, source: try WorkflowCanvasIDs.nodeID(edge.from.address), sourcePort: edge.from.port,
                             target: try WorkflowCanvasIDs.nodeID(edge.to.address), targetPort: edge.to.port)
        }
        return .init(graph: .init(name: document.name, nodes: nodes, edges: edges), registry: .init(components: components),
                     addressByDisplayID: addresses, displayIDByAddress: nodeIDs, edgeIDByDisplayID: edgeIDs,
                     displayIDByEdgeID: displayEdgeIDs, typeByDisplayID: types, ports: ports, issues: issues, sourceDocument: document)
    }

    private struct Metadata {
        let type: WorkflowCanvasType
        let title: String
        let group: String
        let config: [String: JSONValue]
        let fields: [ComponentField]
    }
    private static func metadata(_ node: WorkflowNode, document: WorkflowDocument, modelRegistry: ComponentRegistry,
                                 dataRegistry: DataFlowOperatorRegistry) -> Metadata {
        func make(_ kind: String, _ id: String, _ title: String, _ group: String, _ config: [String: JSONValue],
                  version: Int? = nil, reference: CompositeDefinitionReference? = nil, fields: [ComponentField] = []) -> Metadata {
            .init(type: .init(address: node.address, kind: kind, identifier: id, configurationVersion: version, reference: reference),
                  title: title, group: group, config: config, fields: fields)
        }
        switch node.payload {
        case .dataOperator(let id, let config):
            return make("dataOperator", id, dataRegistry.operatorDefinition(id: id)?.title ?? "Unavailable data operator · \(id)", "Data", config)
        case .modelPrimitive(let type, let version, let config):
            let component = modelRegistry.component(type: type)
            return make("modelPrimitive", type, WorkflowModelInput.componentIDs.contains(type) ? "Model input":type=="output" ? "Model definition":component?.title ?? "Unavailable model operator · \(type)", WorkflowModelInput.componentIDs.contains(type) ? "Boundaries":component?.category ?? "Model", config, version: version, fields: component?.fields ?? [])
        case .modelBlock(let reference, let overrides):
            let definition = document.definitions.first { $0.reference == reference }
            return make("modelBlock", reference.id, definition?.title ?? "Unavailable Block · \(reference.id)", "Blocks", overrides, reference: reference, fields: definition?.parameters.map(\.field) ?? [])
        case .optimizerRecipe(let type,let version,let config):
            let component = modelRegistry.component(type:type)
            return make("optimizerRecipe",type,"Retired · " + (component?.title ?? type),"Repair",config,version:version,fields:[])
        case .lossRecipe(let type,let version,let config):
            let component = modelRegistry.component(type:type)
            return make("lossRecipe",type,"Retired · " + (component?.title ?? type),"Repair",config,version:version,fields:[])
        case .initializationRecipe(let type,let version,let config):
            let component = modelRegistry.component(type:type)
            return make("initializationRecipe",type,"Retired · " + (component?.title ?? "Starting settings"),"Repair",config,version:version,fields:[])
        case .gradientPolicyRecipe(let type,let version,let config):
            let component = modelRegistry.component(type:type)
            return make("gradientPolicyRecipe",type,"Retired · " + (component?.title ?? "Gradient policy"),"Repair",config,version:version,fields:[])
        case .trainabilityRecipe(let type,let version,let config):
            let component = modelRegistry.component(type:type)
            return make("trainabilityRecipe",type,"Retired · " + (component?.title ?? "Trainability"),"Repair",config,version:version,fields:[])
        case .schedulerRecipe(let type,let version,let config):
            let component = modelRegistry.component(type:type)
            return make("schedulerRecipe",type,"Retired · " + (component?.title ?? type),"Repair",config,version:version,fields:[])
        case .datasetRecipe(let type,let version,let config):
            let component=modelRegistry.component(type:type)
            return make("datasetRecipe",type,"Retired · " + (component?.title ?? "Dataset view"),"Repair",config,version:version,fields:[])
        case .batchRecipe(let type,let version,let config):
            let component=modelRegistry.component(type:type)
            return make("batchRecipe",type,"Retired · " + (component?.title ?? "Batch reader"),"Repair",config,version:version,fields:[])
        case .modelDefinition: return make("modelDefinition", "modelDefinition", "Model definition", "Training", [:])
        case .trainingData(let config): return make("trainingData", "trainingData", "Training data mapping", "Training", config)
        case .runSetup(let config): return make("runSetup", "runSetup", "Run setup", "Training", config)
        case .estimator(let config): return make("estimator", "estimator", "Prebuilt models", "Training", config)
        case .train(let config): return make("train", "train", "Train", "Training", config)
        case .frozenModel(let config): return make("frozenModel", "frozenModel", "Load model", "Boundaries", config)
        case .gameSession(let config): return make("gameSession", "gameSession", "Game session", "Evaluation", config)
        case .generateText(let config): return make("generateText", "generateText", "Generate text", "Evaluation", config)
        case .saveModel(let config): return make("saveModel", "saveModel", "Save model", "Evaluation", config)
        case .evaluateTest(let config): return make("evaluateTest", "evaluateTest", "Evaluate", "Evaluation", config)
        }
    }
}
