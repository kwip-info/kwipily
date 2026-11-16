import Foundation

/// One immutable geometry lookup for a registry/presentation pair. Construct once per render,
/// not once per endpoint. It never caches authored positions or mutable drag state.
public struct CanvasGeometrySnapshot {
    public private(set) var projection: CanvasCoordinateProjection
    private let definitions: [String: ComponentDefinition]
    private let sizes: [String: CGSize]
    private let fallbackSize: CGSize
    private var nodeDefinitions: [String: ComponentDefinition] = [:]
    private var nodeSizes: [String: CGSize] = [:]
    private var inputAnchorPorts: [String: [String]] = [:]
    private var outputAnchorPorts: [String: [String]] = [:]
    private var presentation: CanvasPresentationSettings = .init()
    private var customInputAnchors:[String:[String:CGPoint]] = [:]
    private var customOutputAnchors:[String:[String:CGPoint]] = [:]

    public init(registry: ComponentRegistry, presentation: CanvasPresentationSettings) {
        definitions = Dictionary(registry.components.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        sizes = definitions.mapValues {
            CanvasNodeGeometry.size(detail: presentation.nodeDetail, direction: presentation.flowDirection,
                                    inputCount: $0.inputs.count, outputCount: $0.outputs.count)
        }
        fallbackSize = CanvasNodeGeometry.size(detail: presentation.nodeDetail, direction: presentation.flowDirection,
                                               inputCount: 0, outputCount: 0)
        projection = CanvasCoordinateProjection(settings: presentation,
            maximumInputCount: registry.components.map { $0.inputs.count }.max() ?? 0,
            maximumOutputCount: registry.components.map { $0.outputs.count }.max() ?? 0)
    }
    /// Per-instance declarations and anchor geometry belong to one immutable render.
    /// Saved unavailable endpoints keep distinct repair slots; they are not connectable ports.
    public init(registry: ComponentRegistry, presentation: CanvasPresentationSettings, graph: GraphDocument, validation: ValidationResult,
                customSizes:[String:CGSize]=[:],customInputAnchors:[String:[String:CGPoint]]=[:],customOutputAnchors:[String:[String:CGPoint]]=[:]) {
        self.init(registry: registry, presentation: presentation)
        self.presentation = presentation
        let savedInputsByNode = Dictionary(grouping: graph.edges, by: \.target).mapValues { Set($0.map(\.targetPort)) }
        let savedOutputsByNode = Dictionary(grouping: graph.edges, by: \.source).mapValues { Set($0.map(\.sourcePort)) }
        for node in graph.nodes {
            guard let definition = validation.configuredDefinition(nodeID: node.id, graph: graph, registry: registry) else { continue }
            nodeDefinitions[node.id] = definition
            let savedInputs = savedInputsByNode[node.id] ?? []
            let savedOutputs = savedOutputsByNode[node.id] ?? []
            let inputs = definition.inputs + savedInputs.subtracting(definition.inputs).sorted()
            let outputs = definition.outputs + savedOutputs.subtracting(definition.outputs).sorted()
            inputAnchorPorts[node.id] = inputs
            outputAnchorPorts[node.id] = outputs
            nodeSizes[node.id] = CanvasNodeGeometry.size(detail: presentation.nodeDetail, direction: presentation.flowDirection,
                inputCount: inputs.count, outputCount: outputs.count)
        }
        for (id,size) in customSizes where nodeDefinitions[id] != nil && size.width.isFinite && size.height.isFinite && size.width>0 && size.height>=CanvasNodeGeometry.headerHeight {
            nodeSizes[id]=size
        }
        self.customInputAnchors=customInputAnchors
        self.customOutputAnchors=customOutputAnchors
        // A new bundle's eight-port ceiling must not reposition every existing graph.
        // Keep the legacy registry spacing, adding only routing instances actually present.
        let baseline = registry.components.filter { !TensorRoutingConfiguration.componentIDs.contains($0.id) && !RecurrentConfiguration.componentIDs.contains($0.id) && !AttentionConfiguration.componentIDs.contains($0.id) }
        let routing = graph.nodes.filter { TensorRoutingConfiguration.componentIDs.contains($0.type) || RecurrentConfiguration.componentIDs.contains($0.type) || AttentionConfiguration.componentIDs.contains($0.type) }.compactMap { nodeDefinitions[$0.id] }
        projection = CanvasCoordinateProjection(settings: presentation,
            maximumInputCount: (baseline + routing).map { $0.inputs.count }.max() ?? 0,
            maximumOutputCount: (baseline + routing).map { $0.outputs.count }.max() ?? 0)
    }
    public func definition(nodeID: String) -> ComponentDefinition? { nodeDefinitions[nodeID] }
    public func size(nodeID: String) -> CGSize { nodeSizes[nodeID] ?? fallbackSize }
    public func portAnchor(nodeID: String, output: Bool, port: String) -> CGPoint {
        if let point=(output ? customOutputAnchors[nodeID] : customInputAnchors[nodeID])?[port],point.x.isFinite,point.y.isFinite {return point}
        var ports = (output ? outputAnchorPorts[nodeID] : inputAnchorPorts[nodeID]) ?? []
        // Defensive callers also get a repair slot, never the first live endpoint.
        if !ports.contains(port) { ports.append(port) }
        return CanvasNodeGeometry.portAnchor(size: size(nodeID: nodeID), direction: presentation.flowDirection,
            output: output, index: ports.firstIndex(of: port)!, count: ports.count)
    }
    public func definition(type: String) -> ComponentDefinition? { definitions[type] }
    public func size(type: String) -> CGSize { sizes[type] ?? fallbackSize }
}
