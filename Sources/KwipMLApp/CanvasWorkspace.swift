import SwiftUI
import KwipMLCore

struct CanvasRevealRequest:Equatable {
    let id=UUID()
    let nodeIDs:Set<String>
    var fitsBounds = false
}

/// Shared editor surface: graph geometry is distinct from each domain's typed configuration and contracts.
@MainActor protocol CanvasWorkspace: ObservableObject, EditorHistory {
    var graph: GraphDocument { get }
    var canvasViewState: CanvasViewState { get set }
    var canvasViewRevision: UUID { get }
    var canvasRevealRequest:CanvasRevealRequest? {get}
    var registry: ComponentRegistry { get }
    /// Creation catalog can differ from per-node projected types used by the renderer.
    var canvasCatalog: ComponentRegistry { get }
    func canvasIconType(nodeID: String) -> String
    func canvasInsertionTypes(edgeID: String) -> [String]
    var validation: ValidationResult { get }
    var isBusy: Bool { get }
    var allowsInspection: Bool { get }
    var selectedNodeIDs: Set<String> { get }
    var selectedEdgeID: String? { get }
    var pendingSourceID: String? { get set }

    var statusMessage: String { get set }
    var previewWeightLabel: String { get }
    func canvasEvidenceLabel(nodeID: String) -> String
    func previewStatus(for id: String) -> String
    func canvasSummary(_ node: GraphNode) -> String
    func canvasConfiguration(nodeID:String)->[String:JSONValue]
    func canvasPortDescriptor(nodeID:String,port:String,direction:PortDirection) -> CanvasPortDescriptor
    func canvasPortDescription(nodeID: String, port: String) -> String
    func canConnect(source: String, sourcePort: String, target: GraphNode, port: String) -> Bool
    func selectNode(_ id: String, extending: Bool)
    func selectEdge(_ id: String)
    func clearSelection()
    func moveNodes(ids: Set<String>, dx: Double, dy: Double)
    var inlineContainers:[CanvasInlineContainer] {get}
    func toggleInlineContainer(_ id:String)
    /// Preview a container and its visible descendants together; the adapter owns canonical movement.
    func canvasMovingNodeIDs(for ids:Set<String>)->Set<String>
    func duplicateSelection()
    func deleteSelection()
    func canAddType(_ type: String) -> Bool
    func canOpenBlock(_ id: String) -> Bool
    func openBlock(_ id: String)
    func addNode(type: String, x: Double, y: Double)
    func connect(source: String, sourcePort: String, target: String, targetPort: String)
    func reconnect(edgeID: String, source: String, sourcePort: String, target: String, targetPort: String)
    func insertNode(type: String, edgeID: String, x: Double, y: Double)
}

extension CanvasWorkspace {
    var allowsInspection: Bool { !isBusy }
    var inlineContainers:[CanvasInlineContainer] {[]}
    func toggleInlineContainer(_ id:String) {}
    func canvasMovingNodeIDs(for ids:Set<String>)->Set<String> {ids}
    func canvasEvidenceLabel(nodeID: String) -> String { previewWeightLabel }
    func paletteInsertionPoint(type:String,start:CGPoint)->CGPoint {
        CanvasAutoPlacement.origin(type:type,graph:graph,registry:registry,catalog:canvasCatalog,presentation:canvasViewState.presentation,start:start)
    }

    var canvasRevealRequest:CanvasRevealRequest? {nil}
    var canvasCatalog: ComponentRegistry { registry }
    func canvasIconType(nodeID: String) -> String { graph.nodes.first { $0.id == nodeID }?.type ?? "unknown" }
    func canvasInsertionTypes(edgeID: String) -> [String] {
        canvasCatalog.components.filter { $0.inputs.count == 1 && $0.outputs.count == 1 }.map(\.id)
    }
    /// One filtered catalog powers both insertion chooser and drag proposals. An adapter
    /// without atomic wire insertion returns no types, without disabling ordinary drops.
    func canvasInsertionComponents(edgeID: String) -> [ComponentDefinition] {
        let eligible = Set(canvasInsertionTypes(edgeID: edgeID))
        return TensorOperationPalette.consolidated(components:ReshapePalette.consolidated(components:RecurrentCellPalette.consolidated(components:RecurrentPalette.consolidated(components:LinearPalette.consolidated(components:NormalizationPalette.consolidated(components:ConvolutionPalette.consolidated(components:PoolingPalette.consolidated(components:ActivationPalette.consolidated(components:canvasCatalog.components.filter { eligible.contains($0.id) })))))))))
    }
    func canAddType(_ type: String) -> Bool { registry.component(type: type) != nil }
    func canOpenBlock(_ id: String) -> Bool { false }
    func openBlock(_ id: String) {}
    func canvasConfiguration(nodeID:String)->[String:JSONValue] {graph.nodes.first(where:{$0.id == nodeID})?.config ?? [:]}
    func canvasPortDescriptor(nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor { .unknown(direction:direction) }
    func selectNode(_ id: String) { selectNode(id, extending: false) }
}

extension WorkspaceStore: CanvasWorkspace {
    func canAddType(_ type: String) -> Bool { (type == ModelBlockPalette.type && !isLearningLessonActive && tutorialStep == nil) || registry.component(type: type) != nil }
    var canvasViewState: CanvasViewState { get { modelCanvasViewState } set { modelCanvasViewState = newValue } }
    func canvasSummary(_ item: GraphNode) -> String {
        if item.type == "softmax" { return "Softmax · axis \(item.config["dim"]?.integerValue ?? -1)" }
        if item.type == "gelu" { return item.config["approximate_tanh"]?.boolValue == true ? "GELU · tanh approximation" : "GELU · exact" }
        if item.type == "linear" { return "\(CanvasDisplayValue.dimension(item.config["in_features"]?.doubleValue)) → \(CanvasDisplayValue.dimension(item.config["out_features"]?.doubleValue)) features" + (item.config["bias"]?.boolValue == false ? " · no bias" : "") }
        if let shape = validation.tensorLayouts[item.id] { return shape.displayShape }
        return "Shape unresolved"
    }
    func canvasPortDescription(nodeID: String, port: String) -> String {
        let direction:PortDirection = registry.component(type:graph.nodes.first {$0.id == nodeID}?.type ?? "")?.inputs.contains(port) == true ? .input:.output
        return canvasPortDescriptor(nodeID:nodeID,port:port,direction:direction).accessibilityDescription
    }
    func connect(source: String, sourcePort: String, target: String, targetPort: String) {
        applyCommand(.connect(source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target, targetPort)))
    }
    func canConnect(source: String, sourcePort: String, target: GraphNode, port: String) -> Bool {
        (try? GraphCommandEngine.apply(.connect(source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target.id, port)), to: GraphEditorState(document: graph), registry: registry)) != nil
    }
}
