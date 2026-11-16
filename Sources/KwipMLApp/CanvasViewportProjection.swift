import Foundation
import KwipMLCore

enum CanvasViewportProjection {
    static func restored(_ state: CanvasViewState, nodes: Set<String>, edges: Set<String>) -> CanvasViewState {
        var restored = state
        restored.zoom = state.zoom.isFinite ? max(CanvasZoomLimits.minimum,min(CanvasZoomLimits.maximum,state.zoom)) : 0.65
        restored.scrollX = state.scrollX.isFinite ? max(-1_000_000,min(1_000_000,state.scrollX)) : 0
        restored.scrollY = state.scrollY.isFinite ? max(-1_000_000,min(1_000_000,state.scrollY)) : 0
        restored.selectedNodeIDs = state.selectedNodeIDs.intersection(nodes)
        restored.selectedEdgeIDs = state.selectedEdgeIDs.intersection(edges)
        return restored
    }
}
