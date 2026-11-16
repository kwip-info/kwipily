import SwiftUI
import KwipMLCore

/// Requested viewport intent is durable before native scrolling acknowledges its clamped origin.
@MainActor final class CanvasViewportPersistence: ObservableObject {
    private var captureToken = UUID()
    func beginCapture() -> UUID { captureToken=UUID();return captureToken }
    @discardableResult func request<W:CanvasWorkspace>(store:W,revision:UUID,zoom:Double,origin:CGPoint)->Bool {
        captureToken=UUID()
        return write(store:store,revision:revision,zoom:zoom,origin:origin)
    }
    @discardableResult func capture<W:CanvasWorkspace>(store:W,revision:UUID,token:UUID,zoom:Double,origin:CGPoint)->Bool {
        guard token == captureToken else {return false}
        return write(store:store,revision:revision,zoom:zoom,origin:origin)
    }
    private func write<W:CanvasWorkspace>(store:W,revision:UUID,zoom:Double,origin:CGPoint)->Bool {
        guard store.canvasViewRevision == revision,zoom.isFinite,origin.x.isFinite,origin.y.isFinite,
              (CanvasZoomLimits.minimum...CanvasZoomLimits.maximum).contains(zoom) else {return false}
        var state=store.canvasViewState
        state.zoom=zoom;state.scrollX=CanvasCameraGeometry.bounded(origin).x;state.scrollY=CanvasCameraGeometry.bounded(origin).y
        state.selectedNodeIDs=store.selectedNodeIDs;state.selectedEdgeIDs=store.selectedEdgeID.map {[$0]} ?? []
        if state != store.canvasViewState {store.canvasViewState=state}
        return true
    }
}
