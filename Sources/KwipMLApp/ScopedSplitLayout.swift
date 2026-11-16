import AppKit
import SwiftUI
import KwipMLCore

extension View {
    /// Attach inside a direct pane of the intended split, never to a containing workspace root.
    func workspaceSplitLayout(store: WorkspaceStore, scope: WorkspacePanelScope?) -> some View {
        background(ScopedSplitLayout(store: store, scope: scope).allowsHitTesting(false))
    }
}
private struct ScopedSplitLayout: NSViewRepresentable {
    @ObservedObject var store: WorkspaceStore
    let scope: WorkspacePanelScope?
    func makeNSView(context: Context) -> ScopedSplitProbe { ScopedSplitProbe() }
    func updateNSView(_ view: ScopedSplitProbe, context: Context) {
        view.configure(store: store, scope: scope)
    }
    static func dismantleNSView(_ view: ScopedSplitProbe, coordinator: ()) { view.invalidate() }
}

/// No autosave name or global NSSplitView delegate replacement. A probe owns only its nearest split.
private final class ScopedSplitProbe: NSView {
    private weak var store: WorkspaceStore?
    private weak var split: NSSplitView?
    private var scope: WorkspacePanelScope?
    private var projectID: String?
    private var revision: UUID?
    private var notification: NSObjectProtocol?
    private var monitor: Any?
    private var invalidated = false
    private var scheduled = false
    private var applying = false
    private var dragging = false
    private var changedByDrag = false
    private var dragSize = NSSize.zero
    private var lastAvailable: Double?
    private var lastPreference: PanelLayoutPreference?
    private var lastAppliedSizes: [Double]?
    private var contextGeneration = UUID()

    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); if window == nil { detach() } else { schedule() } }
    override func viewDidMoveToSuperview() { super.viewDidMoveToSuperview(); schedule() }
    override func layout() { super.layout(); schedule() }
    func configure(store: WorkspaceStore, scope: WorkspacePanelScope?) {
        if self.store !== store || self.scope != scope || projectID != store.activeProjectID || revision != store.canvasViewRevision {
            detach()
            self.store = store; self.scope = scope; projectID = store.activeProjectID; revision = store.canvasViewRevision
        }
        schedule()
    }
    func invalidate() { invalidated = true; detach() }
    func detach() {
        if let notification { NotificationCenter.default.removeObserver(notification) }
        if let monitor { NSEvent.removeMonitor(monitor) }
        notification = nil; monitor = nil; split = nil
        dragging = false; changedByDrag = false; lastAvailable = nil; lastPreference = nil; lastAppliedSizes = nil
        contextGeneration = UUID()
    }
    private func schedule() {
        guard !scheduled else { return }; scheduled = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }; self.scheduled = false; self.attachAndRestore()
        }
    }
    private func attachAndRestore() {
        guard !invalidated, let store, let scope, let projectID, window != nil,
              store.activeProjectID == projectID, revision == store.canvasViewRevision else { return }
        var parent = superview
        while let candidate = parent, !(candidate is NSSplitView) { parent = candidate.superview }
        guard let nearest = parent as? NSSplitView, nearest.isVertical == scope.vertical,
              nearest.arrangedSubviews.count == scope.panes.count else { return }
        if split !== nearest {
            detach(); split = nearest
            notification = NotificationCenter.default.addObserver(forName: NSSplitView.didResizeSubviewsNotification, object: nearest, queue: .main) { [weak self] _ in
                guard let self, !self.applying else { return }
                if self.dragging, self.split?.bounds.size == self.dragSize { self.changedByDrag = true }
                self.schedule()
            }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .leftMouseUp]) { [weak self] event in
                self?.handle(event); return event
            }
        }
        guard !applying else { return }
        if dragging {
            if NSEvent.pressedMouseButtons & 1 == 0 { finishDrag() }
            return
        }
        let extent = nearest.isVertical ? nearest.bounds.width : nearest.bounds.height
        let available = Double(extent - nearest.dividerThickness * CGFloat(max(0, scope.panes.count - 1)))
        // Wait for a real layout; zero placeholder panes must not establish saved intent.
        guard available > 1, nearest.arrangedSubviews.allSatisfy({ $0.frame.width > 0 && $0.frame.height > 0 }) else { return }
        let preference = store.panelLayouts[scope.rawValue] ?? scope.initialPreference
        let actual = nearest.arrangedSubviews.map { Double(nearest.isVertical ? $0.frame.width : $0.frame.height) }
        let settled = lastAppliedSizes.map { previous in previous.count == actual.count && zip(previous, actual).allSatisfy { abs($0 - $1) < 0.5 } } ?? false
        guard lastAvailable != available || lastPreference != preference || !settled else { return }
        lastAvailable = available; lastPreference = preference
        guard let preference else { return } // Unconfigured layouts keep SwiftUI's native initial proportions.
        let sizes = WorkspacePanelSizing.sizes(scope: scope, available: available, preference: preference)
        guard sizes.count == nearest.arrangedSubviews.count else { return }
        applying = true
        var position: CGFloat = 0
        for index in 0..<max(0, sizes.count - 1) {
            position += CGFloat(sizes[index])
            nearest.setPosition(position, ofDividerAt: index)
            position += nearest.dividerThickness
        }
        lastAppliedSizes = nearest.arrangedSubviews.map { Double(nearest.isVertical ? $0.frame.width : $0.frame.height) }
        applying = false
    }
    private func handle(_ event: NSEvent) {
        guard let split, event.window === split.window, let scope, let store,
              store.activeProjectID == projectID, revision == store.canvasViewRevision else { return }
        if event.type == .leftMouseDown {
            let point = split.convert(event.locationInWindow, from: nil)
            guard split.bounds.contains(point), scope.panes.count > 1 else { return }
            let ranges = split.arrangedSubviews.map { pane -> ClosedRange<Double> in
                split.isVertical ? Double(pane.frame.minX)...Double(pane.frame.maxX) : Double(pane.frame.minY)...Double(pane.frame.maxY)
            }
            guard WorkspacePanelSizing.isDividerHit(coordinate: Double(split.isVertical ? point.x : point.y), paneRanges: ranges) else { return }
            dragging = true; changedByDrag = false; dragSize = split.bounds.size
        } else if dragging {
            let generation = contextGeneration
            DispatchQueue.main.async { [weak self] in
                guard let self, self.contextGeneration == generation else { return }; self.finishDrag()
            }
        }
    }
    private func finishDrag() {
        defer { dragging = false; changedByDrag = false }
        guard dragging, changedByDrag, let split, split.bounds.size == dragSize,
              let store, let scope, let projectID, let revision,
              store.activeProjectID == projectID, store.canvasViewRevision == revision else { return }
        let sizes = split.arrangedSubviews.map { Double(split.isVertical ? $0.frame.width : $0.frame.height) }
        store.recordPanelSizes(sizes, scope: scope, projectID: projectID, revision: revision)
        lastPreference = store.panelLayouts[scope.rawValue]
        lastAppliedSizes = sizes
    }
}
