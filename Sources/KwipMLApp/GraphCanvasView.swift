import SwiftUI
import Combine
import AppKit
import KwipMLCore
import UniformTypeIdentifiers

/// A nested canvas can borrow its stable workbench focus without publishing a competing scene value.
private struct CanvasHistoryFocusModifier: ViewModifier {
    let provider: EditorHistoryProvider
    let enabled: Bool
    @ViewBuilder func body(content: Content) -> some View {
        if enabled { content.focusedSceneObject(provider) } else { content }
    }
}

/// Shared, registry-driven renderer. Node names are always authored graph data.
struct GraphCanvasView<Store: CanvasWorkspace>: View {
    @ObservedObject var store: Store
    @Environment(\.canvasZoomSensitivity) private var zoomSensitivity
    @Environment(\.lessonNodeFocus) private var lessonNodeFocus
    var publishesHistoryFocus = true
    var showsPortControls = true
    @StateObject private var historyProvider = EditorHistoryProvider()
    @State private var movingID: String?
    @State private var dragProjection: CanvasCoordinateProjection?
    @State private var movingIDs = Set<String>()
    @State private var pendingInsertion: CanvasInsertion?
    @State private var connectionOptions = false
    @FocusState private var canvasFocused: Bool
    @State private var movement = CGSize.zero
    @State private var pendingOutputPort = "out"
    @State private var wirePointer: CGPoint?
    @State private var dropTargeted = false
    @State private var inlineDropTarget:String?
    @State private var fitNotice: String?
    @State private var toolbarInset: CGFloat = 60
    @State private var portOptions = false
    private var presentation: CanvasPresentationSettings { store.canvasViewState.presentation }
    private var direction: CanvasFlowDirection { .leftToRight }
    private var renderEvidence:WorkflowCanvasRenderEvidence?
    private func previewStatus(_ id:String)->String {renderEvidence?.statuses[id] ?? store.previewStatus(for:id)}
    private var renderSceneOrigin: CGPoint?
    private var renderGeometry: CanvasGeometrySnapshot?
    private var renderInlineContainers:[CanvasInlineContainer]?
    private var inlineContainers:[CanvasInlineContainer] {renderInlineContainers ?? store.inlineContainers}
    private func inlineContainer(_ id:String)->CanvasInlineContainer? {inlineContainers.first {$0.id==id}}
    private func makeGeometry(containers:[CanvasInlineContainer])->CanvasGeometrySnapshot {
        let expanded=containers.filter(\.isExpanded)
        return CanvasGeometrySnapshot(registry:store.registry,presentation:presentation,graph:store.graph,validation:store.validation,
            customSizes:Dictionary(expanded.map {($0.id,$0.size)},uniquingKeysWith:{first,_ in first}),
            customInputAnchors:Dictionary(expanded.map {($0.id,$0.inputOffsets)},uniquingKeysWith:{first,_ in first}),
            customOutputAnchors:Dictionary(expanded.map {($0.id,$0.outputOffsets)},uniquingKeysWith:{first,_ in first}))
    }
    private var geometry: CanvasGeometrySnapshot {
        renderGeometry ?? makeGeometry(containers:inlineContainers)
    }
    private var projection: CanvasCoordinateProjection { dragProjection ?? geometry.projection }
    private static func projection(registry: ComponentRegistry, presentation: CanvasPresentationSettings, graph: GraphDocument = .init(), validation: ValidationResult = .init()) -> CanvasCoordinateProjection {
        CanvasGeometrySnapshot(registry: registry, presentation: presentation, graph: graph, validation: validation).projection
    }
    private var sceneOrigin: CGPoint {
        renderSceneOrigin ?? CanvasWorldGeometry.sceneOrigin(nodes:store.graph.nodes.map {
            CGRect(origin:projection.project(CGPoint(x:$0.x,y:$0.y)),size:nodeSize($0))
        })
    }
    private var documentSize: CGSize {
        let base = direction == .leftToRight ? CGSize(width: 1600, height: 900) : CGSize(width: 900, height: 1600)
        return CGSize(width: max(base.width, store.graph.nodes.map { position($0).x + nodeSize($0).width + 40 }.max() ?? 0),
                      height: max(base.height, store.graph.nodes.map { position($0).y + nodeSize($0).height + 40 }.max() ?? 0))
    }
    private func nodeSize(_ item: GraphNode) -> CGSize { geometry.size(nodeID: item.id) }
    private func nodeHitBounds(_ item:GraphNode)->CGRect {
        var size=nodeSize(item)
        if inlineContainer(item.id)?.isExpanded == true {size.height=CanvasNodeGeometry.headerHeight}
        return CGRect(origin:position(item),size:size)
    }
    private func nodeHit(_ item:GraphNode,at point:CGPoint)->Bool {
        if nodeHitBounds(item).insetBy(dx:-14/zoom,dy:-14/zoom).contains(point) {return true}
        guard inlineContainer(item.id)?.isExpanded == true,let definition=geometry.definition(nodeID:item.id) else {return false}
        return [(false,definition.inputs),(true,definition.outputs)].contains {output,ports in
            ports.contains {port in
                let anchor=portPosition(item,output:output,port:port)
                return CGRect(x:anchor.x-14/zoom,y:anchor.y-14/zoom,width:28/zoom,height:28/zoom).contains(point)
            }
        }
    }
    private func updateInlineDropTarget(at point:CGPoint,moving ids:Set<String>) {
        inlineDropTarget=inlineContainers.last {container in
            guard container.isExpanded,!ids.contains(container.id),let item=node(container.id) else {return false}
            return CGRect(origin:position(item),size:nodeSize(item)).contains(point)
        }?.id
    }
    private static func size(type: String, registry: ComponentRegistry, presentation: CanvasPresentationSettings) -> CGSize {
        let definition = registry.component(type: type)
        return CanvasNodeGeometry.size(detail: presentation.nodeDetail, direction: presentation.flowDirection,
            inputCount: definition?.inputs.count ?? 0, outputCount: definition?.outputs.count ?? 0)
    }
    private func modelDelta(_ delta: CGSize) -> CGSize {
        let point = projection.inverse(CGPoint(x:delta.width,y:delta.height))
        return CGSize(width: point.x, height: point.y)
    }
    // GeometryReader already updates its subtree during native live resizing.
    // Publishing those same dimensions back into State rebuilt the entire graph
    // a second time. Input closures need the latest frame, not a new observation.
    private final class ViewportLayout {var size=CGSize.zero;var frame=CGRect.zero}
    @State private var viewportLayout=ViewportLayout()
    private var viewportSize:CGSize {get {viewportLayout.size} nonmutating set {viewportLayout.size=newValue}}
    private var zoom:Double {
        get {scrollController.translation.zoom}
        nonmutating set {scrollController.setZoom(newValue)}
    }
    @State private var hasFitted = false
    @State private var restoredRevision: UUID?
    @State private var viewportReset = 0
    // The native clip stays at zero; the visible surface is this exact transform.
    // Publishing a GeometryReader frame after every camera packet caused a second
    // full graph render. Keep hit testing current without that layout feedback loop.
    private var globalContentFrame: CGRect {
        contentFrame(in:globalViewportFrame)
    }
    private func contentFrame(in viewport:CGRect)->CGRect {
        guard viewport.width > 0, viewport.height > 0 else {return .zero}
        return CanvasCameraGeometry.contentFrame(viewport:viewport,sceneOrigin:sceneOrigin,
            cameraOrigin:scrollController.logicalOrigin,zoom:zoom,
            size:CanvasViewportGeometry.documentSize(content:documentSize,zoom:zoom,reservedTop:toolbarInset,viewport:viewport.size))
    }
    private var globalViewportFrame:CGRect {get {viewportLayout.frame} nonmutating set {viewportLayout.frame=newValue}}
    @StateObject private var scrollController = CanvasScrollController()
    @StateObject private var viewportPersistence = CanvasViewportPersistence()
    @State private var marquee: CGRect?
    @State private var marqueeInitial = Set<String>()
    @State private var marqueeOrigin: CGPoint?
    private var overviewVisible: Bool {store.canvasViewState.overviewVisible}
    @Environment(\.appearanceTheme) private var appearanceTheme
    private var rose: Color { appearanceTheme.accentColor }
    private var ground: Color { appearanceTheme.colorScheme == .light ? .white : MLTheme.background }

    var body: some View {
        let profile = InteractionProfiler.shared.begin(.canvasBody)
        defer { InteractionProfiler.shared.end(profile) }
        // A value copy shares SwiftUI's installed state wrappers while carrying immutable
        // metadata for this render and its closures. Rebuilt on registry/presentation updates.
        var rendered = self
        let containers=store.inlineContainers
        rendered.renderInlineContainers=containers
        if let adapter=store as? WorkflowCanvasAdapter {rendered.renderEvidence=adapter.canvasRenderEvidence(validation:store.validation)}
        rendered.renderGeometry = InteractionProfiler.shared.measure(.geometry) {makeGeometry(containers:containers)}
        rendered.renderSceneOrigin = rendered.sceneOrigin
        return rendered.renderedBody.workspacePane(.canvas)
    }

    private var renderedBody: some View {
        GeometryReader { viewport in
            Group {
                ScrollView([.horizontal, .vertical]) {
                    graphContent(viewport:viewport.size)
                        .modifier(CanvasCameraSurface(camera:scrollController.translation,controller:scrollController,
                            sceneOrigin:sceneOrigin,contentSize:documentSize,viewport:viewport.size,reservedTop:toolbarInset))

                }
                .frame(width:viewport.size.width,height:viewport.size.height)
                .contentShape(Rectangle())
                .onDrop(of:[UTType.utf8PlainText.identifier,UTType.text.identifier],delegate:CanvasInlineContainerDropDelegate(update:{location in
                    dropTargeted=true
                    let world=paletteDropPoint(location)
                    updateInlineDropTarget(at:CanvasWorldGeometry.scenePoint(world:world,origin:sceneOrigin),moving:[])
                },exit:{dropTargeted=false;inlineDropTarget=nil},perform:{type,location in
                    acceptDrop([type],at:paletteDropPoint(location))
                }))
                .scrollIndicators(.hidden)
                .defaultScrollAnchor(.topLeading)
                .id(viewportReset)
                .background(ground)
                .onChange(of: viewport.frame(in:.global),initial:true) { _,frame in globalViewportFrame=frame }
                .onChange(of: presentation) { previous, next in
                    let origin=scrollController.origin ?? .zero
                    let previousProjection=Self.projection(registry:store.registry,presentation:previous,graph:store.graph,validation:store.validation)
                    let nextProjection=Self.projection(registry:store.registry,presentation:next,graph:store.graph,validation:store.validation)
                    let selected=store.graph.nodes.first {store.selectedNodeIDs.contains($0.id)}
                    let oldAnchor=selected.map {previousProjection.project(CGPoint(x:$0.x,y:$0.y))} ?? CGPoint(x:(origin.x+viewport.size.width/2)/zoom,y:(origin.y+viewport.size.height/2-toolbarInset)/zoom)
                    let logical=previousProjection.inverse(oldAnchor)
                    let nextAnchor=nextProjection.project(logical)
                    let target=CanvasCameraGeometry.reanchored(origin:origin,oldAnchor:oldAnchor,newAnchor:nextAnchor,zoom:zoom)
                    store.pendingSourceID = nil; wirePointer = nil; pendingInsertion = nil
                    movingID = nil; movingIDs = []; movement = .zero; dragProjection = nil
                    scrollController.input.hover.clear()
                    requestViewport(zoom:zoom,origin:target)
                    installCanvasInput()

                }
                .onChange(of: store.graph) { _, _ in fitNotice=nil; installCanvasInput() }
                .onChange(of: documentSize) { _, _ in
                    scrollController.retargetPendingRestore(document:CanvasViewportGeometry.documentSize(content:documentSize,zoom:zoom,reservedTop:toolbarInset,viewport:viewportSize))
                }
                .onChange(of: appearanceTheme.syntaxPaletteFingerprint) { _, _ in installCanvasInput() }
                .onChange(of: zoom) { _, _ in
                    installCanvasInput()
                    scrollController.retargetPendingRestore(document:CanvasViewportGeometry.documentSize(content:documentSize,zoom:zoom,reservedTop:toolbarInset,viewport:viewportSize))
                    installViewportCapture()
                    captureViewport()

                }
                .overlay {
                    Rectangle().stroke(dropTargeted ? rose : (canvasFocused ? MLTheme.rose.opacity(0.65) : .clear), lineWidth: dropTargeted ? 2 : 1)
                        .frame(width: viewport.size.width, height: viewport.size.height)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
                .overlay(alignment: .top) {
                    CanvasCameraTracking(camera:scrollController.translation) { _ in canvasToolbar(viewport:viewport.size) }
                        .onHover {scrollController.input.hover.set("toolbar",hovered:$0)}
                        .onDisappear {scrollController.input.hover.set("toolbar",hovered:false)}
                        .background { GeometryReader { toolbar in Color.clear.preference(key:CanvasToolbarHeightKey.self,value:toolbar.size.height) } }
                        .padding(10)
                }
                .overlay(alignment: .bottomLeading) {
                    if let fitNotice {
                        Text(fitNotice).font(.system(size:11)).foregroundStyle(MLTheme.muted)
                            .padding(10).frame(maxWidth:300,alignment:.leading)
                            .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                            .padding(12).allowsHitTesting(false)
                    }
                }
                .overlay(alignment: .bottomTrailing) {
                    if overviewVisible {
                        CanvasCameraTracking(camera:scrollController.translation) { _ in
                            if overviewIsClear(viewport:viewport.frame(in:.global)) {miniMap(viewport:viewport.frame(in:.global))}
                        }
                            .onHover {scrollController.input.hover.set("overview",hovered:$0)}
                            .onDisappear {scrollController.input.hover.set("overview",hovered:false)}
                            .padding(CanvasOverviewPlacement.margin)
                    }
                }
                .overlay(alignment:.topLeading) {
                    if let marquee {
                        Rectangle().fill(rose.opacity(0.10)).overlay(Rectangle().stroke(rose,lineWidth:1))
                            .frame(width:marquee.width,height:marquee.height)
                            .offset(x:marquee.minX,y:marquee.minY).allowsHitTesting(false).accessibilityHidden(true)
                    }
                }
                .onPreferenceChange(CanvasToolbarHeightKey.self) { height in
                    guard height > 0 else { return }
                    let next=CanvasViewportGeometry.topInset(toolbarHeight:height)
                    if abs(next-toolbarInset)>0.5 { toolbarInset=next }
                }
                .onChange(of: toolbarInset) { _, _ in
                    installCanvasInput()
                    scrollController.retargetPendingRestore(document:CanvasViewportGeometry.documentSize(content:documentSize,zoom:zoom,reservedTop:toolbarInset,viewport:viewportSize))

                }
                .onChange(of: viewport.size, initial: true) { _, size in
                    fitNotice=nil
                    viewportSize = size
                    scrollController.retargetPendingRestore(document:CanvasViewportGeometry.documentSize(content:documentSize,zoom:zoom,reservedTop:toolbarInset,viewport:size))
                }
                .task(id: InitialCanvasLayout(size:viewport.size,toolbarInset:toolbarInset)) {
                    guard !hasFitted, viewport.size.width > 200, viewport.size.height > 150 else { return }
                    // Split views report usable but temporary sizes during launch.
                    // Restart when geometry changes; never refit a restored or user-chosen view.
                    do { try await Task.sleep(for:.milliseconds(150)) } catch { return }
                    guard !Task.isCancelled, !hasFitted else { return }
                    if !store.canvasViewState.isInitialViewport { restoreViewport() }
                    else { fit(to:viewport.size) }
                    hasFitted = true
                    viewportReset += 1
                }
                .onAppear {
                    installViewportCapture()
                    installCanvasInput()
                    if restoredRevision != store.canvasViewRevision, !store.canvasViewState.isInitialViewport { restoreViewport() }
                    else { restoredRevision = store.canvasViewRevision }

                }
            }
        }
        .onChange(of: store.canvasRevealRequest) { _, request in
            guard let request,hasFitted,viewportSize.width > 0,viewportSize.height > toolbarInset else {return}
            let rectangles=store.graph.nodes.filter {request.nodeIDs.contains($0.id)}.map {CGRect(origin:worldPosition($0),size:nodeSize($0))}
            guard let first=rectangles.first else {return}
            let bounds=rectangles.dropFirst().reduce(first) {$0.union($1)}
            if request.fitsBounds {
                let next = CanvasViewportGeometry.fitZoom(bounds: bounds, viewport: CGSize(width: viewportSize.width - 56, height: viewportSize.height), reservedTop: toolbarInset, maximum: CanvasZoomLimits.maximum)
                requestViewport(zoom: next, origin: CanvasViewportGeometry.selectionOrigin(bounds: bounds, zoom: next, reservedTop: toolbarInset))
                return
            }
            let saved=store.canvasViewState
            let origin=scrollController.hasPendingRestore ? CGPoint(x:saved.scrollX,y:saved.scrollY) : (scrollController.origin ?? .zero)
            let target=CanvasViewportGeometry.revealOrigin(bounds:bounds,zoom:zoom,origin:origin,viewport:viewportSize,reservedTop:toolbarInset)
            if target != origin {requestViewport(zoom:zoom,origin:target)}
        }
        .onChange(of: store.canvasViewRevision) { _, _ in restoreViewport() }
        .onChange(of: store.selectedNodeIDs) { _, _ in captureViewport() }
        .onChange(of: store.selectedEdgeID) { _, _ in captureViewport() }
        .onDisappear { captureViewport(); scrollController.onOriginChanged = nil; scrollController.input.stop() }
        .modifier(CanvasHistoryFocusModifier(provider: historyProvider, enabled: publishesHistoryFocus))
        .onAppear { if publishesHistoryFocus { historyProvider.attach(store) } }
        .onChange(of: store.historyContextID) { _, _ in if publishesHistoryFocus { historyProvider.attach(store) } }
        .onDisappear { historyProvider.detach() }
        .focusable().focusEffectDisabled().focused($canvasFocused)
        .onKeyPress { press in
            guard canvasFocused else { return .ignored }
            if press.key == .escape {
                // Native text editing owns Escape before canvas popovers or gestures.
                if let text = NSApp.keyWindow?.firstResponder as? NSTextView, text.isEditable { return .ignored }
                if portOptions { portOptions = false; return .handled }
                if connectionOptions { connectionOptions = false; return .handled }
                pendingInsertion = nil; store.pendingSourceID = nil; wirePointer = nil
                return .handled
            }
            guard !store.isBusy else { return .ignored }
            if press.modifiers.contains(.command), press.characters.lowercased() == "a" {
                store.clearSelection()
                for item in store.graph.nodes { store.selectNode(item.id, extending: true) }
                return .handled
            }
            if press.modifiers.contains(.command), press.characters.lowercased() == "d", !store.selectedNodeIDs.isEmpty {
                store.duplicateSelection(); return .handled
            }
            if press.key == .delete || CanvasKeyRouting.isDelete(press.characters), !store.selectedNodeIDs.isEmpty || store.selectedEdgeID != nil { store.deleteSelection(); return .handled }
            let distance: CGFloat = press.modifiers.contains(.shift) ? 20 : 5
            let vector: CGSize
            switch press.key {
            case .leftArrow: vector = CGSize(width: -distance, height: 0)
            case .rightArrow: vector = CGSize(width: distance, height: 0)
            case .upArrow: vector = CGSize(width: 0, height: -distance)
            case .downArrow: vector = CGSize(width: 0, height: distance)
            default: return .ignored
            }
            guard !store.selectedNodeIDs.isEmpty else { return .ignored }
            let delta = CanvasInteractionGeometry.translation(modelDelta(vector), origins: store.graph.nodes.filter { store.selectedNodeIDs.contains($0.id) }.map { CGPoint(x: $0.x, y: $0.y) })
            store.moveNodes(ids: store.selectedNodeIDs, dx: delta.width, dy: delta.height)
            return .handled
        }
        .overlay(alignment: .bottomLeading) {
            if let proposal = pendingInsertion {
                HStack {
                    Text("Insert \(store.canvasCatalog.component(type: proposal.type)?.title ?? proposal.type) into the selected connection?")
                    Button("Insert") {
                        guard store.canvasInsertionComponents(edgeID:proposal.edgeID).contains(where:{$0.id == proposal.type}) else { pendingInsertion = nil; return }
                        store.insertNode(type: proposal.type, edgeID: proposal.edgeID, x: proposal.location.x, y: proposal.location.y)
                        pendingInsertion = nil
                    }.disabled(store.isBusy)
                    Button("Cancel") { pendingInsertion = nil }
                }.font(.caption).padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(rose)).padding(12)
            } else if let sourceID = store.pendingSourceID, let source = node(sourceID) {
                HStack(spacing: 10) {
                    Image(systemName: "point.3.connected.trianglepath.dotted")
                    Text("Connect \(source.name): choose an input port")
                    Button("Cancel") { store.pendingSourceID = nil; wirePointer = nil }
                        .buttonStyle(.borderless)
                }
                .font(.caption).padding(10)
                .background(MLTheme.panel)
                .overlay(Rectangle().stroke(rose.opacity(0.5)))
                .padding(12)
            }
        }
    }

    private func restoreViewport() {
        guard restoredRevision != store.canvasViewRevision else { return }
        let state = CanvasViewportProjection.restored(store.canvasViewState, nodes: Set(store.graph.nodes.map(\.id)), edges: Set(store.graph.edges.map(\.id)))
        restoredRevision = store.canvasViewRevision
        hasFitted = true
        zoom = state.zoom
        scrollController.requestRestore(to: CGPoint(x: state.scrollX, y: state.scrollY), document: CanvasViewportGeometry.documentSize(content:documentSize,zoom:state.zoom,reservedTop:toolbarInset,viewport:viewportSize))
        store.clearSelection()
        for id in state.selectedNodeIDs.sorted() { store.selectNode(id, extending: true) }
        if state.selectedNodeIDs.isEmpty, let id = state.selectedEdgeIDs.sorted().first { store.selectEdge(id) }
        installViewportCapture()
    }

    private func installViewportCapture() {
        guard hasFitted else { return }
        let scale = zoom
        let revision = store.canvasViewRevision
        let token = viewportPersistence.beginCapture()
        scrollController.onOriginChanged = { [weak store, weak viewportPersistence] origin in
            guard let store, let viewportPersistence else { return }
            viewportPersistence.capture(store:store,revision:revision,token:token,zoom:scale,origin:origin)
        }
    }

    /// Save intent synchronously; native acknowledgement may occur after the user leaves this context.
    private func requestViewport(zoom next:Double,origin:CGPoint,liveCamera:Bool = false) {
        let revision = restoredRevision ?? store.canvasViewRevision
        guard store.canvasViewRevision == revision else { return }
        fitNotice=nil
        hasFitted=true
        zoom=next
        viewportPersistence.request(store:store,revision:revision,zoom:next,origin:origin)
        installViewportCapture()
        if liveCamera,scrollController.scrollView?.documentView != nil {
            // The native clip stays at zero; the live camera is a SwiftUI transform.
            // Pair its origin with scale now instead of painting one frame with
            // the previous origin while waiting for resized document bounds.
            scrollController.navigate(to:origin)
        } else {
            scrollController.requestRestore(to:origin,document:CanvasViewportGeometry.documentSize(content:documentSize,zoom:next,reservedTop:toolbarInset,viewport:viewportSize))
        }
    }
    private func setZoom(_ next:Double) {
        let state=store.canvasViewState
        let origin=scrollController.hasPendingRestore ? CGPoint(x:state.scrollX,y:state.scrollY) : (scrollController.origin ?? CGPoint(x:state.scrollX,y:state.scrollY))
        requestViewport(zoom:next,origin:origin)
    }

    private func captureViewport() {
        guard hasFitted, restoredRevision == store.canvasViewRevision, !scrollController.hasPendingRestore else { return }
        let origin = scrollController.origin ?? .zero
        var state = store.canvasViewState
        state.zoom = zoom; state.scrollX = origin.x; state.scrollY = origin.y
        state.selectedNodeIDs = store.selectedNodeIDs; state.selectedEdgeIDs = store.selectedEdgeID.map { [$0] } ?? []
        if store.canvasViewState != state { store.canvasViewState = state }
    }

    private func installCanvasInput() {
        let input=scrollController.input
        input.controller=scrollController;input.reservedTop=toolbarInset
        scrollController.gridScale=zoom;scrollController.reservedTop=toolbarInset
        scrollController.gridBackground=appearanceTheme.colorScheme == .light ? .white : appearanceTheme.native.background;scrollController.gridColor=appearanceTheme.native.rule
        input.keyboardPanAllowed = {canvasFocused}
        input.viewportPoint = {windowPoint,window in
            guard let clip=scrollController.scrollView?.contentView,globalViewportFrame.width > 0 else {return nil}
            return CanvasCameraGeometry.viewportPoint(windowPoint:windowPoint,window:window,viewportFrame:globalViewportFrame,clip:clip)
        }
        input.onConnectionPress = { point in
            guard store.allowsInspection,globalContentFrame.width > 0 else {return false}
            let location=CanvasCameraGeometry.renderedGraphPoint(viewportPoint:point,viewportFrame:globalViewportFrame,contentFrame:globalContentFrame,zoom:zoom,reservedTop:toolbarInset)
            guard !store.graph.nodes.contains(where:{nodeHit($0,at:location)}) else {return false}
            if let link=inlineBoundaryAt(location),let adapter=store as? WorkflowCanvasAdapter {
                canvasFocused=true
                return adapter.inspectPort(nodeID:link.containerNodeID ?? link.childNodeID,port:link.routed ? link.containerPort:link.childPort,direction:link.direction)
            }
            guard let edge=Self.wireAt(CanvasWorldGeometry.worldPoint(scene:location,origin:sceneOrigin),graph:store.graph,registry:store.registry,zoom:zoom,presentation:presentation,geometry:geometry) else {return false}
            store.selectEdge(edge);canvasFocused=true
            return true
        }
        input.onCancel = {
            guard marqueeOrigin != nil else {return}
            if store.allowsInspection {store.clearSelection();for id in marqueeInitial.sorted() {store.selectNode(id,extending:true)}}
            marquee=nil;marqueeOrigin=nil
        }
        input.onZoom = { pointer,delta,precise in
            let state=store.canvasViewState
            let origin=scrollController.hasPendingRestore ? CGPoint(x:state.scrollX,y:state.scrollY) : (scrollController.origin ?? CGPoint(x:state.scrollX,y:state.scrollY))
            let next=CanvasCameraGeometry.zoom(origin:origin,zoom:zoom,pointer:pointer,delta:delta,precise:precise,reservedTop:toolbarInset,sensitivity:zoomSensitivity)
            var transaction=Transaction(animation:nil)
            transaction.disablesAnimations=true
            withTransaction(transaction) {
                requestViewport(zoom:next.zoom,origin:next.origin,liveCamera:true)
            }
        }
        input.canMarquee = { point in
            guard store.allowsInspection else {return false}
            guard globalContentFrame.width > 0 else {return false}
            let location=CanvasCameraGeometry.renderedGraphPoint(viewportPoint:point,viewportFrame:globalViewportFrame,contentFrame:globalContentFrame,zoom:zoom,reservedTop:toolbarInset)
            guard !store.graph.nodes.contains(where:{nodeHit($0,at:location)}) else {return false}
            guard inlineBoundaryAt(location)==nil else {return false}
            return Self.wireAt(CanvasWorldGeometry.worldPoint(scene:location,origin:sceneOrigin),graph:store.graph,registry:store.registry,zoom:zoom,presentation:presentation,geometry:geometry) == nil
        }
        input.onMarquee = { start,end,extending,complete in
            guard store.allowsInspection else {marquee=nil;marqueeOrigin=nil;return}
            if marqueeOrigin == nil {
                marqueeOrigin=CGPoint(x:globalViewportFrame.minX-globalContentFrame.minX,y:globalViewportFrame.minY-globalContentFrame.minY)
                marqueeInitial=store.selectedNodeIDs
            }
            let origin=marqueeOrigin ?? .zero
            func graphPoint(_ point:CGPoint)->CGPoint {CanvasViewportGeometry.graphLocation(documentPoint:CGPoint(x:origin.x+point.x,y:origin.y+point.y),zoom:zoom,reservedTop:toolbarInset)}
            let bounds=Dictionary(uniqueKeysWithValues:store.graph.nodes.map {($0.id,nodeHitBounds($0))})
            let selected=CanvasCameraGeometry.selection(start:graphPoint(start),end:graphPoint(end),bounds:bounds,initial:marqueeInitial,extending:extending)
            if selected != store.selectedNodeIDs || store.selectedEdgeID != nil {
                store.clearSelection(); for id in selected.sorted() {store.selectNode(id,extending:true)}
            }
            canvasFocused=true;store.pendingSourceID=nil;wirePointer=nil
            marquee=complete ? nil:CanvasCameraGeometry.rectangle(start:start,end:end)
            if complete {marqueeOrigin=nil}
        }
        input.install()
    }

    private func canvasToolbar(viewport: CGSize) -> some View {
        ViewThatFits(in: .horizontal) {
            toolbarRow(viewport:viewport)
            VStack(alignment:.trailing,spacing:6) {
                selectionControls
                zoomControls(viewport:viewport)
            }
            .padding(4).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
        }.frame(maxWidth:.infinity,alignment:.trailing)
    }
    private func toolbarRow(viewport: CGSize) -> some View {
        HStack(spacing:6) {
            selectionControls
            if !store.selectedNodeIDs.isEmpty || store.selectedEdgeID != nil {
                Rectangle().fill(MLTheme.rule).frame(width:1,height:16).padding(.horizontal,2)
            }
            zoomControls(viewport:viewport)
        }.fixedSize(horizontal:true,vertical:false)
            .padding(4).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func zoomControls(viewport: CGSize) -> some View {
        HStack(spacing: 4) {
            CanvasToolbarButton(title: "Overview", symbol: "map", compact: true, help: overviewVisible ? "Hide overview" : "Show overview") {var state=store.canvasViewState;state.overviewVisible.toggle();store.canvasViewState=state}
            CanvasToolbarButton(title: "Zoom out", symbol: "minus", compact: true) { setZoom(max(CanvasZoomLimits.minimum, zoom - 0.1)) }
                .disabled(zoom <= CanvasZoomLimits.minimum)
            Button { setZoom(1) } label: {
                Image(systemName:"arrow.counterclockwise").font(.system(size:11,weight:.medium))
                    .frame(width:28,height:28).contentShape(Rectangle())
            }.buttonStyle(.plain).help("Canvas zoom \(Int((zoom * 100).rounded())) percent · Reset to 100 percent")
                .accessibilityLabel("Canvas zoom \(Int((zoom * 100).rounded())) percent. Reset to 100 percent")
            CanvasToolbarButton(title: "Zoom in", symbol: "plus", compact: true) { setZoom(min(CanvasZoomLimits.maximum, zoom + 0.1)) }
                .disabled(zoom >= CanvasZoomLimits.maximum)
            Rectangle().fill(MLTheme.rule).frame(width: 1, height: 16).padding(.horizontal, 2)
            if !store.selectedNodeIDs.isEmpty {
                CanvasToolbarButton(title: "Fit selection", symbol: "scope", compact: true, help: "Fit and navigate to selected blocks") { focusSelection(in: viewport) }
            }
            CanvasToolbarButton(title: "Fit all", symbol: "arrow.up.left.and.arrow.down.right", compact: true, help: "Fit the architecture in the available canvas") {
                fit(to: viewport); viewportReset += 1
            }
        }
    }
    @ViewBuilder private var selectionControls: some View {
        if !store.selectedNodeIDs.isEmpty || store.selectedEdgeID != nil {
            HStack(spacing: 4) {
                if !store.selectedNodeIDs.isEmpty {
                    if showsPortControls { CanvasToolbarButton(title:"Ports",symbol:"point.3.connected.trianglepath.dotted",compact:true,help:"Inspect ports and connect using explicit choices") { portOptions = true }
                        .popover(isPresented:$portOptions) { portChooser.padding(14).frame(width:350).background(MLTheme.panel) } }
                    CanvasToolbarButton(title: "Duplicate", symbol: "doc.on.doc", compact: true, help: "Independent copy of selected blocks and their internal connections · ⌘D") { store.duplicateSelection() }.disabled(store.isBusy)
                    CanvasToolbarButton(title: "Delete", symbol: "trash", compact: true) { store.deleteSelection() }.disabled(store.isBusy)
                } else if let edge = store.graph.edges.first(where: { $0.id == store.selectedEdgeID }) {
                    if showsPortControls { CanvasToolbarButton(title: "Edit connection", symbol: "slider.horizontal.3", compact: true, help:connectionSummary(edge)) { connectionOptions = true }
                        .popover(isPresented: $connectionOptions, arrowEdge: .bottom) { connectionEditor(edge).padding(14).frame(width: 310).background(MLTheme.panel) }.disabled(store.isBusy) }
                    CanvasToolbarButton(title: "Delete wire", symbol: "trash", compact: true) { store.deleteSelection() }.disabled(store.isBusy)
                }
            }
            .accessibilityElement(children:.contain)
            .accessibilityLabel(store.selectedEdgeID == nil ? "\(store.selectedNodeIDs.count) selected blocks" : "Selected connection actions")
        }
    }

    private var portChooser: some View {
        ThemedScrollView {
            VStack(alignment:.leading,spacing:12) {
                Text("Ports & connections").font(.headline)
                Text(store.isBusy ? "Read-only port inspection. Connection editing is unavailable in the current state." : "Choose an output, then an input. Each port remains a separate endpoint at every display size.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(store.graph.nodes.filter { store.pendingSourceID != nil || store.selectedNodeIDs.contains($0.id) }) { item in
                    Text(item.name).font(.caption.weight(.semibold))
                    ForEach(geometry.definition(nodeID:item.id)?.outputs ?? [],id: \.self) { port in
                        portChoice(item, port:port, output:true)
                    }
                    ForEach(geometry.definition(nodeID:item.id)?.inputs ?? [],id: \.self) { port in
                        portChoice(item, port:port, output:false)
                    }
                    Divider()
                }
                if store.pendingSourceID != nil { Button("Cancel connection") { store.pendingSourceID=nil;wirePointer=nil } }
            }
        }.frame(maxHeight:400)
    }

    private func portChoice(_ item: GraphNode, port: String, output: Bool) -> some View {
        let descriptor = store.canvasPortDescriptor(nodeID:item.id,port:port,direction:output ? .output : .input)
        let selected = store.pendingSourceID == item.id && pendingOutputPort == port
        let label = port + " · " + descriptor.accessibilityDescription
        return HStack(alignment: .center, spacing: 8) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    CanvasPortGlyphView(glyph:descriptor.glyph)
                    Text(port).font(.system(size: 11, weight: .semibold))
                    Text(output ? "Output" : "Input").font(.caption2).foregroundStyle(MLTheme.muted)
                }
                Text(descriptor.accessibilityDescription).font(.caption2).foregroundStyle(MLTheme.muted)
            }.frame(maxWidth: .infinity, alignment: .leading)
            if let adapter = store as? WorkflowCanvasAdapter {
                Button("Inspect") {
                    if adapter.inspectPort(nodeID:item.id,port:port,direction:output ? .output : .input) { portOptions = false }
                }.buttonStyle(FlatButtonStyle())
                    .accessibilityLabel("Inspect " + item.name + ", " + (output ? "output " : "input ") + port)
            }
            Button {
                if output { pendingOutputPort=port;store.pendingSourceID=item.id;wirePointer=nil }
                else if let source=store.pendingSourceID { store.connect(source:source,sourcePort:pendingOutputPort,target:item.id,targetPort:port);wirePointer=nil }
            } label: {
                Image(systemName: output ? (selected ? "checkmark" : "arrow.right") : "arrow.right.to.line")
            }.buttonStyle(FlatButtonStyle()).disabled(store.isBusy || (!output && store.pendingSourceID == nil))
                .accessibilityLabel(item.name + ", " + label + (output ? ", start connection" : ", connect selected output"))
        }.padding(8).overlay(Rectangle().stroke(MLTheme.rule))
    }

    private func connectionSummary(_ edge:GraphEdge)->String {
        (node(edge.source)?.name ?? edge.source)+"."+edge.sourcePort+" → "+(node(edge.target)?.name ?? edge.target)+"."+edge.targetPort
    }

    private func connectionEditor(_ edge: GraphEdge) -> some View {
        let sources = store.graph.nodes.flatMap { item in
            (geometry.definition(nodeID: item.id)?.outputs ?? []).map { (item.id, $0, "\(item.name) · \($0)") }
        }
        let targets = store.graph.nodes.flatMap { item in
            (geometry.definition(nodeID: item.id)?.inputs ?? []).map { (item.id, $0, "\(item.name) · \($0)") }
        }
        return VStack(alignment: .leading, spacing: 10) {
            Text("Edit connection").font(.headline)
            Text(connectionSummary(edge)).font(.caption.weight(.semibold)).textSelection(.enabled)
            Text(store.canvasPortDescriptor(nodeID:edge.source,port:edge.sourcePort,direction:.output).accessibilityDescription+" · "+previewStatus(edge.source))
                .font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            Text("Choose a new endpoint. Invalid connections leave the current wire unchanged.").font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title: "Connection source", selection: sources.firstIndex(where: { $0.0 == edge.source && $0.1 == edge.sourcePort }).map(String.init) ?? "",
                options: sources.enumerated().map { FlatChoiceOption(id: String($0.offset), title: $0.element.2) }) { value in
                    guard let index = Int(value), sources.indices.contains(index) else { return }
                    store.reconnect(edgeID: edge.id, source: sources[index].0, sourcePort: sources[index].1, target: edge.target, targetPort: edge.targetPort)
                    connectionOptions = false
                }
            FlatChoiceControl(title: "Connection target", selection: targets.firstIndex(where: { $0.0 == edge.target && $0.1 == edge.targetPort }).map(String.init) ?? "",
                options: targets.enumerated().map { FlatChoiceOption(id: String($0.offset), title: $0.element.2) }) { value in
                    guard let index = Int(value), targets.indices.contains(index) else { return }
                    store.reconnect(edgeID: edge.id, source: edge.source, sourcePort: edge.sourcePort, target: targets[index].0, targetPort: targets[index].1)
                    connectionOptions = false
                }
            if !store.canvasInsertionComponents(edgeID:edge.id).isEmpty {
                FlatChoiceControl(title: "Insert component into connection", selection: "", options: store.canvasInsertionComponents(edgeID:edge.id).map { FlatChoiceOption(id: $0.id, title: $0.title) }, placeholder: "Insert a component…") { type in
                    guard store.canvasInsertionComponents(edgeID:edge.id).contains(where:{$0.id == type}),
                          let source = node(edge.source), let target = node(edge.target) else { return }
                    let origin = CanvasInteractionGeometry.boundedOrigin(CGPoint(x:(source.x+target.x)/2,y:(source.y+target.y)/2))
                    store.insertNode(type: type, edgeID: edge.id, x: origin.x, y: origin.y)
                    connectionOptions = false
                }
            }
        }
    }

    private static func wireAt(_ point: CGPoint, graph: GraphDocument, registry: ComponentRegistry, zoom: Double, presentation: CanvasPresentationSettings, geometry: CanvasGeometrySnapshot) -> String? {
        func anchor(_ node: GraphNode, output: Bool, port: String) -> CGPoint {
            let offset = geometry.portAnchor(nodeID: node.id, output: output, port: port)
            let origin = geometry.projection.project(CGPoint(x:node.x,y:node.y))
            return CGPoint(x: origin.x + offset.x, y: origin.y + offset.y)
        }
        let matches = graph.edges.filter { edge in
            guard let source = graph.nodes.first(where: { $0.id == edge.source }), let target = graph.nodes.first(where: { $0.id == edge.target }) else { return false }
            return CanvasInteractionGeometry.hitsWire(point:point,
                start:anchor(source,output:true,port:edge.sourcePort),
                end:anchor(target,output:false,port:edge.targetPort),zoom:zoom)
        }
        return matches.count == 1 ? matches[0].id : nil
    }

    static func dropOrigin(_ point: CGPoint, type: String, registry: ComponentRegistry, catalog: ComponentRegistry, presentation: CanvasPresentationSettings, geometry: CanvasGeometrySnapshot? = nil) -> CGPoint {
        let size = size(type: type,registry: catalog,presentation: presentation)
        let origin = (geometry?.projection ?? projection(registry:registry,presentation:presentation)).inverse(CGPoint(x:point.x-size.width/2,y:point.y-size.height/2))
        return CanvasInteractionGeometry.boundedOrigin(origin)
    }

    private func overviewIsClear(viewport:CGRect)->Bool {
        let frame=contentFrame(in:viewport)
        guard frame.width>0 else {return false}
        return CanvasOverviewPlacement.shouldShow(viewport:viewport.size,toolbarInset:toolbarInset,
            graphNodes:store.graph.nodes.map { CGRect(origin:position($0),size:nodeSize($0)) },zoom:zoom,
            scrollOrigin:CGPoint(x:viewport.minX-frame.minX,y:viewport.minY-frame.minY))
    }

    private func miniMap(viewport: CGRect) -> some View {
        let graph=store.graph
        let index=CanvasNodeIndex(nodes:graph.nodes)
        let frame=contentFrame(in:viewport)
        return Canvas { context, size in
            let extent = CanvasViewportGeometry.graphExtent(content:documentSize,zoom:zoom,reservedTop:toolbarInset,viewport:viewport.size)
            let factor = min(size.width / extent.width, size.height / extent.height)
            for edge in graph.edges {
                if let source = index.node(edge.source), let target = index.node(edge.target) {
                    var path = Path()
                    let start = portPosition(source,output:true,port:edge.sourcePort), end = portPosition(target,output:false,port:edge.targetPort)
                    path.move(to: CGPoint(x:start.x*factor,y:start.y*factor))
                    path.addLine(to: CGPoint(x:end.x*factor,y:end.y*factor))
                    context.stroke(path, with: .color(MLTheme.muted.opacity(0.45)), lineWidth: 0.5)
                }
            }
            for item in graph.nodes {
                let rect = CGRect(x: position(item).x * factor, y: position(item).y * factor, width: nodeSize(item).width * factor, height: nodeSize(item).height * factor)
                context.fill(Path(rect), with: .color(ComponentStyle.tint(store.canvasIconType(nodeID:item.id)).opacity(store.selectedNodeIDs.contains(item.id) ? 0.8 : 0.4)))
            }
            if frame.width > 0 {
                let graphVisible = CanvasViewportGeometry.visibleGraphRect(
                    scrollOrigin:CGPoint(x:viewport.minX-frame.minX,y:viewport.minY-frame.minY),
                    viewport:viewport.size,zoom:zoom,reservedTop:toolbarInset)
                let visible = CGRect(x:graphVisible.minX*factor,y:graphVisible.minY*factor,width:graphVisible.width*factor,height:graphVisible.height*factor)
                    .intersection(CGRect(origin:.zero,size:size))
                context.stroke(Path(visible), with: .color(rose), lineWidth: 1)
            }
        }
        .frame(width:CanvasOverviewPlacement.mapSize.width,height:CanvasOverviewPlacement.mapSize.height)
        .background(MLTheme.background.opacity(0.94))
        .overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 0.75))
        .allowsHitTesting(false)
        .accessibilityLabel("Architecture overview. Right-drag, Control-drag, middle drag or Space-drag to pan; scroll to zoom. Drag empty canvas to select.")
    }

    private func focusSelection(in viewport: CGSize) {
        let items = store.graph.nodes.filter { store.selectedNodeIDs.contains($0.id) }
        guard !items.isEmpty else { return }
        let bounds = items.map { CGRect(origin:worldPosition($0),size:nodeSize($0)) }.reduce(CGRect.null) { $0.union($1) }
        let next = CanvasViewportGeometry.fitZoom(bounds:bounds,viewport:CGSize(width:viewport.width-56,height:viewport.height),reservedTop:toolbarInset,maximum:CanvasZoomLimits.maximum)
        requestViewport(zoom:next,origin:CanvasViewportGeometry.selectionOrigin(bounds:bounds,zoom:next,reservedTop:toolbarInset))
    }

    private func fit(to viewport: CGSize) {
        let next = CanvasViewportGeometry.fitAllViewport(nodes:store.graph.nodes.map { CGRect(origin:worldPosition($0),size:nodeSize($0)) },viewport:viewport,reservedTop:toolbarInset)
        requestViewport(zoom:next.zoom,origin:next.origin)
        fitNotice = next.fullyFits ? nil : "Fit is limited to 1% zoom. Enlarge the canvas or pan to inspect the remaining components."
    }

    private func paletteDropPoint(_ location:CGPoint)->CGPoint {
        let origin=scrollController.origin ?? .zero
        return CanvasViewportGeometry.graphLocation(documentPoint:CGPoint(x:origin.x+location.x,y:origin.y+location.y),zoom:zoom,reservedTop:toolbarInset)
    }

    /// The drop modifier lives outside the scale transform, so its location is explicitly
    /// converted to model coordinates. Node/port gestures instead use the unscaled named space.
    private func acceptDrop(_ types: [String], at location: CGPoint) -> Bool {
        guard !store.isBusy, let type = types.first, store.canAddType(type) else { return false }
        guard location.x.isFinite,location.y.isFinite,
              CanvasInteractionGeometry.originRange.contains(location.x),CanvasInteractionGeometry.originRange.contains(location.y) else {return false}
        if let edgeID = Self.wireAt(location, graph: store.graph, registry: store.registry, zoom: zoom, presentation: presentation, geometry: geometry),
           store.canvasInsertionComponents(edgeID:edgeID).contains(where:{$0.id == type}) {
            store.selectEdge(edgeID)
            pendingInsertion = CanvasInsertion(type: type, edgeID: edgeID,
                location: Self.dropOrigin(location,type:type,registry:store.registry,catalog:store.canvasCatalog,presentation:store.canvasViewState.presentation,geometry:geometry))
            return true
        }
        let origin = Self.dropOrigin(location,type:type,registry:store.registry,catalog:store.canvasCatalog,presentation:presentation,geometry:geometry)
        let before = Set(store.graph.nodes.map(\.id))
        store.addNode(type:type,x:origin.x,y:origin.y)
        let added = before != Set(store.graph.nodes.map(\.id))
        if added { canvasFocused = true }
        return added
    }

    private func graphContent(viewport: CGSize) -> some View {
        let profile=InteractionProfiler.shared.begin(.canvasScene)
        defer {InteractionProfiler.shared.end(profile)}
        let graph=store.graph
        let index=CanvasNodeIndex(nodes:graph.nodes)
        return ZStack(alignment: .topLeading) {
                ground
                    .contentShape(Rectangle())

                    .onTapGesture {
                        store.clearSelection()
                        canvasFocused = true
                        store.pendingSourceID = nil
                        wirePointer = nil
                    }
                ForEach(inlineContainers.filter(\.isExpanded)) {container in
                    if let item=index.node(container.id) {
                        CanvasInlineContainerFrame(size:nodeSize(item),tint:ComponentStyle.tint(store.canvasIconType(nodeID:item.id)),
                            selected:store.selectedNodeIDs.contains(item.id),targeted:inlineDropTarget==item.id)
                            .offset(x:position(item).x,y:position(item).y)
                    }
                }
                ZStack(alignment: .topLeading) {
                    ForEach(graph.edges) { edge in
                        if let source = index.node(edge.source), let target = index.node(edge.target) {
                            wire(edge: edge, source: source, target: target)
                        }
                    }
                }
                .accessibilityElement(children: .contain)
                .accessibilityLabel("Connections")
                ForEach(inlineContainers.filter(\.isExpanded)) {container in
                    if let item=index.node(container.id) {
                        inlineBoundaryWires(container,item:item,index:index)
                    }
                }
                if let sourceID = store.pendingSourceID, let source = index.node(sourceID), let wirePointer {
                    curve(from: portPosition(source, output: true, port: pendingOutputPort), to: wirePointer)
                        .stroke(rose, style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                        .allowsHitTesting(false)
                }
                ZStack(alignment: .topLeading) {
                    ForEach(graph.nodes.filter {inlineContainer($0.id)?.isExpanded == true}) {item in
                        block(item,viewport:viewport).offset(x:position(item).x,y:position(item).y)
                    }
                    ForEach(graph.nodes.filter {inlineContainer($0.id)?.isExpanded != true}) { item in
                        block(item,viewport:viewport)
                            .offset(x: position(item).x, y: position(item).y)
                            .accessibilityElement(children: .contain)
                            .accessibilityLabel(item.name.isEmpty ? "Component" : item.name)
                    }
                }
                .accessibilityElement(children: .contain)
                .accessibilityLabel("Components")
            }
            .frame(width:documentSize.width,height:documentSize.height)
            .coordinateSpace(name: "graphCanvas")
            .overlay(alignment: .topLeading) {
                if graph.nodes.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Build your architecture").font(.title3.weight(.semibold))
                        Text("Drag a component from the library, or use its Add button.")
                            .foregroundStyle(.secondary)
                        Text("Connect an output port to an input port. Select a block to configure it.")
                            .foregroundStyle(.secondary)
                    }.padding(40).allowsHitTesting(false)
                }
            }

    }

    private func node(_ id: String) -> GraphNode? { store.graph.nodes.first { $0.id == id } }

    private func position(_ item: GraphNode) -> CGPoint {
        CanvasWorldGeometry.scenePoint(world:worldPosition(item),origin:sceneOrigin)
    }
    private func worldPosition(_ item: GraphNode) -> CGPoint {
        projection.project(CGPoint(x:item.x + (movingIDs.contains(item.id) ? movement.width : 0),y:item.y + (movingIDs.contains(item.id) ? movement.height : 0)))
    }
    private func portOffset(_ item: GraphNode, output: Bool, port: String) -> CGPoint {
        geometry.portAnchor(nodeID: item.id, output: output, port: port)
    }
    private func portPosition(_ item: GraphNode, output: Bool, port: String) -> CGPoint {
        let offset = portOffset(item,output:output,port:port), origin = position(item)
        return CGPoint(x:origin.x+offset.x,y:origin.y+offset.y)
    }
    private func curve(from: CGPoint, to: CGPoint) -> Path {
        let controls = CanvasNodeGeometry.controlPoints(start:from,end:to,direction:direction)
        return Path { path in path.move(to:from);path.addCurve(to:to,control1:controls.0,control2:controls.1) }
    }

    private func wire(edge: GraphEdge, source: GraphNode, target: GraphNode) -> some View {
        let start = portPosition(source, output: true, port: edge.sourcePort)
        let end = portPosition(target, output: false, port: edge.targetPort)
        let selected = store.selectedEdgeID == edge.id
        let executing = renderEvidence.map {$0.executingEdges.contains(edge.id)} ?? ((store as? WorkflowCanvasAdapter)?.isCanvasEdgeExecuting(edge.id) ?? false)
        let flowStatus = previewStatus(source.id)
        let descriptor = store.canvasPortDescriptor(nodeID:source.id,port:edge.sourcePort,direction:.output)
        let path = curve(from: start, to: end)
        return ZStack(alignment: .topLeading) {
            path.stroke(executing ? Color.blue : selected ? rose : (flowStatus == "Stale" ? Color.orange.opacity(0.7) : MLTheme.teal.opacity(flowStatus == "Not evaluated" ? 0.45 : 0.8)),
                        style: StrokeStyle(lineWidth: executing || selected ? 2 : 1.25, dash: descriptor.kind == .fittedState ? [2,3] : (flowStatus == "Stale" ? [5,4] : [])))
            Path { arrow in
                let offsets = [CGPoint(x:-7,y:0),CGPoint(x:-15,y:-3.5),CGPoint(x:-15,y:3.5)]
                arrow.move(to:CGPoint(x:end.x+offsets[0].x,y:end.y+offsets[0].y))
                arrow.addLine(to:CGPoint(x:end.x+offsets[1].x,y:end.y+offsets[1].y))
                arrow.addLine(to:CGPoint(x:end.x+offsets[2].x,y:end.y+offsets[2].y))
                arrow.closeSubpath()
            }
            .fill(executing ? Color.blue : selected ? rose : MLTheme.teal)
            .allowsHitTesting(false)
            path.stroke(.clear, lineWidth: 16)
                .contentShape(path.strokedPath(StrokeStyle(lineWidth: 16)))
                .onContinuousHover(coordinateSpace:.named("graphCanvas")) { phase in
                    switch phase {
                    case .active(let location):
                        scrollController.input.hover.set("wire:"+edge.id,hovered:path.strokedPath(StrokeStyle(lineWidth:16)).contains(location))
                    case .ended: scrollController.input.hover.set("wire:"+edge.id,hovered:false)
                    }
                }
                .onDisappear {scrollController.input.hover.set("wire:"+edge.id,hovered:false)}
                .accessibilityLabel("Connection from \(source.name) to \(target.name), input \(edge.targetPort). \(descriptor.accessibilityDescription). \(flowStatus)")
                .accessibilityAddTraits(.isButton)
                .accessibilityAction {
                    store.selectEdge(edge.id)
                }
                .allowsHitTesting(false)

        }
    }

    private func inlineBoundaryEndpoint(_ child:GraphNode,link:CanvasInlineContainer.BoundaryLink)->CGPoint {
        return portPosition(child,output:link.direction == .output,port:link.childPort)
    }
    private func inlineBoundaryAt(_ point:CGPoint)->CanvasInlineContainer.BoundaryLink? {
        let index=CanvasNodeIndex(nodes:store.graph.nodes)
        for container in inlineContainers.reversed() where container.isExpanded {
            guard let item=index.node(container.id) else {continue}
            for link in container.boundaryLinks {
                guard let child=index.node(link.childNodeID) else {continue}
                let output=link.direction == .output
                let outer=portPosition(item,output:output,port:link.containerPort)
                let boundary=CGPoint(x:outer.x+(output ? -150:150),y:outer.y)
                let inside=inlineBoundaryEndpoint(child,link:link)
                if curve(from:output ? inside:boundary,to:output ? boundary:inside)
                    .strokedPath(StrokeStyle(lineWidth:max(16,12/zoom))).contains(point) {return link}
            }
        }
        return nil
    }
    private func inlineBoundaryWires(_ container:CanvasInlineContainer,item:GraphNode,index:CanvasNodeIndex)->some View {
        ZStack(alignment:.topLeading) {
            ForEach(Array(container.boundaryLinks.enumerated()),id:\.offset) {_,link in
                if let child=index.node(link.childNodeID) {
                    let output=link.direction == .output
                    let outer=portPosition(item,output:output,port:link.containerPort)
                    let boundary=CGPoint(x:outer.x+(output ? -150:150),y:outer.y)
                    let inside=inlineBoundaryEndpoint(child,link:link)
                    let adapter=store as? WorkflowCanvasAdapter
                    let selected=adapter?.portInspection?.endpoint.address == adapter?.projection.addressByDisplayID[link.childNodeID]
                        && adapter?.portInspection?.endpoint.port == link.childPort && adapter?.portInspection?.direction == link.direction
                    curve(from:output ? inside:boundary,to:output ? boundary:inside)
                        .stroke(selected ? rose:(link.routed ? MLTheme.rose:MLTheme.teal).opacity(0.65),style:StrokeStyle(lineWidth:selected ? 2:1.25,dash:link.routed ? [5,4]:[]))
                        .accessibilityLabel("\(link.routed ? (output ? "Weighted combination":"Routed input") : "Container boundary") · \(link.containerPort) · \(child.name)")
                        .accessibilityAddTraits(.isButton)
                        .accessibilityAction { _=adapter?.inspectPort(nodeID:link.containerNodeID ?? link.childNodeID,port:link.routed ? link.containerPort:link.childPort,direction:link.direction) }
                }
            }
        }.allowsHitTesting(false)
    }

    private func block(_ item: GraphNode,viewport:CGSize) -> some View {
        let definition = geometry.definition(nodeID: item.id)
        let nodeSize = nodeSize(item)
        let container=inlineContainer(item.id)
        let expanded=container?.isExpanded == true
        let selected = store.selectedNodeIDs.contains(item.id)
        let lessonReferenced=lessonNodeFocus.contains(nodeID:item.id,type:store.canvasIconType(nodeID:item.id))
        let family = ComponentVisualFamily.resolve(type:store.canvasIconType(nodeID:item.id))
        let familyTint = appearanceTheme.colorScheme == .light ? family.secondary : family.primary
        let feedback = renderEvidence?.feedback[item.id] ?? store.canvasNodeFeedback(for:item.id)
        let flowStatus = previewStatus(item.id)
        return ZStack(alignment: .topLeading) {
            // Keep one stable gesture/accessibility host even when its distant
            // artwork is omitted. Ports, actions, labels and graph membership remain.
            Color.clear.frame(width:nodeSize.width,height:expanded ? CanvasNodeGeometry.headerHeight:nodeSize.height)
                .overlay {
                    if CanvasCameraGeometry.shouldPaintNode(worldBounds:CGRect(origin:worldPosition(item),size:nodeSize),
                        cameraOrigin:scrollController.paintOrigin,zoom:scrollController.paintZoom,viewport:viewport,reservedTop:toolbarInset,
                        active:movingIDs.contains(item.id) || store.pendingSourceID == item.id,overscan:max(viewport.width,viewport.height)+256) {
                        ZStack(alignment:.topLeading) {
                            if !expanded {
                                CanvasNodeVisual(name:item.name,title:definition?.title ?? item.type,
                                    iconType:store.canvasIconType(nodeID:item.id),nodeSize:nodeSize,selected:selected,
                                    familyTint:familyTint,selectionTint:rose,light:appearanceTheme.colorScheme == .light,
                                    feedback:feedback,flowStatus:flowStatus,rows:tablePortRows(item,definition:definition)).equatable()
                            }
                            if let container {
                                CanvasInlineContainerHeader(title:item.name.isEmpty ? definition?.title ?? item.type:item.name,
                                    typeTitle:definition?.title ?? "Block",iconType:store.canvasIconType(nodeID:item.id),
                                    count:container.childNodeIDs.count,expanded:expanded,tint:familyTint,fit:{(store as? WorkflowCanvasAdapter)?.fitInlineContainer(item.id)}) {
                                        guard !store.isBusy else {return};store.toggleInlineContainer(item.id)
                                    }
                                    .clipShape(UnevenRoundedRectangle(topLeadingRadius:10,topTrailingRadius:10))
                            }
                        }
                        .overlay {
                            if container != nil && !expanded {
                                RoundedRectangle(cornerRadius:10)
                                    .stroke(selected ? rose:familyTint.opacity(appearanceTheme.colorScheme == .light ? 0.9:0.65),
                                            lineWidth:CanvasNodeSurfaceStyle(detail:.medium).borderWidth(isSelected:selected))
                                    .allowsHitTesting(false)
                            }
                        }
                        .overlay {
                            if lessonReferenced {CanvasLessonFocusHalo(light:appearanceTheme.colorScheme == .light)}
                        }

                    }
                }
            .help(feedback.accessibilityDescription)
            .contentShape(Rectangle())
            .onHover {scrollController.input.hover.set("node:"+item.id,hovered:$0)}
            .onDisappear {scrollController.input.hover.set("node:"+item.id,hovered:false)}
            .onTapGesture(count: 2) {
                guard !store.isBusy else {return}
                if container != nil {store.toggleInlineContainer(item.id)}
                else if store.canOpenBlock(item.id) {store.openBlock(item.id)}
            }
            .onTapGesture {
                store.selectNode(item.id, extending: NSEvent.modifierFlags.contains(.shift))
                canvasFocused = true
            }
            .gesture(DragGesture(minimumDistance: 4, coordinateSpace: .named("graphCanvas"))
                .onChanged { value in
                    let profile = InteractionProfiler.shared.begin(.dragUpdate)
                    defer { InteractionProfiler.shared.end(profile) }
                    guard !store.isBusy else { return }
                    if movingID == nil {
                        dragProjection = projection
                        movingID = item.id
                        if !store.selectedNodeIDs.contains(item.id) { store.selectNode(item.id) }
                        movingIDs = store.canvasMovingNodeIDs(for:store.selectedNodeIDs)
                        canvasFocused = true
                    }
                    movement = CanvasInteractionGeometry.translation(modelDelta(value.translation),
                        origins: store.graph.nodes.filter { movingIDs.contains($0.id) }.map { CGPoint(x: $0.x, y: $0.y) })
                    updateInlineDropTarget(at:value.location,moving:movingIDs)
                }
                .onEnded { _ in
                    guard movingID == item.id else { return }
                    store.moveNodes(ids: movingIDs, dx: movement.width, dy: movement.height)
                    movingID = nil; movingIDs = []; movement = .zero; dragProjection = nil;inlineDropTarget=nil
                })
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(item.name), \(definition?.title ?? item.type). \(summary(item)). \(flowStatus). \(feedback.accessibilityDescription)")
            .accessibilityAddTraits(.isButton)
            .accessibilityValue([selected ? "Selected":nil,lessonReferenced ? "Referenced by current lesson step":nil].compactMap {$0}.joined(separator:"; "))
            .accessibilityAction(named: "Select") { store.selectNode(item.id) }
            .accessibilityAction(named: "Toggle selection") { store.selectNode(item.id, extending: true) }
            .accessibilityActions {
                if let container {Button(container.isExpanded ? "Collapse container":"Expand container") {store.toggleInlineContainer(item.id)}}
                else if store.canOpenBlock(item.id) { Button("Open Block") { store.openBlock(item.id) } }
            }
            if expanded,let container {
                CanvasInlineLaneLabels(lanes:container.lanes,tint:familyTint)
                CanvasInlineBoundaryGroup(title:container.hasRoutedBindings ? "Route selected units":"Inputs",ports:definition?.inputs ?? [],output:false,tint:familyTint)
                    .offset(x:0,y:CanvasNodeGeometry.headerHeight+17)
                CanvasInlineBoundaryGroup(title:container.hasRoutedBindings ? "Weighted combination":"Outputs",ports:definition?.outputs ?? [],output:true,tint:familyTint)
                    .offset(x:nodeSize.width-150,y:CanvasNodeGeometry.headerHeight+17)
            }
            if expanded {
                ForEach(tablePortRows(item,definition:definition)) {row in
                    Button {
                        store.selectNode(item.id)
                        _=(store as? WorkflowCanvasAdapter)?.inspectPort(nodeID:item.id,port:row.port,direction:row.output ? .output:.input)
                    } label: {
                        CanvasPortGlyphView(glyph:store.canvasPortDescriptor(nodeID:item.id,port:row.port,direction:row.output ? .output:.input).glyph,tint:MLTheme.teal,filled:false)
                            .frame(width:24,height:24).contentShape(Rectangle())
                    }.buttonStyle(.plain).disabled(store.isBusy)
                        .position(x:row.output ? nodeSize.width-150:150,y:row.anchorY)
                        .accessibilityLabel("Inspect container "+(row.output ? "output ":"input ")+row.port+" internal boundary")
                        .help(store.canvasIconType(nodeID:item.id)=="mixture_experts" ? "MoE tensor boundary. Routing connects this contract to the named expert chains.":"Manage this boundary's bindings in Configuration → Inputs and outputs.")
                }
            }
            ForEach(definition?.inputs ?? [], id: \.self) { port in
                inputPort(item, port: port)
                    .position(portOffset(item,output:false,port:port))
            }
            ForEach(definition?.outputs ?? [], id: \.self) { port in
                outputPort(item, port: port)
                    .position(portOffset(item,output:true,port:port))
            }

        }.frame(width: nodeSize.width, height: nodeSize.height,alignment:.topLeading)
    }

    private func tablePortRows(_ item:GraphNode,definition:ComponentDefinition?)->[CanvasTablePortRow] {
        [(false,definition?.inputs ?? []),(true,definition?.outputs ?? [])].flatMap { output,ports in
            ports.map { port in
                let descriptor=store.canvasPortDescriptor(nodeID:item.id,port:port,direction:output ? .output:.input)
                return CanvasTablePortRow(port:port,output:output,anchorY:portOffset(item,output:output,port:port).y,
                    shortLabel:descriptor.shortLabel,description:descriptor.accessibilityDescription)
            }
        }
    }

    private func summary(_ item: GraphNode) -> String { store.canvasSummary(item) }

    private func portBadge(_ item: GraphNode, port: String, output: Bool) -> String {
        (output ? renderEvidence?.badges[item.id]?[port]:nil)
            ?? (store as? WorkflowCanvasAdapter)?.canvasPortBadge(nodeID:item.id,port:port,direction:output ? .output : .input)
            ?? (port + " · declared " + store.canvasPortDescriptor(nodeID:item.id,port:port,direction:output ? .output : .input).kind.rawValue)
    }

    private func inputPort(_ item: GraphNode, port: String) -> some View {
        let pending = store.pendingSourceID
        let compatible = pending.map { store.canConnect(source: $0, sourcePort: pendingOutputPort, target: item, port: port) } ?? false
        return Button {
            guard !store.isBusy else { return }
            guard let source = store.pendingSourceID else { store.selectNode(item.id); return }
            store.connect(source: source, sourcePort: pendingOutputPort, target: item.id, targetPort: port)
            wirePointer = nil
        } label: {
            CanvasPortGlyphView(glyph:store.canvasPortDescriptor(nodeID:item.id,port:port,direction:.input).glyph,tint:pending == nil ? MLTheme.muted : (compatible ? rose : .orange),filled:compatible)
                .overlay {
                    if pending != nil {
                        Image(systemName: compatible ? "checkmark" : "xmark")
                            .font(.system(size: 7, weight: .bold))
                            .foregroundStyle(compatible ? ground : .orange)
                    }
                }
                .frame(width: 24, height: 24)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain).disabled(store.isBusy)
        .onHover {scrollController.input.hover.set("input:"+item.id+":"+port,hovered:$0)}
        .onDisappear {scrollController.input.hover.set("input:"+item.id+":"+port,hovered:false)}
        .help("\(item.name) · input \(port). \(store.canvasPortDescriptor(nodeID:item.id,port:port,direction:.input).accessibilityDescription). Declared input contract; use \(showsPortControls ? "Ports" : "Results → Ports & connections") → Inspect for linked captured source output.")
        .accessibilityLabel("\(item.name), \(port), \(store.canvasPortDescriptor(nodeID:item.id,port:port,direction:.input).accessibilityDescription)")
        .accessibilityValue(pending == nil ? "" : (compatible ? "Compatible" : "Connection constraints require review"))
        .accessibilityAction(named: "Inspect input") { _ = (store as? WorkflowCanvasAdapter)?.inspectPort(nodeID:item.id,port:port,direction:.input) }
    }

    private func outputPort(_ item: GraphNode, port: String) -> some View {
        CanvasPortGlyphView(glyph:store.canvasPortDescriptor(nodeID:item.id,port:port,direction:.output).glyph,tint:MLTheme.teal,filled:store.pendingSourceID == item.id && pendingOutputPort == port)
            .frame(width:24,height:24)
            .contentShape(Rectangle())
            .accessibilityElement(children:.ignore)
            .accessibilityHidden(false)
            .onTapGesture {
                guard !store.isBusy else { return }
                let wasActive = store.pendingSourceID == item.id && pendingOutputPort == port
                pendingOutputPort = port
                store.pendingSourceID = wasActive ? nil : item.id
                wirePointer = nil
            }
            .gesture(DragGesture(minimumDistance: 3, coordinateSpace: .named("graphCanvas"))
                .onChanged { value in guard !store.isBusy else { return };pendingOutputPort = port; if store.pendingSourceID != item.id { store.pendingSourceID = item.id }; wirePointer = value.location }
                .onEnded { value in
                    guard !store.isBusy else { wirePointer=nil;return }
                    let targets = store.graph.nodes.filter { $0.id != item.id }.flatMap { target in
                        (geometry.definition(nodeID:target.id)?.inputs ?? []).map { (target,$0) }
                    }
                    let anchors = targets.map { portPosition($0.0,output:false,port:$0.1) }
                    if let index = CanvasInteractionGeometry.portTarget(point:value.location,anchors:anchors,zoom:zoom) {
                        store.connect(source:item.id,sourcePort:port,target:targets[index].0.id,targetPort:targets[index].1)
                    } else {
                        store.statusMessage = "Choose a distinct input port, or use \(showsPortControls ? "Ports" : "Results → Ports & connections") to select an exact endpoint."
                    }
                    wirePointer = nil
                })
            .onHover {scrollController.input.hover.set("output:"+item.id+":"+port,hovered:$0)}
            .onDisappear {scrollController.input.hover.set("output:"+item.id+":"+port,hovered:false)}
            .help("\(item.name) · output \(port). \(portBadge(item,port:port,output:true)). Click then choose an input, or drag a connection. Use \(showsPortControls ? "Ports" : "Results → Ports & connections") → Inspect for this exact output.")
            .accessibilityLabel("\(item.name), \(port), \(store.canvasPortDescriptor(nodeID:item.id,port:port,direction:.output).accessibilityDescription), start connection")
            .accessibilityAddTraits(.isButton)
            .accessibilityAction(named: "Inspect output") { _ = (store as? WorkflowCanvasAdapter)?.inspectPort(nodeID:item.id,port:port,direction:.output) }
            .accessibilityAction { guard !store.isBusy else { return };pendingOutputPort = port; store.pendingSourceID = item.id }
    }


}

private struct CanvasToolbarHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat,nextValue:()->CGFloat) { value=max(value,nextValue()) }
}



/// Invalid drafts remain inspectable even when imported numeric configuration is extreme.
enum CanvasDisplayValue {
    static func dimension(_ value: Double?) -> String {
        guard let value else { return "?" }
        guard value.isFinite else { return "Invalid" }
        if let integer = Int(exactly: value) { return String(integer) }
        return String(format: "%.4g", value)
    }
}

/// Presentation camera preserves SwiftUI's native clip and its gesture bookkeeping.
/// The native surface stays at zero; SwiftUI applies the logical camera transform.
/// Frequent camera translation invalidates this tiny modifier, not the graph's
/// layout/gesture tree. Coarse paint-origin changes refresh the overscanned scene.
@MainActor final class CanvasCameraTranslationState:ObservableObject {
    @Published var origin=CGPoint.zero
    @Published var zoom=0.65
}
private struct CanvasCameraSurface:ViewModifier {
    @ObservedObject var camera:CanvasCameraTranslationState
    let controller:CanvasScrollController
    let sceneOrigin:CGPoint
    let contentSize:CGSize
    let viewport:CGSize
    let reservedTop:CGFloat
    func body(content:Content)->some View {
        let zoom=camera.zoom
        let document=CanvasViewportGeometry.documentSize(content:contentSize,zoom:zoom,reservedTop:reservedTop,viewport:viewport)
        return content.padding(.top,reservedTop/zoom)
            .scaleEffect(zoom,anchor:.topLeading)
            .frame(width:document.width,height:document.height,alignment:.topLeading)
            .background(CanvasScrollCapture(controller:controller).allowsHitTesting(false))
            .offset(x:sceneOrigin.x*zoom-camera.origin.x,y:sceneOrigin.y*zoom-camera.origin.y)
    }
}
private struct CanvasCameraTracking<Content:View>:View {
    @ObservedObject var camera:CanvasCameraTranslationState
    let content:(CGPoint)->Content
    init(camera:CanvasCameraTranslationState,@ViewBuilder content:@escaping(CGPoint)->Content) {self.camera=camera;self.content=content}
    var body:some View {content(camera.origin)}
}

@MainActor final class CanvasScrollController: ObservableObject {
    weak var scrollView: NSScrollView?
    let objectWillChange=ObservableObjectPublisher()
    let translation=CanvasCameraTranslationState()
    var logicalOrigin:CGPoint {translation.origin}
    private(set) var paintOrigin=CGPoint.zero
    private(set) var paintZoom=0.65
    var origin: CGPoint? { scrollView == nil ? nil:logicalOrigin }
    var onOriginChanged: ((CGPoint) -> Void)?
    let input = CanvasInputMonitor()
    var gridScale: CGFloat = 1 { didSet { updateGrid() } }
    var reservedTop: CGFloat = 60 { didSet { updateGrid() } }
    var gridBackground: NSColor = .windowBackgroundColor { didSet { updateGrid() } }
    var gridColor: NSColor = .separatorColor { didSet { updateGrid() } }
    private func updateGrid() {
        guard let clip=scrollView?.contentView as? CanvasInfiniteClipView else {return}
        clip.gridScale=gridScale;clip.reservedTop=reservedTop;clip.backgroundColor=gridBackground;clip.gridColor=gridColor
    }
    private var observer: NSObjectProtocol?
    private var pendingRestore: (CGPoint, CGSize)?
    var hasPendingRestore: Bool { pendingRestore != nil }
    deinit { if let observer { NotificationCenter.default.removeObserver(observer) } }

    func requestRestore(to point: CGPoint, document: CGSize) {
        pendingRestore = (point, document)
        DispatchQueue.main.async { [weak self] in self?.restoreWhenReady() }
    }
    func retargetPendingRestore(document: CGSize) {
        guard let (point, _) = pendingRestore else { return }
        pendingRestore = (point,document)
        DispatchQueue.main.async { [weak self] in self?.restoreWhenReady() }
    }
    func restoreWhenReady() {
        guard let (point, expected) = pendingRestore, let scrollView, let document = scrollView.documentView else { return }
        let clip = scrollView.contentView
        guard clip.bounds.width > 0, clip.bounds.height > 0,
              abs(document.bounds.width - max(expected.width, clip.bounds.width)) < 2,
              abs(document.bounds.height - max(expected.height, clip.bounds.height)) < 2 else { return }
        pendingRestore = nil
        navigate(to: point)
        if let origin { onOriginChanged?(origin) }
    }
    private func publishOrigin() {
        resetNativeScroll()
    }
    private func resetNativeScroll() {
        guard let scroll=scrollView,scroll.contentView.bounds.origin != .zero else {return}
        scroll.contentView.scroll(to:.zero)
        scroll.reflectScrolledClipView(scroll.contentView)
    }

    func attach(_ scroll: NSScrollView) {
        if scrollView !== scroll {
            if let observer { NotificationCenter.default.removeObserver(observer) }
            scrollView = scroll
            // Replacing this clip leaves SwiftUI gesture coordinates attached to
            // the old surface. Keep it installed and transform the graph instead.
            scroll.drawsBackground=false;scroll.contentView.drawsBackground=false
            scroll.horizontalScrollElasticity = .none;scroll.verticalScrollElasticity = .none
            resetNativeScroll()
            updateGrid()
            scroll.contentView.postsBoundsChangedNotifications = true
            observer = NotificationCenter.default.addObserver(forName: NSView.boundsDidChangeNotification, object: scroll.contentView, queue: .main) { [weak self] _ in
                Task { @MainActor [weak self] in self?.publishOrigin() }
            }
        }
        restoreWhenReady()
        // Explicit native policy overrides macOS “Always show scroll bars”. The clip
        // view remains scrollable by wheel, trackpad and our background pan gesture.
        if scroll.scrollerStyle != .overlay { scroll.scrollerStyle = .overlay }
        if scroll.hasHorizontalScroller { scroll.hasHorizontalScroller = false }
        if scroll.hasVerticalScroller { scroll.hasVerticalScroller = false }
        scroll.autohidesScrollers = true
        scroll.horizontalScroller?.isHidden = true
        scroll.verticalScroller?.isHidden = true
    }

    func setZoom(_ next:Double) {
        guard next.isFinite,CanvasZoomLimits.supported.contains(next),translation.zoom != next else {return}
        translation.zoom=next
        gridScale=next
        if max(next/paintZoom,paintZoom/next)>1.5 {
            objectWillChange.send()
            paintZoom=next;paintOrigin=logicalOrigin
        }
    }
    func navigate(to point: CGPoint) {
        guard scrollView?.documentView != nil else {return}
        pendingRestore=nil
        let target=CanvasCameraGeometry.bounded(point)
        if logicalOrigin != target {
            translation.origin=target
            let size=scrollView?.contentView.bounds.size ?? .zero
            let threshold=max(64,min(size.width,size.height)/2)
            if abs(target.x-paintOrigin.x)>threshold || abs(target.y-paintOrigin.y)>threshold {
                objectWillChange.send()
                paintOrigin=target;paintZoom=translation.zoom
            }
        }
        resetNativeScroll()
        onOriginChanged?(target)
    }

    func pan(from origin: CGPoint, translation: CGSize) {
        navigate(to:CanvasCameraGeometry.bounded(CGPoint(x:origin.x-translation.width,y:origin.y-translation.height)))
    }

}

private struct CanvasScrollCapture: NSViewRepresentable {
    let controller: CanvasScrollController
    func makeNSView(context: Context) -> CanvasScrollProbe {
        let view = CanvasScrollProbe()
        view.controller = controller
        return view
    }
    func updateNSView(_ view: CanvasScrollProbe, context: Context) {
        view.controller = controller
        view.capture()
    }
}

// Internal so the opt-in profiler can exercise the installed camera bridge.
final class CanvasScrollProbe: NSView {
    weak var controller: CanvasScrollController?
    override func hitTest(_ point: NSPoint) -> NSView? { nil }
    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); capture() }
    override func viewDidMoveToSuperview() { super.viewDidMoveToSuperview(); capture() }
    override func layout() { super.layout(); capture() }
    func capture() {
        DispatchQueue.main.async { [weak self] in
            guard let self, let scroll = self.enclosingScrollView else { return }
            self.controller?.attach(scroll)
        }
    }
}

/// Pure screen-point panning math, independent of the graph's render scale.
enum CanvasPanGeometry {
    static func origin(start: CGPoint, translation: CGSize, document: CGSize, viewport: CGSize, flipped: Bool) -> CGPoint {
        CGPoint(x: max(0, min(max(0, document.width - viewport.width), start.x - translation.width)),
                y: max(0, min(max(0, document.height - viewport.height), start.y + (flipped ? -translation.height : translation.height))))
    }
}

private struct CanvasInsertion {
    let type: String
    let edgeID: String
    let location: CGPoint
}

private struct InitialCanvasLayout: Equatable {
    let size: CGSize
    let toolbarInset: CGFloat
}
