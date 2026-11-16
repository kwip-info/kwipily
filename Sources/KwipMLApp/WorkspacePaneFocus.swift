import SwiftUI
import AppKit

enum WorkspacePane:String {case inspector="Configuration",canvas="Canvas",results="Results",lesson="Lesson"}

/// Passive window-local observation. Never consumes a click, changes the responder
/// or participates in authoring/selection. Only the small border views observe it.
@MainActor final class WorkspacePaneFocus:ObservableObject {
    @Published private(set) var active:WorkspacePane = .canvas
    @Published private(set) var windowActive=false
    private struct Registration {weak var view:NSView?;let pane:WorkspacePane}
    private var views:[UUID:Registration]=[:]
    private weak var window:NSWindow?
    private var monitor:Any?
    private var notifications:[NSObjectProtocol]=[]
    func activate(_ pane:WorkspacePane) {if active != pane {active=pane}}
    func register(_ id:UUID,view:NSView,pane:WorkspacePane) {
        views[id]=Registration(view:view,pane:pane)
        guard let window=view.window,self.window !== window else {return}
        detach();self.window=window;windowActive=window.isKeyWindow
        monitor=NSEvent.addLocalMonitorForEvents(matching:[.leftMouseDown,.rightMouseDown,.otherMouseDown,.scrollWheel,.keyDown]) { [weak self] event in
            self?.observe(event);return event
        }
        notifications=[NSWindow.didBecomeKeyNotification,NSWindow.didResignKeyNotification].map {name in
            NotificationCenter.default.addObserver(forName:name,object:window,queue:.main) { [weak self,weak window] _ in
                guard let self,let window else {return};self.windowActive=window.isKeyWindow
            }
        }
    }
    func unregister(_ id:UUID) {views.removeValue(forKey:id);if views.isEmpty {detach();window=nil;windowActive=false}}
    func pane(at point:NSPoint)->WorkspacePane? {
        views.values.first {registration in
            guard let view=registration.view,view.window===window,!view.isHiddenOrHasHiddenAncestor else {return false}
            return view.bounds.contains(view.convert(point,from:nil))
        }?.pane
    }
    func observe(_ event:NSEvent) {
        guard let window,event.window===window else {return}
        if event.type == .keyDown {
            DispatchQueue.main.async { [weak self,weak window] in
                guard let self,let window,let responder=window.firstResponder as? NSView else {return}
                let target=(responder as? NSTextView)?.delegate as? NSView ?? responder
                guard target !== window.contentView else {return}
                if let pane=self.pane(at:target.convert(NSPoint(x:target.bounds.midX,y:target.bounds.midY),to:nil)) {self.activate(pane)}
            }
        } else if let pane=pane(at:event.locationInWindow) {activate(pane)}
    }
    private func detach() {
        if let monitor {NSEvent.removeMonitor(monitor)};monitor=nil
        for token in notifications {NotificationCenter.default.removeObserver(token)};notifications=[]
    }
    deinit {if let monitor {NSEvent.removeMonitor(monitor)};for token in notifications {NotificationCenter.default.removeObserver(token)}}
}
private struct WorkspacePaneFocusKey:EnvironmentKey {static let defaultValue:WorkspacePaneFocus?=nil}
extension EnvironmentValues {
    var workspacePaneFocus:WorkspacePaneFocus? {get {self[WorkspacePaneFocusKey.self]} set {self[WorkspacePaneFocusKey.self]=newValue}}
}
private struct WorkspacePaneBorder:View {
    @ObservedObject var focus:WorkspacePaneFocus
    @Environment(\.appearanceTheme) private var theme
    let pane:WorkspacePane
    var body:some View {
        let active=focus.active==pane && focus.windowActive
        Rectangle().strokeBorder(active ? theme.accentColor.opacity(0.85):Color.clear,lineWidth:1.5)
            .overlay(alignment:.top) {if active {Rectangle().fill(theme.accentColor).frame(height:2)}}
            .accessibilityLabel("\(pane.rawValue) panel").accessibilityValue(active ? "Active":"Inactive")
            .allowsHitTesting(false)
    }
}
private struct WorkspacePaneProbe:NSViewRepresentable {
    let focus:WorkspacePaneFocus,pane:WorkspacePane
    func makeNSView(context:Context)->Probe {Probe()}
    func updateNSView(_ view:Probe,context:Context) {view.focus=focus;view.pane=pane;view.register()}
    static func dismantleNSView(_ view:Probe,coordinator:()) {view.focus?.unregister(view.id);view.focus=nil}
    final class Probe:NSView {
        let id=UUID()
        weak var focus:WorkspacePaneFocus?
        var pane:WorkspacePane = .canvas
        override func hitTest(_ point:NSPoint)->NSView? {nil}
        override func viewDidMoveToWindow() {super.viewDidMoveToWindow();register()}
        func register() {if window != nil {focus?.register(id,view:self,pane:pane)} else {focus?.unregister(id)}}
    }
}
private struct WorkspacePaneModifier:ViewModifier {
    @Environment(\.workspacePaneFocus) private var focus
    let pane:WorkspacePane
    @ViewBuilder func body(content:Content)->some View {
        if let focus {content.background(WorkspacePaneProbe(focus:focus,pane:pane)).overlay {WorkspacePaneBorder(focus:focus,pane:pane)}}
        else {content}
    }
}
private struct WindowActivatingControls:ViewModifier {
    @ViewBuilder func body(content:Content)->some View {
        if #available(macOS 15.0,*) {content.allowsWindowActivationEvents()}
        else {content}
    }
}
extension View {
    func workspacePane(_ pane:WorkspacePane)->some View {modifier(WorkspacePaneModifier(pane:pane))}
    func windowActivatingControls()->some View {modifier(WindowActivatingControls())}
}
