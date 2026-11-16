import AppKit
import KwipMLCore

/// A window-scoped event bridge. Native text editing and controls outside the clip
/// retain their normal scroll/key behavior. Pan owns its gesture through mouse-up.
@MainActor final class CanvasInputMonitor {
    weak var controller: CanvasScrollController?
    var reservedTop: CGFloat = 60
    var viewportPoint: ((CGPoint,NSWindow)->CGPoint?)?
    let hover = CanvasHoverOwnership()
    var keyboardPanAllowed: (() -> Bool)?
    var onCancel: (() -> Void)?
    var onZoom: ((CGPoint, CGFloat, Bool) -> Void)?
    var onConnectionPress: ((CGPoint)->Bool)?
    var canMarquee: ((CGPoint) -> Bool)?
    var onMarquee: ((CGPoint, CGPoint, Bool, Bool) -> Void)?
    private var monitor: Any?
    private var resignObserver: NSObjectProtocol?
    private var spaceHeld = false
    private var panStart: (point: CGPoint, origin: CGPoint, drag: NSEvent.EventType, up: NSEvent.EventType)?
    private var marqueeStart: CGPoint?
    private var extending = false
    private var ownedConnection = false
    deinit { if let monitor { NSEvent.removeMonitor(monitor) };if let resignObserver {NotificationCenter.default.removeObserver(resignObserver)} }
    func install() {
        guard monitor == nil else {return}
        resignObserver=NotificationCenter.default.addObserver(forName:NSWindow.didResignKeyNotification,object:nil,queue:.main) { [weak self] notification in
            MainActor.assumeIsolated {
                guard let self, let window=notification.object as? NSWindow,window === self.controller?.scrollView?.window else {return}
                self.cancelGesture()
            }
        }
        monitor=NSEvent.addLocalMonitorForEvents(matching:[.scrollWheel,.keyDown,.keyUp,.otherMouseDown,.otherMouseDragged,.otherMouseUp,.leftMouseDown,.leftMouseDragged,.leftMouseUp,.rightMouseDown,.rightMouseDragged,.rightMouseUp]) { [weak self] event in
            guard let self else {return event}
            return self.routedEvent(event)
        }
    }
    func stop() {
        if let monitor { NSEvent.removeMonitor(monitor) }; monitor=nil
        if let resignObserver {NotificationCenter.default.removeObserver(resignObserver)};resignObserver=nil
        cancelGesture();hover.clear()
    }
    func routedEvent(_ event:NSEvent)->NSEvent? {handle(event)}
    /// Return true only when the canvas owns this wheel packet. The caller's
    /// viewport gate keeps ribbon and inspector scrolling native.
    func routeWheel(point:CGPoint,inside:Bool,deltaX:CGFloat,deltaY:CGFloat,precise:Bool,momentum:Bool)->Bool {
        let profile = InteractionProfiler.shared.begin(.wheelHandler)
        defer { InteractionProfiler.shared.end(profile) }
        guard inside,panStart == nil,marqueeStart == nil else {return false}
        switch CanvasCameraGeometry.wheelAction(deltaX:deltaX,deltaY:deltaY,precise:precise,momentum:momentum) {
        case .pan(let delta):
            guard let origin=controller?.origin else {return false}
            controller?.navigate(to:CanvasCameraGeometry.bounded(CGPoint(x:origin.x-delta,y:origin.y)))
            return true
        case .zoom(let delta):
            guard let onZoom else {return false}
            onZoom(point,delta,precise);return true
        case .none:
            // Suppress a vertical momentum tail inside the canvas so its
            // underlying native clip cannot move away from the logical camera.
            return momentum
        }
    }
    private func cancelGesture() {
        if panStart != nil {NSCursor.pop()};panStart=nil
        if marqueeStart != nil {onCancel?()};marqueeStart=nil;spaceHeld=false;ownedConnection=false
    }
    func handle(_ event:NSEvent)->NSEvent? {
        if event.type == .keyUp,event.keyCode == 49,spaceHeld {spaceHeld=false;return nil}
        guard let scroll=controller?.scrollView, let window=scroll.window, event.window === window else {return event}
        // Retain ownership outside the viewport and after Control/Space is released.
        if let start=panStart {
            if event.type == .keyDown,event.keyCode == 53 {cancelGesture();return nil}
            if event.type == start.drag {
                controller?.navigate(to:CanvasCameraGeometry.bounded(CGPoint(x:start.origin.x-(event.locationInWindow.x-start.point.x),y:start.origin.y+(event.locationInWindow.y-start.point.y))))
                return nil
            }
            if event.type == start.up {panStart=nil;NSCursor.pop();return nil}
            // A different button cannot replace or terminate the active pan.
            return event
        }
        let eventPoint=(event.type == .keyDown || event.type == .keyUp) ? window.mouseLocationOutsideOfEventStream:event.locationInWindow
        let bounds=scroll.contentView.bounds
        let point:CGPoint
        if let viewportPoint {
            guard let converted=viewportPoint(eventPoint,window) else {return event}
            point=converted
        } else {
            let native=scroll.contentView.convert(eventPoint,from:nil)
            point=CanvasCameraGeometry.topLeading(native,bounds:bounds,flipped:scroll.contentView.isFlipped)
        }
        let inside=point.x >= 0 && point.y >= reservedTop && point.x < bounds.width && point.y < bounds.height
        let editing=(window.firstResponder as? NSTextView)?.isEditable == true
        if event.type == .keyDown || event.type == .keyUp {
            if event.keyCode == 53,(panStart != nil || marqueeStart != nil || ownedConnection) {cancelGesture();return nil}
            guard event.keyCode == 49,!editing else {return event}
            if event.type == .keyUp {let owned=spaceHeld;spaceHeld=false;return owned ? nil:event}
            guard !(window.firstResponder is NSControl),keyboardPanAllowed?() == true,inside,event.modifierFlags.intersection([.command,.option,.control]).isEmpty else {return event}
            spaceHeld=true;return nil
        }
        if event.type == .scrollWheel {
            return routeWheel(point:point,inside:inside,deltaX:event.scrollingDeltaX,deltaY:event.scrollingDeltaY,
                precise:event.hasPreciseScrollingDeltas,momentum:!event.momentumPhase.isEmpty) ? nil:event
        }
        let secondaryPan=event.type == .rightMouseDown || (event.type == .leftMouseDown && event.modifierFlags.contains(.control))
        let middlePan=event.type == .otherMouseDown && event.buttonNumber == 2
        if secondaryPan || middlePan || (event.type == .leftMouseDown && spaceHeld) {
            guard inside,!hover.isCanvasChrome,marqueeStart == nil,!ownedConnection,let origin=controller?.origin else {return event}
            let drag:NSEvent.EventType=event.type == .rightMouseDown ? .rightMouseDragged : (middlePan ? .otherMouseDragged:.leftMouseDragged)
            let up:NSEvent.EventType=event.type == .rightMouseDown ? .rightMouseUp : (middlePan ? .otherMouseUp:.leftMouseUp)
            panStart=(event.locationInWindow,origin,drag,up);NSCursor.closedHand.push();return nil
        }
        if ownedConnection {
            if event.type == .leftMouseDragged {return nil}
            if event.type == .leftMouseUp {ownedConnection=false;return nil}
        }
        if event.type == .leftMouseDown,inside,onConnectionPress?(point) == true {
            ownedConnection=true;return nil
        }
        if event.type == .leftMouseDown,inside,!hover.isOccupied,canMarquee?(point) == true {
            marqueeStart=point;extending=event.modifierFlags.contains(.shift) || event.modifierFlags.contains(.command)
            onMarquee?(point,point,extending,false);return nil
        }
        if let start=marqueeStart {
            if event.type == .leftMouseDragged {onMarquee?(start,point,extending,false);return nil}
            if event.type == .leftMouseUp {onMarquee?(start,point,extending,true);marqueeStart=nil;return nil}
        }
        return event
    }
}

/// Hover updates native gesture ownership without publishing SwiftUI state.
/// Distinct tokens preserve a port nested in a node when one region exits first.
final class CanvasHoverOwnership {
    private var tokens=Set<String>()
    var isOccupied:Bool {!tokens.isEmpty}
    var isCanvasChrome:Bool {tokens.contains("toolbar") || tokens.contains("overview")}
    func set(_ token:String,hovered:Bool) {
        if hovered {tokens.insert(token)} else {tokens.remove(token)}
    }
    func clear() {tokens.removeAll()}
}
