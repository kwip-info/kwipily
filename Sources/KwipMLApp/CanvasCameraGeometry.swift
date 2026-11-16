import AppKit
import KwipMLCore

enum CanvasCameraGeometry {
    static func shouldPaintNode(worldBounds:CGRect,cameraOrigin:CGPoint,zoom:Double,viewport:CGSize,reservedTop:CGFloat,active:Bool,overscan:CGFloat?=nil)->Bool {
        guard !active,viewport.width > 0,viewport.height > 0,zoom.isFinite,zoom > 0 else {return true}
        let screen=CGRect(x:worldBounds.minX*zoom-cameraOrigin.x,
            y:worldBounds.minY*zoom-cameraOrigin.y+reservedTop,
            width:worldBounds.width*zoom,height:worldBounds.height*zoom)
        // Paint ahead of the viewport and include border/status-badge overflow.
        // The complete rectangle matters for tall nodes with many dynamic ports.
        let overscan=overscan ?? (max(160,256*zoom)+16)
        return CGRect(origin:.zero,size:viewport).insetBy(dx:-overscan,dy:-overscan).intersects(screen)
    }

    static func gridStep(zoom: Double) -> CGFloat {
        let base=24*max(0.1,zoom)
        return 24 * max(1,ceil(12/base))
    }
    /// Both the finite graph and the surrounding camera paint the same grid phase.
    /// Construct only viewport-spanning lines, never a path for the full document.
    static func gridPath(in rect:CGRect, spacing:CGFloat, offset:CGPoint)->CGPath {
        let path=CGMutablePath()
        guard !rect.isEmpty,!rect.isNull,spacing.isFinite,spacing > 0 else {return path}
        for x in stride(from:floor((rect.minX-offset.x)/spacing)*spacing+offset.x,through:rect.maxX,by:spacing) {
            path.move(to:CGPoint(x:x,y:rect.minY));path.addLine(to:CGPoint(x:x,y:rect.maxY))
        }
        for y in stride(from:floor((rect.minY-offset.y)/spacing)*spacing+offset.y,through:rect.maxY,by:spacing) {
            path.move(to:CGPoint(x:rect.minX,y:y));path.addLine(to:CGPoint(x:rect.maxX,y:y))
        }
        return path
    }
    static func topLeading(_ point:CGPoint,bounds:CGRect,flipped:Bool)->CGPoint {
        CGPoint(x:point.x-bounds.minX,y:flipped ? point.y-bounds.minY:bounds.maxY-point.y)
    }
    static func viewportPoint(windowPoint:CGPoint,window:NSWindow,viewportFrame:CGRect,clip:NSClipView)->CGPoint {
        // SwiftUI's .global origin includes the native title bar. Content bounds
        // omit it, which displaced pointer hits by 28–32 points on titled windows.
        let global=CGPoint(x:windowPoint.x,y:window.frame.height-windowPoint.y)
        return CGPoint(x:global.x-viewportFrame.minX,y:global.y-viewportFrame.minY)
    }
    static func contentFrame(viewport:CGRect,sceneOrigin:CGPoint,cameraOrigin:CGPoint,zoom:Double,size:CGSize)->CGRect {
        CGRect(origin:CGPoint(x:viewport.minX+sceneOrigin.x*zoom-cameraOrigin.x,
                              y:viewport.minY+sceneOrigin.y*zoom-cameraOrigin.y),size:size)
    }
    static func renderedGraphPoint(viewportPoint:CGPoint,viewportFrame:CGRect,contentFrame:CGRect,zoom:Double,reservedTop:CGFloat)->CGPoint {
        CGPoint(x:(viewportFrame.minX+viewportPoint.x-contentFrame.minX)/zoom,
            y:(viewportFrame.minY+viewportPoint.y-contentFrame.minY-reservedTop)/zoom)
    }
    static func bounded(_ point: CGPoint) -> CGPoint {
        CGPoint(x: max(-1_000_000,min(1_000_000,point.x)), y:max(-1_000_000,min(1_000_000,point.y)))
    }
    static func zoom(origin: CGPoint, zoom: Double, pointer: CGPoint, delta: CGFloat, precise: Bool, reservedTop: CGFloat, sensitivity:Double = 1) -> (zoom: Double, origin: CGPoint) {
        guard zoom.isFinite, zoom > 0, delta.isFinite else { return (zoom,origin) }
        // Wheel drivers differ substantially in packet magnitude. Bound the
        // logarithmic step, preserving direction and the pointer anchor.
        let speed=sensitivity.isFinite ? max(0.25,min(2,sensitivity)):1
        let limit=(precise ? 0.06:0.12)*speed
        let step=max(-limit,min(limit,Double(delta)*(precise ? 0.003:0.04)*speed))
        let next = max(CanvasZoomLimits.minimum,min(CanvasZoomLimits.maximum,zoom * exp(step)))
        let ratio = next / zoom
        return (next,bounded(CGPoint(x:(origin.x+pointer.x)*ratio-pointer.x,
            y:(origin.y+pointer.y-reservedTop)*ratio-pointer.y+reservedTop)))
    }
    enum WheelAction: Equatable {
        case none, zoom(CGFloat), pan(CGFloat)
    }
    static func wheelAction(deltaX:CGFloat,deltaY:CGFloat,precise:Bool,momentum:Bool)->WheelAction {
        guard deltaX.isFinite,deltaY.isFinite else {return .none}
        // Horizontal gestures belong to navigation; ignore incidental vertical
        // drift. Momentum can continue panning but never keeps changing scale.
        if abs(deltaX) > abs(deltaY)*1.2 {
            return .pan(max(-120,min(120,deltaX*(precise ? 1:12))))
        }
        guard !momentum,deltaY != 0 else {return .none}
        return .zoom(deltaY)
    }
    static func reanchored(origin:CGPoint,oldAnchor:CGPoint,newAnchor:CGPoint,zoom:Double)->CGPoint {
        bounded(CGPoint(x:origin.x+(newAnchor.x-oldAnchor.x)*zoom,y:origin.y+(newAnchor.y-oldAnchor.y)*zoom))
    }
    static func rectangle(start: CGPoint, end: CGPoint) -> CGRect {
        CGRect(x:min(start.x,end.x),y:min(start.y,end.y),width:abs(end.x-start.x),height:abs(end.y-start.y))
    }
    static func selection(start: CGPoint, end: CGPoint, bounds: [String:CGRect], initial: Set<String>, extending: Bool) -> Set<String> {
        let rect = rectangle(start:start,end:end)
        let hits = Set(bounds.compactMap { rect.intersects($0.value) ? $0.key : nil })
        return extending ? initial.union(hits) : hits
    }
}

/// Camera movement does not resize the document or change any authored coordinates.
/// Only the visible native clip is painted, including blank space beyond the document.
final class CanvasInfiniteClipView: NSClipView {
    var gridScale: CGFloat = 1 { didSet { needsDisplay=true } }
    var reservedTop: CGFloat = 60 { didSet { needsDisplay=true } }
    var gridColor: NSColor = .separatorColor { didSet { needsDisplay=true } }
    override func constrainBoundsRect(_ proposedBounds: NSRect) -> NSRect {
        CGRect(origin:CanvasCameraGeometry.bounded(proposedBounds.origin),size:proposedBounds.size)
    }
    override func draw(_ dirtyRect: NSRect) {
        backgroundColor.setFill(); dirtyRect.fill()

    }
}
