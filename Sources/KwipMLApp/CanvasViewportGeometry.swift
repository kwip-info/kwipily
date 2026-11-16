import CoreGraphics
import KwipMLCore

/// Toolbar reservation is in screen points; graph coordinates remain independent of it.
enum CanvasViewportGeometry {
    /// Minimal screen-space pan. Large targets align their leading edge; zoom is unchanged.
    static func revealOrigin(bounds:CGRect,zoom:Double,origin:CGPoint,viewport:CGSize,reservedTop:CGFloat)->CGPoint {
        func axis(_ lower:CGFloat,_ upper:CGFloat,_ current:CGFloat,_ extent:CGFloat)->CGFloat {
            let space=max(0,extent-24)
            if upper-lower > space || lower < current+12 {return lower-12}
            if upper > current+extent-12 {return upper-extent+12}
            return current
        }
        return CGPoint(x:axis(bounds.minX*zoom,bounds.maxX*zoom,origin.x,viewport.width),
                       y:axis(bounds.minY*zoom,bounds.maxY*zoom,origin.y,max(0,viewport.height-reservedTop)))
    }
    static func topInset(toolbarHeight: CGFloat) -> CGFloat { max(60,toolbarHeight+24) }
    static func documentSize(content: CGSize, zoom: Double, reservedTop: CGFloat, viewport: CGSize = .zero) -> CGSize {
        CGSize(width:max(viewport.width,content.width*zoom),height:max(viewport.height,content.height*zoom+reservedTop))
    }
    static func graphExtent(content: CGSize, zoom: Double, reservedTop: CGFloat, viewport: CGSize) -> CGSize {
        let size = documentSize(content:content,zoom:zoom,reservedTop:reservedTop,viewport:viewport)
        return CGSize(width:size.width/zoom,height:max(0,size.height-reservedTop)/zoom)
    }
    static func graphLocation(documentPoint: CGPoint, zoom: Double, reservedTop: CGFloat) -> CGPoint {
        CGPoint(x:documentPoint.x/zoom,y:(documentPoint.y-reservedTop)/zoom)
    }
    static func visibleGraphRect(scrollOrigin: CGPoint, viewport: CGSize, zoom: Double, reservedTop: CGFloat) -> CGRect {
        CGRect(x:scrollOrigin.x/zoom,y:scrollOrigin.y/zoom,width:max(0,viewport.width)/zoom,height:max(0,viewport.height-reservedTop)/zoom)
    }
    static func fitZoom(bounds: CGRect, viewport: CGSize, reservedTop: CGFloat, maximum: Double) -> Double {
        max(CanvasZoomLimits.minimum,min(CanvasZoomLimits.maximum,maximum,max(1,viewport.width-24)/max(1,bounds.width),max(1,viewport.height-reservedTop-24)/max(1,bounds.height)))
    }
    static func fitAllViewport(nodes:[CGRect],viewport:CGSize,reservedTop:CGFloat)->(zoom:Double,origin:CGPoint,fullyFits:Bool) {
        guard !nodes.isEmpty else {return (1,.zero,true)}
        let bounds=nodes.reduce(CGRect.null) {$0.union($1)}
        let zoom=fitZoom(bounds:bounds,viewport:viewport,reservedTop:reservedTop,maximum:1)
        // Fit content, not the empty space between authored nodes and the document origin.
        // Screen-space margins remain constant at every zoom; topology and layout are untouched.
        let origin=CGPoint(x:bounds.minX*zoom-12,y:bounds.minY*zoom-12)
        let fullyFits = bounds.width*zoom <= max(0,viewport.width-24)+0.000001
            && bounds.height*zoom <= max(0,viewport.height-reservedTop-24)+0.000001
        return (zoom,origin,fullyFits)
    }
    static func fitAllZoom(nodes: [CGRect], viewport: CGSize, reservedTop: CGFloat) -> Double {
        fitAllViewport(nodes:nodes,viewport:viewport,reservedTop:reservedTop).zoom
    }
    static func selectionOrigin(bounds: CGRect, zoom: Double, reservedTop: CGFloat) -> CGPoint {
        // Padding and the unobscured viewport inset cancel: retain a 12-point graph gap.
        CGPoint(x:bounds.minX*zoom-30,y:bounds.minY*zoom-12)
    }
}
