import Foundation

/// Pure display geometry: transpose is its own inverse; authored positions are untouched.
public enum CanvasNodeGeometry {
    public static func project(point: CGPoint, direction: CanvasFlowDirection) -> CGPoint {
        direction == .leftToRight ? point : CGPoint(x: point.y, y: point.x)
    }
    public static func inverse(point: CGPoint, direction: CanvasFlowDirection) -> CGPoint { project(point: point, direction: direction) }
    public static let tableWidth: CGFloat = 256
    public static let headerHeight: CGFloat = 58
    public static let rowHeight: CGFloat = 34
    public static let footerHeight: CGFloat = 24
    // Icons straddle the frame edge so the wire remains visible to its center.
    public static let portInset: CGFloat = 0

    /// Legacy detail values are retained for document compatibility only. Every
    /// canvas uses the same upright table, including top-to-bottom arrangements.
    public static func size(detail: CanvasNodeDetail, direction: CanvasFlowDirection, inputCount: Int, outputCount: Int) -> CGSize {
        let rows = max(1, min(128, max(0,inputCount)) + min(128, max(0,outputCount)))
        return CGSize(width:tableWidth,height:headerHeight + CGFloat(rows)*rowHeight + footerHeight)
    }
    public static func portAnchor(size: CGSize, direction: CanvasFlowDirection, output: Bool, index: Int, count: Int) -> CGPoint {
        let boundedCount = min(128,max(1,count)), boundedIndex = max(0,min(index,boundedCount-1))
        let rowCount = max(1,Int(((size.height-headerHeight-footerHeight)/rowHeight).rounded()))
        let row = (output ? max(0,rowCount-boundedCount) : 0) + boundedIndex
        return CGPoint(x:output ? size.width-portInset : portInset,y:headerHeight + (CGFloat(row)+0.5)*rowHeight)
    }
    public static func controlPoints(start: CGPoint, end: CGPoint, direction: CanvasFlowDirection) -> (CGPoint,CGPoint) {
        // Tangents follow the row icons, independently of graph arrangement.
        let bend = max(45,abs(end.x-start.x)*0.45)
        return (CGPoint(x:start.x+bend,y:start.y),CGPoint(x:end.x-bend,y:end.y))
    }
}

/// Reversible display spacing keeps medium-authored separation when cards grow.
/// Scale is based on the registry's maximum port counts, never node positions.
public struct CanvasCoordinateProjection: Equatable {
    public let direction: CanvasFlowDirection
    public let scaleX: CGFloat
    public let scaleY: CGFloat
    public init(settings: CanvasPresentationSettings, maximumInputCount: Int = 2, maximumOutputCount: Int = 2) {
        // Old sidecars retain their value, but all live canvases use left-to-right.
        direction = .leftToRight
        let size = CanvasNodeGeometry.size(detail:settings.nodeDetail,direction:direction,inputCount:maximumInputCount,outputCount:maximumOutputCount)
        let baseline = direction == .leftToRight ? CGSize(width:150,height:84) : CGSize(width:84,height:150)
        scaleX = max(1,size.width/baseline.width)
        scaleY = max(1,size.height/baseline.height)
    }
    public func project(_ point: CGPoint) -> CGPoint {
        let transposed = CanvasNodeGeometry.project(point:point,direction:direction)
        return CGPoint(x:transposed.x*scaleX,y:transposed.y*scaleY)
    }
    public func inverse(_ point: CGPoint) -> CGPoint {
        CanvasNodeGeometry.inverse(point:CGPoint(x:point.x/scaleX,y:point.y/scaleY),direction:direction)
    }
}
