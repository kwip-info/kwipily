import CoreGraphics

/// Automatic visibility only; this never changes graph layout, fit or viewport state.
enum CanvasOverviewPlacement {
    static let mapSize = CGSize(width:144,height:81)
    static let margin:CGFloat = 10
    static let protectivePadding:CGFloat = 6

    static func screenFrame(graph:CGRect,zoom:Double,scrollOrigin:CGPoint,toolbarInset:CGFloat)->CGRect {
        CGRect(x:graph.minX*zoom-scrollOrigin.x,y:graph.minY*zoom+toolbarInset-scrollOrigin.y,width:graph.width*zoom,height:graph.height*zoom)
    }
    static func overlayFrame(viewport:CGSize,toolbarInset:CGFloat)->CGRect? {
        guard viewport.width.isFinite,viewport.height.isFinite,toolbarInset.isFinite,toolbarInset>=0,
              viewport.width>=mapSize.width+2*margin,viewport.height>=toolbarInset+mapSize.height+2*margin else { return nil }
        return CGRect(x:viewport.width-margin-mapSize.width,y:viewport.height-margin-mapSize.height,width:mapSize.width,height:mapSize.height)
    }
    static func shouldShow(viewport:CGSize,toolbarInset:CGFloat,graphNodes:[CGRect],zoom:Double,scrollOrigin:CGPoint)->Bool {
        guard zoom.isFinite,zoom>0,scrollOrigin.x.isFinite,scrollOrigin.y.isFinite,
              let map=overlayFrame(viewport:viewport,toolbarInset:toolbarInset) else {return false}
        let visible=CGRect(x:0,y:toolbarInset,width:viewport.width,height:viewport.height-toolbarInset)
        for node in graphNodes {
            guard [node.minX,node.minY,node.width,node.height].allSatisfy(\.isFinite) else {return false}
            let frame=screenFrame(graph:node,zoom:zoom,scrollOrigin:scrollOrigin,toolbarInset:toolbarInset).intersection(visible)
            guard !frame.isNull,!frame.isEmpty else {continue}
            let protected=frame.insetBy(dx:-protectivePadding,dy:-protectivePadding)
            if protected.minX<=map.maxX && protected.maxX>=map.minX && protected.minY<=map.maxY && protected.maxY>=map.minY {return false}
        }
        return true
    }
}
