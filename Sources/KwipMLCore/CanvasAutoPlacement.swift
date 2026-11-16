import Foundation
import CoreGraphics

/// Finds a free ribbon insertion slot without altering authored positions.
public enum CanvasAutoPlacement {
    public static func origin(type:String,graph:GraphDocument,registry:ComponentRegistry,catalog:ComponentRegistry,presentation:CanvasPresentationSettings,start:CGPoint)->CGPoint {
        let geometry=CanvasGeometrySnapshot(registry:registry,presentation:presentation,graph:graph,validation:GraphValidator.validate(graph,registry:registry))
        let projection=geometry.projection
        func size(_ type:String,_ registry:ComponentRegistry)->CGSize {
            let component=registry.component(type:type)
            return CanvasNodeGeometry.size(detail:presentation.nodeDetail,direction:presentation.flowDirection,inputCount:component?.inputs.count ?? 0,outputCount:component?.outputs.count ?? 0)
        }
        let occupied=graph.nodes.map {CGRect(origin:projection.project(CGPoint(x:$0.x,y:$0.y)),size:geometry.size(nodeID:$0.id)).insetBy(dx:-12,dy:-12)}
        let candidateSize=size(type,catalog)
        // Index only the candidate rows intersected by each occupied rectangle.
        // Exact intersection remains the authority at row boundaries.
        let limit=max(1,graph.nodes.count)*128+4
        let lastRow=(limit-1)/4
        let step=120 * projection.scaleY
        let extent=candidateSize.height
        var blocked=Set<Int>()
        func candidate(_ index:Int)->CGPoint {
            CGPoint(x:start.x+CGFloat(index%4)*180,y:start.y+CGFloat(index/4)*120)
        }
        for column in 0..<4 {
            let base=projection.project(candidate(column))
            let origin=base.y
            for rect in occupied {
                let lower=(rect.minY-extent-origin)/step
                let upper=(rect.maxY-origin)/step
                guard lower.isFinite,upper.isFinite,upper >= 0,lower <= CGFloat(lastRow) else {continue}
                let first=Int(max(0,floor(lower)))
                let last=Int(min(CGFloat(lastRow),ceil(upper)))
                guard first <= last else {continue}
                for row in first...last {
                    let index=row*4+column
                    if index < limit,rect.intersects(CGRect(origin:projection.project(candidate(index)),size:candidateSize)) {
                        blocked.insert(index)
                    }
                }
            }
        }
        for index in 0..<limit where !blocked.contains(index) {return candidate(index)}
        // Bounded fallback beyond all occupied geometry, including unusual imports.
        let point=CGPoint(x:max(projection.project(start).x,occupied.map(\.maxX).max() ?? 0)+24,y:projection.project(start).y)
        return projection.inverse(point)
    }
}
