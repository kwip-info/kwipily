import CoreGraphics

/// World positions are persisted. Scene translation is presentation only.
enum CanvasWorldGeometry {
    static func sceneOrigin(nodes:[CGRect])->CGPoint {
        let valid=nodes.filter { !$0.isNull && !$0.isInfinite && $0.minX.isFinite && $0.minY.isFinite }
        return CGPoint(x:min(0,(valid.map(\.minX).min() ?? 40)-40),
                       y:min(0,(valid.map(\.minY).min() ?? 40)-40))
    }
    static func scenePoint(world:CGPoint,origin:CGPoint)->CGPoint {
        CGPoint(x:world.x-origin.x,y:world.y-origin.y)
    }
    static func worldPoint(scene:CGPoint,origin:CGPoint)->CGPoint {
        CGPoint(x:scene.x+origin.x,y:scene.y+origin.y)
    }
}
