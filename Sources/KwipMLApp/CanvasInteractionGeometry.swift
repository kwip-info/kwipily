import CoreGraphics

/// Logical graph coordinates; hit tolerance stays constant in displayed screen points.
enum CanvasInteractionGeometry {
    static let originRange: ClosedRange<CGFloat> = -1_000_000...1_000_000
    static func boundedOrigin(_ point: CGPoint) -> CGPoint {
        guard point.x.isFinite,point.y.isFinite else {return .zero}
        return CGPoint(x:max(originRange.lowerBound,min(originRange.upperBound,point.x)),y:max(originRange.lowerBound,min(originRange.upperBound,point.y)))
    }
    static func portTarget(point: CGPoint, anchors: [CGPoint], zoom: Double) -> Int? {
        guard zoom.isFinite, zoom > 0, point.x.isFinite, point.y.isFinite else { return nil }
        let hits = anchors.indices.filter { index in
            let anchor = anchors[index]
            return anchor.x.isFinite && anchor.y.isFinite && hypot(anchor.x-point.x,anchor.y-point.y)*zoom <= 12
        }
        return hits.count == 1 ? hits[0] : nil
    }
    static func translation(_ proposed: CGSize, origins: [CGPoint]) -> CGSize {
        guard !origins.isEmpty, proposed.width.isFinite, proposed.height.isFinite,
              origins.allSatisfy({ $0.x.isFinite && $0.y.isFinite && originRange.contains($0.x) && originRange.contains($0.y) }) else { return .zero }
        let minX = origins.map(\.x).min()!, maxX = origins.map(\.x).max()!
        let minY = origins.map(\.y).min()!, maxY = origins.map(\.y).max()!
        return CGSize(width: max(originRange.lowerBound - minX, min(originRange.upperBound - maxX, proposed.width)),
                      height: max(originRange.lowerBound - minY, min(originRange.upperBound - maxY, proposed.height)))
    }

    static func hitsWire(point: CGPoint, start: CGPoint, end: CGPoint, zoom: Double) -> Bool {
        guard zoom.isFinite, zoom > 0,
              [point.x, point.y, start.x, start.y, end.x, end.y].allSatisfy(\.isFinite) else { return false }
        // Port hit regions take precedence over inserting into a wire.
        guard hypot(point.x - start.x, point.y - start.y) * zoom > 20,
              hypot(point.x - end.x, point.y - end.y) * zoom > 20 else { return false }
        let bend = max(45, abs(end.x - start.x) * 0.45)
        let a = CGPoint(x: start.x + bend, y: start.y)
        let b = CGPoint(x: end.x - bend, y: end.y)
        var previous = start
        for sample in 1...96 {
            let t = CGFloat(sample) / 96, u = 1 - t
            let next = CGPoint(x: u*u*u*start.x + 3*u*u*t*a.x + 3*u*t*t*b.x + t*t*t*end.x,
                               y: u*u*u*start.y + 3*u*u*t*a.y + 3*u*t*t*b.y + t*t*t*end.y)
            let dx = next.x - previous.x, dy = next.y - previous.y
            let length = dx*dx + dy*dy
            let amount = length > 0 ? max(0, min(1, ((point.x-previous.x)*dx + (point.y-previous.y)*dy)/length)) : 0
            if hypot(point.x-previous.x-amount*dx, point.y-previous.y-amount*dy) * zoom <= 9 { return true }
            previous = next
        }
        return false
    }
}

/// macOS keyboards deliver backward delete as BS or DEL, forward delete as NSDeleteFunctionKey.
enum CanvasKeyRouting {
    static func isDelete(_ characters: String) -> Bool { ["\u{8}", "\u{7f}", "\u{f728}"].contains(characters) }
}
