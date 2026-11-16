import Foundation

/// A render-local lookup. Recreate from the current graph; never cache across edits.
public struct CanvasNodeIndex {
    private let nodesByID:[String:GraphNode]
    public init(nodes:[GraphNode]) {
        // Match the previous first(where:) behavior even for an invalid draft.
        nodesByID=Dictionary(nodes.map {($0.id,$0)},uniquingKeysWith:{first,_ in first})
    }
    public func node(_ id:String)->GraphNode? {nodesByID[id]}
}
