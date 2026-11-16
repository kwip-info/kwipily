import Foundation

/// UI-only settings. These never enter an executable graph or semantic digest.
/// Legacy serialized values; live canvas arrangement is always left-to-right.
public enum CanvasFlowDirection: String, Codable, CaseIterable { case leftToRight, topToBottom }
/// Legacy serialized values only; all values render the universal table frame.
public enum CanvasNodeDetail: String, Codable, CaseIterable { case small, medium, large }
public struct CanvasPresentationSettings: Codable, Equatable {
    public var flowDirection: CanvasFlowDirection
    public var nodeDetail: CanvasNodeDetail
    public init(flowDirection: CanvasFlowDirection = .leftToRight, nodeDetail: CanvasNodeDetail = .medium) {
        self.flowDirection = flowDirection; self.nodeDetail = nodeDetail
    }
}

/// One supported range for controls, fitting, restoration and persisted view validation.
public enum CanvasZoomLimits {
    public static let minimum: Double = 0.01
    public static let maximum: Double = 1.5
    public static let supported = minimum...maximum
}
