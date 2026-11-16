import Foundation
import KwipMLCore

/// One universal frame. `detail` accepts legacy saved layouts without changing rendering.
struct CanvasNodeSurfaceStyle: Equatable {
    init(detail: CanvasNodeDetail) {}
    var iconSize: CGFloat {32}
    var cornerRadius: CGFloat {10}
    func borderWidth(isSelected: Bool) -> CGFloat {isSelected ? 2 : 1.5}
}
