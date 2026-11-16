import SwiftUI
import KwipMLCore

/// Explicit author references; never guess a node from display text or select it for the reader.
struct LessonNodeFocus: Codable, Equatable {
    var nodeIDs:[String] = []
    var types:[String] = []
    var addresses:[WorkflowNodeAddress] = []
    var presentation:CanvasLessonFocus {
        .init(nodeIDs:Set(nodeIDs).union(addresses.compactMap {try? WorkflowCanvasIDs.nodeID($0)}),types:Set(types))
    }
    func contains(nodeID:String,type:String)->Bool {presentation.contains(nodeID:nodeID,type:type)}
}

/// Resolve addresses once per lesson step, not during pointer or camera updates.
struct CanvasLessonFocus:Equatable {
    var nodeIDs:Set<String> = []
    var types:Set<String> = []
    func contains(nodeID:String,type:String)->Bool {nodeIDs.contains(nodeID) || types.contains(type)}
}
private struct CanvasLessonFocusKey:EnvironmentKey {static let defaultValue=CanvasLessonFocus()}
extension EnvironmentValues {
    var lessonNodeFocus:CanvasLessonFocus {
        get {self[CanvasLessonFocusKey.self]}
        set {self[CanvasLessonFocusKey.self]=newValue}
    }
}

/// The dashed outer halo leaves selection, failure feedback and port hit targets intact.
struct CanvasLessonFocusHalo:View {
    let light:Bool
    private var tint:Color {light ? Color(red:0.56,green:0.36,blue:0.04):Color(red:0.94,green:0.75,blue:0.36)}
    var body:some View {
        RoundedRectangle(cornerRadius:14)
            .stroke(tint,style:StrokeStyle(lineWidth:2,dash:[6,4]))
            .padding(-5)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}
