import SwiftUI

struct TrainingChartViewport:Equatable {
    private(set) var lower=0.0,upper=1.0
    var isFitted:Bool {lower==0 && upper==1}
    mutating func setLower(_ value:Double) {guard value.isFinite else {return};lower=min(max(0,value),upper-0.001)}
    mutating func setUpper(_ value:Double) {guard value.isFinite else {return};upper=max(min(1,value),lower+0.001)}
    mutating func fit() {self=Self()}
}
enum TrainingChartLayout {
    static func plotHeight(availableHeight:CGFloat?,expanded:Bool)->CGFloat {
        guard let height=availableHeight,height.isFinite,height>0 else {return expanded ? 360:210}
        // Header, wrapping controls, navigator and readout keep their own space.
        // Very short docks scroll rather than reducing the plot to nothing.
        return max(140,height-210)
    }
}
private struct TrainingChartViewportHeightKey:EnvironmentKey {static let defaultValue:CGFloat?=nil}
extension EnvironmentValues {
    var trainingChartViewportHeight:CGFloat? {
        get {self[TrainingChartViewportHeightKey.self]}
        set {self[TrainingChartViewportHeightKey.self]=newValue}
    }
}
