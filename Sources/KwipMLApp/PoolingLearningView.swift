import SwiftUI

/// Reading navigation only. Original pooling lessons own their assessments and saved evidence.
enum PoolingLearningContent {
    static let lessonID="pooling-overview"
    static let defaultComponentID="max_pool2d"
    static let placementInstructions="Place Pooling, then choose Configuration → Method, Sizing and Dimensions. A lesson offers only its permitted combinations."
    static let assessmentDisclosure="Each operation keeps its original mathematical lessons and assessment. Reading this overview does not award completion."
    struct Entry:Identifiable {
        let componentID:String
        let title:String
        let lessonID:String
        let summary:String
        var id:String {componentID}
    }
    static let entries:[Entry]=[
        .init(componentID:"max_pool1d",title:"Max · fixed window · 1D",lessonID:"pooling",summary:"Choose local winners along each channel and inspect their positions."),
        .init(componentID:"max_pool2d",title:"Max · fixed window · 2D",lessonID:"pooling",summary:"Choose local winners across height and width."),
        .init(componentID:"max_pool3d",title:"Max · fixed window · 3D",lessonID:"pooling",summary:"Choose local winners across depth, height and width."),
        .init(componentID:"avg_pool1d",title:"Average · fixed window · 1D",lessonID:"pooling",summary:"Compare window averages and partial boundary divisors."),
        .init(componentID:"avg_pool2d",title:"Average · fixed window · 2D",lessonID:"adaptive-pooling",summary:"Compare padding inclusion and an explicit average divisor."),
        .init(componentID:"adaptive_avg_pool1d",title:"Average · adaptive output · 1D",lessonID:"adaptive-pooling",summary:"Choose an output length and compare the resulting regions."),
        .init(componentID:"adaptive_avg_pool2d",title:"Average · adaptive output · 2D",lessonID:"adaptive-pooling",summary:"Choose output height and width, or keep either input axis."),
        .init(componentID:"adaptive_max_pool2d",title:"Max · adaptive output · 2D",lessonID:"adaptive-pooling",summary:"Choose output regions and inspect each winning spatial position.")
    ]
}

struct PoolingLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Pooling").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text(PoolingLearningContent.placementInstructions).font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(PoolingLearningContent.entries) {entry in
                HStack(alignment:.top,spacing:12) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(entry.title).font(.system(size:13,weight:.semibold))
                        Text(entry.summary).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Read lesson") {ApplicationNavigationWindows.shared.showLearn(entryID:entry.lessonID)}
                        .buttonStyle(.link)
                        .accessibilityLabel("Read \(entry.title) lesson")
                        .accessibilityIdentifier("pooling.lesson."+entry.componentID)
                }
            }
            if !inLessonRail { Text(PoolingLearningContent.assessmentDisclosure).font(.caption).foregroundStyle(MLTheme.muted) }
        }
    }
}
