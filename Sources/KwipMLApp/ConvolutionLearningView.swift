import SwiftUI

/// Reading navigation only; original mathematical lessons retain assessment ownership.
enum ConvolutionLearningContent {
    static let lessonID="convolution-overview"
    static let defaultComponentID="conv2d"
    static let placementInstructions="Place Convolution, then choose Configuration → Operation and Dimensions. Configure channels, spatial geometry and the operation's supported padding controls."
    static let assessmentDisclosure="The original lessons and mathematical assessments remain separate. Reading this overview does not award completion. Transposed convolution does not reconstruct an arbitrary original input."
    struct Entry:Identifiable {
        let componentID:String
        let title:String
        let lessonID:String
        let summary:String
        var id:String {componentID}
    }
    static let entries:[Entry]=[
        .init(componentID:"conv1d",title:"Standard · 1D",lessonID:"convolution",summary:"Share learned windows along a line, with explicit channel groups."),
        .init(componentID:"conv2d",title:"Standard · 2D",lessonID:"convolution",summary:"Share learned windows across height and width."),
        .init(componentID:"conv3d",title:"Standard · 3D",lessonID:"convolution",summary:"Share learned windows across depth, height and width."),
        .init(componentID:"conv_transpose1d",title:"Transposed · 1D",lessonID:"transposed-convolution",summary:"Spread learned windows into an output line and add overlaps."),
        .init(componentID:"conv_transpose2d",title:"Transposed · 2D",lessonID:"transposed-convolution",summary:"Spread learned windows across an output map and choose its extent."),
        .init(componentID:"conv_transpose3d",title:"Transposed · 3D",lessonID:"transposed-convolution",summary:"Spread learned windows through an output volume."),
        .init(componentID:"depthwise_conv2d",title:"Depthwise · 2D",lessonID:"depthwise",summary:"Apply separate spatial filters to each channel, with a channel multiplier.")
    ]
}

struct ConvolutionLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Convolution").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text(ConvolutionLearningContent.placementInstructions).font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(ConvolutionLearningContent.entries) {entry in
                HStack(alignment:.top,spacing:12) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(entry.title).font(.system(size:13,weight:.semibold))
                        Text(entry.summary).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Read lesson") {ApplicationNavigationWindows.shared.showLearn(entryID:entry.lessonID)}
                        .buttonStyle(.link)
                        .accessibilityLabel("Read \(entry.title) convolution lesson")
                        .accessibilityIdentifier("convolution.lesson."+entry.componentID)
                }
            }
            if !inLessonRail { Text(ConvolutionLearningContent.assessmentDisclosure).font(.caption).foregroundStyle(MLTheme.muted) }
        }
    }
}
