import SwiftUI

/// The rail owns teaching when present; standalone reference uses the same authored blocks.
struct TrainingLessonReferenceView<Controls:View>:View {
    let entryID:String
    @ViewBuilder let controls:()->Controls
    @Environment(\.lessonRailPresented) private var inRail
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            controls()
            if !inRail,let steps=TrainingTopicLessonCopy.steps(entryID) {
                ForEach(steps) {step in
                    InspectorSection(step.title,symbol:"book.closed",identifier:"learning.training."+entryID+"."+step.id,initiallyExpanded:step.id=="goal") {
                        ForEach(step.blocks) {block in LessonRichText(text:block.text)}
                    }
                }
            }
        }
    }
}
