import SwiftUI
import KwipMLCore

struct ModelOutputLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            LessonTextPanel(title:ModelOutputPracticeExample.title,text:ModelOutputPracticeExample.objective)
            LessonTextPanel(title:"Try it on the canvas",text:ModelOutputPracticeExample.instructions)
            Button("Open Model Output practice") {_ = store.loadComponentWorkflowExample(ModelOutputPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.model-output.open-example")
            LessonTextPanel(title:"Check your understanding",text:ModelOutputPracticeExample.understanding)
        }.accessibilityIdentifier("learning.model-output")
    }
}
