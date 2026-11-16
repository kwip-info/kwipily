import SwiftUI
import KwipMLCore

struct ModelInputLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            ForEach(ModelInputLessonContent.steps.prefix(2)) {step in
                LessonTextPanel(title:step.title,text:step.blocks[0].text)
            }
            Button("Open named token-input practice") {_=store.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
            ForEach(ModelInputLessonContent.steps.suffix(2)) {step in
                LessonTextPanel(title:step.title,text:step.blocks[0].text)
            }
        }.accessibilityIdentifier("learning.model-input")
    }
}
