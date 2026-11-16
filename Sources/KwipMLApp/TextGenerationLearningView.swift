import SwiftUI
import KwipMLCore

struct TextGenerationLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        TrainingLessonReferenceView(entryID:"text-generation") {
            Button("Open text generation practice") {_ = store.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
        }.accessibilityIdentifier("learning.text-generation")
    }
}
