import SwiftUI
import KwipMLCore

struct SaveModelLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        TrainingLessonReferenceView(entryID:"save-model") {
            Button("Open classifier practice") {_ = store.loadPreparedWorkflowExample(device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
        }.accessibilityIdentifier("learning.save-model")
    }
}

struct LoadModelLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        TrainingLessonReferenceView(entryID:"load-model") {
            Button("Open text generation practice") {_ = store.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
        }.accessibilityIdentifier("learning.load-model")
    }
}

struct FineTuningLearningView:View {
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        TrainingLessonReferenceView(entryID:"fine-tuning") {
            Button("Open fine-tuning practice") {_=store.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
        }.accessibilityIdentifier("learning.fine-tuning")
    }
}
