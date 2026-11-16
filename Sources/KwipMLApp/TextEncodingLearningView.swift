import SwiftUI
import KwipMLCore

struct TextEncodingLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Text encoding · editable practice",text:DataPracticeLessonCopy.packets["text-encoding"]!.introduction) }
            Button("Open Text encoding practice") {_=store.loadComponentWorkflowExample(TextEncodingPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["text-encoding"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open token-sequence classifier") {_=store.loadComponentWorkflowExample(TextEncodingPracticeExample.trainingExampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)


        }.accessibilityIdentifier("learning.text-encoding")
    }
}
