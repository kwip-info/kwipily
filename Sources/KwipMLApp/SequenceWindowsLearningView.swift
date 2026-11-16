import SwiftUI
import KwipMLCore

struct SequenceWindowsLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Sequence windows · editable practice",text:DataPracticeLessonCopy.packets["sequence-windows"]!.introduction) }
            Button("Open Sequence windows practice") {_=store.loadComponentWorkflowExample(SequenceWindowsPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["sequence-windows"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }


            Button("Open next-token training practice") {_=store.loadComponentWorkflowExample(SequenceTrainingPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)

        }.accessibilityIdentifier("learning.sequence-windows")
    }
}
