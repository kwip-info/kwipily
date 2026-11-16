import SwiftUI
import KwipMLCore
struct ReshapeTableLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Reshape Table · editable practice",text:DataPracticeLessonCopy.packets["reshapeTable"]!.introduction) }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["reshapeTable"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open Long practice") {_ = store.loadComponentWorkflowExample(ReshapeTablePracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.reshapeTable.open-long")

            Button("Open Wide practice") {_ = store.loadComponentWorkflowExample(ReshapeTablePracticeExample.wideExampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.reshapeTable.open-wide")
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:DataPracticeLessonCopy.packets["reshapeTable"]!.reflection) }
            ReshapeTableInformationView()
        }
    }
}
