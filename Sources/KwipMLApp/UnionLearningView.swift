import SwiftUI
import KwipMLCore

struct UnionLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Union · editable practice",text:DataPracticeLessonCopy.packets["union"]!.introduction) }
            if !inLessonRail { LessonTextPanel(title:"Words to know",text:"**Union:** stacks rows from tables. **Join:** pairs fields across tables. **Alignment:** determines which source fields share an output column. **Coverage:** includes all fields, shared fields, or requires identical fields. **Input identity:** a stable port independent of a component’s editable name.") }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["union"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open Union practice") {_ = store.loadComponentWorkflowExample(UnionPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.union.open-example")
            UnionInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:DataPracticeLessonCopy.packets["union"]!.reflection) }
        }
    }
}
