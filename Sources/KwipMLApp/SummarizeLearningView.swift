import SwiftUI
import KwipMLCore

struct SummarizeLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Summarize · editable practice",text:DataPracticeLessonCopy.packets["summarize"]!.introduction) }
            if !inLessonRail { LessonTextPanel(title:"Words to know",text:"**Group:** rows sharing a key, such as order_id. **Aggregation:** a calculation producing one value per group. **Separator:** text placed between joined values. **Missing:** an absent value, distinct from an empty string.") }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["summarize"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open Summarize practice") {_ = store.loadComponentWorkflowExample(SummarizePracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.summarize.open-example")
            SummarizeInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:DataPracticeLessonCopy.packets["summarize"]!.reflection) }
        }
    }
}
