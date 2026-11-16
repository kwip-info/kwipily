import SwiftUI
import KwipMLCore

struct SortInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            LessonTextPanel(title:"Physical row order",text:"Sort reorders the actual table by an ordered list of keys. The first differing key decides the order. Equal composite keys retain their incoming order even with descending keys. Sorting a Results viewer header changes presentation only.")
            LessonTextPanel(title:"Comparison and missing values",text:"Each key has its own ascending/descending direction and independent missing-first/missing-last setting. Text uses exact Unicode comparison by default; Unicode casefold matches case variants such as Straße and STRASSE. Casefold changes comparison only, never emitted values. There is no locale or natural-number guessing. Images pass through but cannot be keys; temporal keys compare canonical dates, times or instants.")
            LessonTextPanel(title:"Population and evidence",text:"Sort preserves all columns, original row identities, partition role and fitted-state evidence. Working data defines its population; Sort does not automatically read a complete source. View and Table Output request complete dependencies. External sorting uses bounded disk scratch and streaming output, subject to shared cell/storage budgets; there is no 100,000-row limit.")
        }
    }
}
struct UniqueInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            LessonTextPanel(title:"Two complementary outputs",text:"Choose at least one key column. First or Last retains one member of each equal key group in Unique. Duplicates contains every non-retained row. Both outputs preserve incoming order, original values and identities; together they cover the input population without overlap.")
            LessonTextPanel(title:"Missing and text keys",text:"Equal lets missing keys match. Distinct rows makes every row with any missing key its own group. Stop run refuses a missing key. Exact text comparison preserves case distinctions; Unicode casefold matches case variants without editing emitted text. Images pass through but cannot be keys; temporal keys compare canonical dates, times or instants.")
            LessonTextPanel(title:"Scope and evidence",text:"Uniqueness applies to the current working population; a sample does not prove whole-source uniqueness. View and Table Output request complete dependencies. Unique preserves partition and fitted-state evidence and uses bounded disk key tracking plus streamed output. Repeated entity keys can still require group-aware partitioning for evaluation.")
        }
    }
}
struct SortLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Sort · editable practice",text:DataPracticeLessonCopy.packets["sort"]!.introduction) }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["sort"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open Sort practice") {_ = store.loadComponentWorkflowExample(SortPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.sort.open-example")
            SortInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:DataPracticeLessonCopy.packets["sort"]!.reflection) }
        }
    }
}
struct UniqueLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Unique · editable practice",text:DataPracticeLessonCopy.packets["unique"]!.introduction) }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["unique"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open Unique practice") {_ = store.loadComponentWorkflowExample(UniquePracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.unique.open-example")
            UniqueInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:DataPracticeLessonCopy.packets["unique"]!.reflection) }
        }
    }
}
