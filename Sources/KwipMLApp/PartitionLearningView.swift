import SwiftUI
import KwipMLCore

enum PartitionLessonContent {
    static let title="Partition rows, entities and time"
    static let objective="A model needs an honest test on observations it did not learn from. Partition keeps Training, Validation and Test separate according to the prediction problem: new rows, new people or future measurements.\n\nThe supplied canvas reads thirty synthetic measurements: ten entities with three rows each, two classes, and times from 0 to 29000 milliseconds. Partition sends three populations to separate Views. The Exclusions output records any time-gap rows you deliberately leave out. No model is trained in this practice."
    static let terms="- **Entity:** a subject or group whose measurements should stay together.\n- **Stratum:** a value used to balance a random allocation.\n- **Embargo:** a gap after a time cutoff.\n- **Holdout:** observations reserved from fitting. Validation guides choices; Test is reserved for final evaluation."
    static let boundary="Why can entity percentages differ from row percentages? Why must validation and test stay out of fitted statistics? Which exact rows does the embargo exclude? Compare your predictions with the captured outputs. This editable practice has no scored completion or mastery award."
    static let steps=[
        "1. Select Partition and inspect Random, seed 42, Validation 20% and Test 20%.\n2. Keep these original choices for the first run.\n3. Follow Training, Validation and Test to their separate Views.\n\nTraining is for fitting. Validation helps compare development choices. Test is reserved for the final check. A seed makes a random assignment repeatable for the same input and settings.",
        "1. Run Flow and inspect each output.\n2. Expect eighteen Training, six Validation and six Test rows, with no Exclusions.\n3. Change the seed, Apply and run again.\n\nCompare row identities, not just the counts. Restore seed 42 before the next step.",
        "1. Enable Stratify by column and select class.\n2. Apply and run.\n3. Check each class's fifteen rows: nine Training, three Validation and three Test.\n\nA stratum is a group used to balance the random allocation. A rare group that cannot fill the requested populations stops rather than silently changing your allocation.",
        "1. Switch to By entity and add entity as a key.\n2. Keep Validation and Test at 20%, Apply and run.\n3. Check that each entity occurs in only one output.\n\nTen entities allocate six/two/two here, producing eighteen/six/six rows because every entity has three measurements. Uneven entity sizes would give different row percentages.",
        "1. Choose By time, time column time and Epoch milliseconds.\n2. Set Training cutoff 9000 and Validation cutoff 19000.\n3. Set embargo to 2 Seconds, allow entities across time, Apply and run.\n4. Expect ten/eight/eight retained rows; Exclusions should contain times 10000, 11000, 20000 and 21000.\n\nAn embargo is a gap after each cutoff. These four observations are deliberately excluded, not lost.",
        "1. Set embargo to zero, add entity and choose Require disjoint entities.\n2. Apply and run, then inspect the attributed diagnostic.\n3. e3 crosses Training/Validation and e6 crosses Validation/Test.\n4. Allow across time and rerun to restore the chronological example.\n\nWhich split answers your own prediction question? Time order alone does not guarantee different people in each population. This practice is unscored."
    ]
}
struct PartitionLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Partition · editable practice",text:PartitionLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title:"Words to know",text:PartitionLessonContent.terms) }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:PartitionLessonContent.steps.enumerated().map{"**Step \($0.offset+1)**\n\($0.element)"}.joined(separator:"\n\n")) }
            Button("Open Partition practice") {_ = store.loadComponentWorkflowExample(PartitionPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.partition.open-example")
            PartitionInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:PartitionLessonContent.boundary) }
        }.accessibilityIdentifier("learning.partition")
    }
}
