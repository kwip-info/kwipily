import SwiftUI
import KwipMLCore

enum PreprocessLessonContent {
    static let title="Prepare training and held-out tables together"
    static let objective="A missing value or a large numeric scale can make a model harder to train. Preparation can repair those values, but learning the repair from evaluation rows would leak information into training.\n\nThe supplied canvas reads twelve synthetic observations and partitions them by time: six Training, three Validation and three Test. One Preprocess receives all three tables and sends each prepared population to its own View. It learns replacements, category vocabulary and scaling from Training only. Validation and Test use those same learned rules."
    static let terms="- **Imputation:** replacing missing observations.\n- **Vocabulary:** categories learned from Training.\n- **Population standard deviation:** spread used for scaling, with divisor N.\n- **Leakage:** using held-out observations to learn preparation rules.\n- **Fixed rule:** a choice that learns no statistic."
    static let boundary="Why do held-out outliers not change the training mean? Why is c absent from the learned vocabulary? When should missing values be handled in Formula or Filter instead? This editable practice offers no scored completion or mastery award."
    static let steps=[
        "1. Follow Partition's Training, Validation and Test connections into Preprocess.\n2. Select Preprocess and inspect x and city in its column grid.\n3. Find the matching prepared output for each input population.\n\nThere is one shared preparation node, not separate fit/apply nodes. No training model is needed to inspect these tables.",
        "1. Run Flow and inspect the prepared Training table.\n2. Compare source x values 1, 2, missing, 4, 5 and 6.\n3. Mean replacement learns 3.6 from the five observed values.\n\nStandard scaling then centers and scales all six retained Training values, including the replacement. The replaced value becomes zero after centering because it equals the learned mean.",
        "1. Select city. Most frequent learns b; One-hot learns categories a and b.\n2. Inspect c in Validation/Test: All-zero encoding records a diagnostic instead of learning another category.\n3. Choose Unknown categories → Stop run, Apply and inspect the failure.\n4. Restore All-zero encoding and rerun.\n\nOne-hot represents a category using separate zero/one columns. Unknown evaluation categories must follow a declared policy.",
        "1. Select x, try Median replacement or turn Center off, then Apply and run.\n2. Compare all three prepared outputs and captured learned state.\n3. Check that Validation values near 100 and Test values near 200 did not change Training's learned statistics.\n\nFit means learning those values from Training. Transform means applying them unchanged to each population.",
        "Test is optional, Training is required. To try an absent Test input, disconnect both that input and its downstream consumer, then reconnect the original flow afterward.\n\nFor separate cleanup before Partition, open Table cleanup practice in Exercise choices & reference. Save this canvas first. Its Fixed table rules use constants and missing-value checks without learning statistics; statistical replacement, encoding and scaling belong in partitioned mode.",
        "Explain why an entirely missing Training column needs a policy and why a tied most-frequent category needs a deterministic rule. Unconfigured target and time columns pass through unchanged.\n\nCompare captured values after an actual run rather than assuming all preparation succeeded. Optional recorded package export can replay saved rules; it does not refit them on evaluation rows. This practice is unscored."
    ]
}
struct PreprocessLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Preprocess · editable practice",text:PreprocessLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title:"Words to know",text:PreprocessLessonContent.terms) }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:PreprocessLessonContent.steps.enumerated().map{"**Step \($0.offset+1)**\n\($0.element)"}.joined(separator:"\n\n")) }
            Button("Open Preprocess practice") {_ = store.loadComponentWorkflowExample(PreprocessPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.preprocess.open-example")
            LessonTextPanel(title:"Table cleanup · try each rule",text:"Use the fixed Table mode before Partition. On a Text column containing spaces, tabs, combining marks, punctuation and decimal digits, compare None, NFC and NFKC. Then enable character removal, Trim ends or Collapse runs, and Lowercase or Uppercase. Empty text stays empty until Convert to missing is selected; the missing policy then preserves, fills, drops or stops. A typed fixed replacement is inserted after cleanup. Search only changes the visible rule list. Apply commits all changes together; Revert restores them.")
            Button("Open Table cleanup practice") {_ = store.loadComponentWorkflowExample(PreprocessPracticeExample.cleanupExampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.preprocess.open-cleanup-example")
            LessonTextPanel(title:"Predict the result",text:"With NFKC, punctuation and decimal-digit removal, Collapse runs and Lowercase, `  Ａ!１２  B  ` becomes `a b`. With Convert to missing and Drop row, ` !1 ` disappears while every retained row keeps its original identity. Why does removing control characters before whitespace handling delete a tab rather than turn it into a space? Why must mean imputation remain in partitioned mode?")
            PreprocessInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:PreprocessLessonContent.boundary) }
        }.accessibilityIdentifier("learning.preprocess")
    }
}
