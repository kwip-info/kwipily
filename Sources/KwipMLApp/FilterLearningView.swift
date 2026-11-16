import SwiftUI
import KwipMLCore

/// Consolidated Filter practice is additive; legacy lesson and project identities stay valid.
enum FilterLessonContent {
    static let version = 2
    static let title = "Choose rows and inspect both Filter branches"
    static let isAssessed = false
    static let objective = "A preparation flow often needs to separate observations that meet a rule from those that do not. Both groups matter: a rejected row can explain a quality problem.\n\nThe supplied Table Input reads eight synthetic observations with amount and status fields. One amount is missing. Filter sends Matching and Nonmatching to separate View tools. You will compare those two tables while changing one rule at a time. This is row selection, not a Training/Validation/Test split."
    static let terms = "- **Condition:** a rule checked for each observation.\n- **All:** every condition must be true, like AND.\n- **Any:** at least one condition must be true, like OR.\n- **Expression:** a written rule over named columns, not arbitrary Python.\n- **Text pattern:** a regular expression describing text to match.\n- **Null:** a missing cell, different from zero or empty text."
    static let steps = [
        "1. Select Filter and inspect Simple mode.\n2. Find the condition amount greater than or equal to 10.\n3. Follow Matching and Nonmatching to their separate Views.\n\nPredict where a row with amount 10 belongs. Decide what an unknown amount can tell you before running.",
        "1. Run Flow and inspect both captured tables.\n2. Matching should contain A01, A03, A07 and A08.\n3. Nonmatching should contain A02, A04, A05 and A06.\n\nTogether they contain all eight original rows once, with relative order retained inside each branch. A nonmatching row has not disappeared.",
        "1. Add status equals ready as a second Simple condition.\n2. Use case-insensitive comparison so READY also matches.\n3. Select All, Apply and run: expect A01, A03 and A08 in Matching.\n4. Predict the result of Any, then run and compare.\n\nAll narrows this result because both conditions are required; Any accepts either condition.",
        "1. Switch to Expression mode.\n2. Start with [amount] >= 10, Apply and run.\n3. Compare it with the original Simple result.\n\nSquare brackets name a column. AND and OR combine supported conditions; the editor does not execute Python, SQL or arbitrary functions.",
        "1. Switch to Text pattern mode on status.\n2. Use ^ready$, Full matching and Ignore case.\n3. Apply and run: expect A01, A03, A04, A06 and A08.\n\nFull requires the entire value. Search permits a match inside longer text. Case and line flags are explicit choices.",
        "1. Restore Simple with only amount >= 10.\n2. Compare missing-value Nonmatching with Error, applying and running each choice.\n3. Inspect the attributed failure when the final condition is unknown.\n4. Restore Nonmatching and rerun.\n\nUse Is missing for an intentional missing-value rule. Null is never automatically treated as zero."
    ]
    static let boundary = "Filter preserves column names, types, captured image references and row identities. It selects rows; it does not rename, cast or calculate columns. Both output ports remain available in every mode. A nonmatching row has not been deleted from its own branch. Conditions on image pixels require an explicit image-processing tool. Filter does not create training, validation or test roles; use Partition for scientific dataset splits. This editable practice has no scored pass, completion receipt or mastery award."
}

struct FilterLearningView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !inLessonRail { LessonTextPanel(title: "Filter · editable practice", text: FilterLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title: "Words to know", text: FilterLessonContent.terms) }
            if !inLessonRail { LessonTextPanel(title: "Try it on the canvas", text: FilterLessonContent.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n")) }
            Button("Open Filter practice") { _ = store.loadComponentWorkflowExample(FilterPracticeExample.exampleID, device: "cpu") }
                .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.filter.open-example")
            if !inLessonRail { LessonTextPanel(title: "What the output means", text: FilterLessonContent.boundary) }
            Text("Content v\(FilterLessonContent.version) · Authored 2026-10-13 · Editable practice · Results require execution")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }.accessibilityIdentifier("learning.filter")
    }
}
