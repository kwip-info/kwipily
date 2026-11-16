import SwiftUI
import KwipMLCore

enum FormulaLessonContent {
    static let title="Create columns with Formula, text and date/time"
    static let objective="Order tables often need derived values and small text fields extracted from labels. Formula lets you describe those column calculations in order, without using another node for every expression.\n\nThe supplied canvas is Table Input → Formula → View. Eight synthetic orders contain price, quantity, names and product labels. Four configured operations calculate total, clean names, extract product/batch fields and mark an ABC prefix. They add values to each row; they do not filter the orders or train a model."
    static let terms="- **Expression:** a bounded computation over named columns.\n- **Capture group:** a selected part of a text match.\n- **Replacement:** text written in place of a match.\n- **Null:** a missing value.\n- **Operation order:** later rows can use earlier calculated columns."
    static let boundary="Why must adjusted_total follow total? Why is batch Text? Does replacing whitespace also trim the ends? Which rows should match ^ABC? Compare predictions with captured Results. This editable practice has no scored completion or mastery award. Existing saved Calculate and Regex blocks keep their original execution."
    static let steps=[
        "1. Run the supplied flow and inspect Formula → Results → Output.\n2. Check total in source order: 25, 24, 20, 10, 10, 18, 15, 18.\n3. Compare the first order's price and quantity with its total.\n\nA formula runs per observation. A Boolean result is a value in a column, not a row-routing decision.",
        "1. Select Formula and the total operation.\n2. Inspect [price] * [quantity], with Decimal output.\n3. Add [total] + 1 as adjusted_total after total.\n4. Try moving it before total; inspect the missing-column finding, then restore the valid order and Apply.\n\nLater operations can use columns created earlier. Square brackets identify a column. Optional: after total, add require([quantity] > 0, [total]) as checked_total. The supplied positive quantities pass unchanged. If a later input has zero or negative quantity, this explicit assumption stops the run; it does not silently exclude that row.",
        "1. Select clean_name and inspect guided Replace text.\n2. Compare pattern \\s+, replacement one space, and All matches.\n3. Inspect Expression to see regex_replace, then return to the guided editor.\n\nRepeated spaces become one. Dorothy's leading/trailing whitespace becomes one space at each end; replacement does not trim it. Use trim([clean_name]) in a later expression if you intend to remove those ends.",
        "1. Inspect the two-column capture operation with Entire value pattern ^([A-Z]+)-([0-9]+)$.\n2. Check group 1 → product_code and group 2 → batch.\n3. Run and compare ABC/104, XYZ/207 and LMN/003 in the first three rows.\n4. Try No match → Stop run, inspect the diagnostic, then restore Missing values.\n\nCapture groups are the selected parenthesized parts of a match. Keep batch as Text so 003 keeps its zeros. bad and invalid have no complete match.",
        "1. Select is_valid and inspect regex_match with ^ABC and Contains match.\n2. Run and find true for A01, A05 and A07.\n3. Confirm all eight orders remain.\n\nThis checks a prefix. Use Filter when you want to route rows using the resulting Boolean.",
        "1. After capturing upstream input with a run, change a Formula draft and choose Test on sample.\n2. Inspect at most eight pinned captured rows, without applying the draft or rereading the source.\n3. Edit again and observe that this draft test clears.\n4. Apply to commit or Revert to restore the saved operations.\n\nThis bounded draft test is separate from the workflow's working-data setting. Duplicate operations need unique output names before applying."
    ]
}
struct FormulaLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Formula · editable practice",text:FormulaLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title:"Words to know",text:FormulaLessonContent.terms) }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:FormulaLessonContent.steps.enumerated().map{"**Step \($0.offset+1)**\n\($0.element)"}.joined(separator:"\n\n")) }
            Button("Open Formula practice") {_ = store.loadComponentWorkflowExample(FormulaPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.formula.open-example")
            FormulaTemporalLearningView(store:store)
            FormulaInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:FormulaLessonContent.boundary) }
        }.accessibilityIdentifier("learning.formula")
    }
}
