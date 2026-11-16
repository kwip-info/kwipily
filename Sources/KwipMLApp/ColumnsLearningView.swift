import SwiftUI
import KwipMLCore

/// A current editable practice over the Columns grid.
enum ColumnsLessonContent {
    static let version = 3
    static let title = "Choose columns, names, order and types"
    static let isAssessed = false
    static let objective = "Before a model uses a table, its columns need clear names and appropriate value types. A customer identifier such as 001 must stay text if those zeros matter; a score such as 12.5 must become a decimal number before numeric calculations.\n\nThe supplied canvas is Table Input → Columns → View. It reads six synthetic observations as text. Columns keeps score_text, person_id and active_text, renames two of them and converts scores and true/false values. It excludes notes. View displays the resulting table; no model training is involved."
    static let terms = "- **Source name:** the incoming column, used to identify it even after renaming.\n- **Output name:** the name passed to the next tool.\n- **Type:** the kind of value, such as text, whole number, decimal number or Boolean (true/false).\n- **Cast:** convert a value to a chosen type.\n- **Keep type:** retain the incoming type rather than guessing a conversion.\n- **Image:** retain an existing captured image reference; this choice does not load a pathname or turn text into pixels."
    static let steps = [
        "1. Select Columns and open Configuration.\n2. Find score_text, person_id, active_text and notes in the grid.\n3. Compare Keep, Output name and Type with the incoming names.\n\nEach grid row represents a column, not a data observation. Keep selects columns; row order in this grid controls their output order.",
        "1. Run Flow with the supplied settings.\n2. Select Columns and open Results → Output.\n3. Check six rows and three columns, in order: score, person_id, active.\n\nThe first observation should contain decimal 12.5, text 001 and Boolean true. Compare those values with the source text. Leading zeros in person_id remain meaningful because its Type is Keep type.",
        "1. Move person_id ahead of score in the grid.\n2. Rename score to assessment_score and include notes.\n3. Apply, then run again.\n4. Compare output column order, names, types and values.\n\nReordering grid rows changes columns. It does not reorder observations. Excluding notes also does not remove observations.",
        "1. Set active_text to a whole-number type and choose Error for conversion failure.\n2. Apply and run.\n3. Inspect the marker on Columns and its Diagnostics, or global Diagnostics with nothing selected.\n4. Restore Boolean, Apply and run successfully.\n\nError refuses an invalid conversion. Null would retain the observation with a missing converted cell. Neither choice makes true into a valid integer automatically.",
        "Select a text-to-number conversion and compare its trimming setting. Trimming can remove surrounding whitespace before conversion; Keep type leaves incoming values untouched.\n\nPredict how text 001 would look after an integer conversion. Keep identifiers as text when numeric arithmetic is not their purpose.",
        "Inspect the policy for newly appearing columns: keep, drop or refuse them. Distinguish a missing cell from a configured column disappearing entirely.\n\nBefore finishing, restore the supplied columns and types, Apply and run. Explain which changes affected names, column order, types or selected fields; none of them should filter the rows."
    ]
    static let boundary = "Choose Image only for a column that already contains image references: its captured pixels and missing values stay intact. Scalar values cannot be cast to Image, and image references cannot be cast to numbers or text. Keep type also retains an incoming image column. Columns changes selection, column order, names and value types. It does not filter or reorder rows, fill missing values, learn training statistics or infer features. Use unique nonempty output names and choose a type that fits the data. This editable practice has no scored pass, completion receipt or mastery award.\n\nIn a template project, Match fields can add an ordinary Columns block named Match input fields. It records your explicit choices when incoming headers differ from the template. Matching uses names, so reordering a later CSV does not swap its meaning. It does not invent a missing customer identifier or rewrite later formulas. Review the saved block before running new data."
}

struct ColumnsLearningView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !inLessonRail { LessonTextPanel(title: "Columns · editable practice", text: ColumnsLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title: "Words to know", text: ColumnsLessonContent.terms) }
            if !inLessonRail { LessonTextPanel(title: "Try it on the canvas", text: ColumnsLessonContent.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n")) }
            Button("Open Columns practice") { _ = store.loadComponentWorkflowExample(ColumnsPracticeExample.exampleID, device: "cpu") }
                .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.columns.open-example")
            if !inLessonRail { LessonTextPanel(title: "What the output means", text: ColumnsLessonContent.boundary) }
            Text("Content v\(ColumnsLessonContent.version) · Updated 2026-10-24 · Editable practice · Results require execution")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }.accessibilityIdentifier("learning.columns")
    }
}
