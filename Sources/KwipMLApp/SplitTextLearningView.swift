import SwiftUI
import KwipMLCore

enum SplitTextLessonContent {
    static let version = 2
    static let title = "Separate text into columns or rows"
    static let isAssessed = false
    static let objective = "Address data may put several fields in one cell. Split Text separates those fields while keeping the person each record belongs to.\n\nThe supplied canvas is Table Input → Split Text → View. It reads three synthetic addresses plus fixed-width records. First you will split an address into street, city and postal_code. Then you will use starting positions and widths to read region, year and amount from the original record. A quoted separator inside a street must stay part of that street."
    static let steps = [
        "Run Flow and inspect Split Text → Results → Output. Expect person_id and record, followed by street, city and postal_code. The second street is 456 | Oak Ave: its quoted separator is part of the street, not another field.",
        "Keep postal_code as Text so leading zeros can survive. Try Keep source column, Apply and run again; compare original address with the extracted values. Add an output field and compare Short fields Missing versus Empty text. An empty string is a value; missing means no value is present.",
        "Switch Delimited output to Rows, set the first output name to token and include token ordinal. Apply and run. Each address now produces three observations with ordinals 1, 2 and 3. Other fields repeat. This belongs before Partition: repeated people need group-aware evaluation splitting.",
        "Switch to Fixed width and source record. Configure region at Start 1 Width 2 as Text, year at Start 3 Width 4 as Integer, and amount at Start 7 Width 7 as Number. Apply and run. The first record gives region 12, year 2023 and amount 1250.5. Fields read the original record, so overlapping ranges are allowed.",
        "Try a width larger than the source record. Short record Missing affects the incomplete field; Stop refuses it. Restore the valid widths. Unicode code points and UTF-8 bytes differ for accented letters and emoji; a byte slice that splits a character follows Conversion failure. Use Formula for regex or explicit temporal parsing."
    ]
    static let boundary = "This editable practice has no scored pass, mastery award or completion receipt. Columns and Fixed width retain the same observations; Rows changes the population and refuses existing partition/fitted lineage. Delimiters are literal, with explicit quoting and escaping. Missing source, short fields and invalid conversion are separate decisions. Run results establish the actual output; a preview is only an observation of its captured population."
}
struct SplitTextLearningView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !inLessonRail { LessonTextPanel(title: "Split Text · editable practice", text: SplitTextLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title: "Words to know", text: "**Delimiter:** literal text between fields. **Quoted field:** a wrapped value whose internal delimiter stays literal. **Ordinal:** the original 1-based token position. **Fixed width:** a field identified by starting position and length. **Unicode code point:** one encoded character value; a visible character may contain several.") }
            if !inLessonRail { LessonTextPanel(title: "Try it on the canvas", text: SplitTextLessonContent.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n")) }
            Button("Open Split Text practice") { _ = store.loadComponentWorkflowExample(SplitTextPracticeExample.exampleID, device: "cpu") }.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.splitText.open-example")
            if !inLessonRail { LessonTextPanel(title: "What the output means", text: SplitTextLessonContent.boundary) }
            Text("Content v\(SplitTextLessonContent.version) · Authored 2026-10-13 · Editable practice · Results require execution").font(.caption).foregroundStyle(MLTheme.muted)
        }.accessibilityIdentifier("learning.splitText")
    }
}
