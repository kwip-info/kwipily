import SwiftUI
import KwipMLCore

/// Public editable practice. Opening or running this collection does not award mastery.
enum FileInputLessonContent {
    static let version = 3
    static let title = "Read files, documents and image pixels"
    static let isAssessed = false
    static let objective = "A directory is not automatically a useful training table. First decide what one row should represent and which content needs reading.\n\nThe supplied canvas contains File Input → View. Its local example collection has ten two-line text files, one short Word document and one tiny color PNG. File Input records file identity and extracts text or captured pixels. View lets you inspect the resulting table. No personal files, network source or model training are required."
    static let terms = "- **File identity:** the file_id retained for the captured file.\n- **Unit:** what one row represents, such as a file, line, page or image frame.\n- **Pixel:** one location in an image.\n- **Channel:** one value at that location, such as red, green or blue.\n- **Snapshot:** the captured content for this run, not a later reread of a changed file."
    static let steps = [
        "1. Select File Input and inspect Configuration.\n2. Find Directory mode, selected file types and include/exclude filters.\n3. Keep one row per file and First image frame for the initial run.\n\nThe filters decide which files enter; the reader's unit decides how many rows each selected file produces. Apply any changes explicitly.",
        "1. Run Flow and select View → Results → Output.\n2. Check twelve rows in the original collection.\n3. Compare text_content and image_content.\n\nText and document rows have text_content. The image row has image_content. The other content cell is missing by design; a missing image on a text row is not a reader failure.",
        "1. Find the colors PNG row in View and open its image cell.\n2. Check width 3, height 2 and three channels.\n3. Compare red, green and blue on the first row with white, gray and black on the second.\n\nHWC means height × width × channels. Here uint8 stores each channel as a whole number from 0 to 255. These are captured pixels, not model predictions.",
        "1. Set Text files → Rows to one row per line.\n2. Apply and run again.\n3. Check twenty text rows plus one document and one image row: twenty-two in total.\n4. Compare file_id and unit_index, then restore one row per file.\n\nRows from one file share its identity but have different unit indexes.",
        "Compare Apply & preview with the executed View table. File Input's bounded reader preview examines at most eight files and uses image metadata rather than full pixels; it can omit this collection's document and image.\n\nView requests the complete selected upstream collection within its source filters, explicit limits and resource budgets. Working-data sampling selects files before extraction. It cannot prove that unselected files would read successfully.",
        "For your own collection, select only the reader content you need. Stored PDF text extraction is different from OCR of a scanned image. Image reading does not resize, crop or normalize unless the corresponding explicit operation is configured.\n\nPredict what Stop, recorded error rows or skipped files would do to the output count. Run and inspect Diagnostics before deciding the collection was fully read.",
        "Explain the boundary before connecting a model: an image cell refers to captured pixels; it is not yet a tensor with a training shape. Text likewise needs an explicit tokenization or feature path.\n\nSave this practice if you want to keep your reader choices. Start fresh opens a separate supplied collection canvas without overwriting this work."
    ]
    static let boundary = "An image cell is a typed reference to captured pixels, with explicit geometry, channels, data type and layout. It is not an ML tensor or a pathname to be loaded implicitly. This first wave does not connect image cells directly to model training, numeric casts, join keys or ordinary delimited output. Text extraction is not OCR. PCM WAV reading is now available as an explicit Audio reader choice; it preserves integer samples, rate and channels. Prepare media performs explicit conversion into model tensors. File Input does not itself run OCR or speech models. Continue with Read text from scanned images, Transcribe a private recording or Classify sounds in a recording for those separate tools and explicit model setup. Other audio codecs and video are unsupported. Editable practice has no scored pass, completion receipt or mastery award."
}

struct FileInputLearningView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !inLessonRail { LessonTextPanel(title: "File Input · editable practice", text: FileInputLessonContent.objective) }
            if !inLessonRail { LessonTextPanel(title: "Words to know", text: FileInputLessonContent.terms) }
            if !inLessonRail { LessonTextPanel(title: "Try it on the canvas", text: FileInputLessonContent.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n")) }
            Button("Open file exploration practice") { _ = store.loadComponentWorkflowExample(FileInputViewPracticeExample.exampleID, device: "cpu") }
                .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.file-input.open-example")
            if !inLessonRail { LessonTextPanel(title: "What the output means", text: FileInputLessonContent.boundary) }
            Text("Content v\(FileInputLessonContent.version) · Authored 2026-10-13 · Editable practice · Results require execution")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }.accessibilityIdentifier("learning.file-input")
    }
}
