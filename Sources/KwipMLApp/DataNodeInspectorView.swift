import SwiftUI
import KwipMLCore

struct DataNodeInspectorView<Context: DataNodeEditing>: View {
    @ObservedObject var context: Context
    var showsOutputs = true
    private var selected: DataFlowNode? { context.node }
    var body: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let node = selected {
                    ToolInspectorHeader(title: context.operatorTitle, authoredName: context.authoredName, iconType: node.operatorID)
                    InspectorConfigurationSection {
                        TextField("Block name", text: Binding(get: { context.authoredName }, set: { context.rename($0) })).flatField().disabled(context.isBusy)
                        if !["kwip.data.decide", "kwip.data.gameRules", "kwip.data.classifyAudio", "kwip.data.transcribeAudio", "kwip.data.extractText", "kwip.data.prepareMedia", "kwip.data.sequenceWindows", "kwip.data.textEncode", "kwip.data.view", "kwip.data.sourceFiles", "kwip.data.columns", "kwip.data.filterRows", "kwip.data.formula", "kwip.data.partition", "kwip.data.preprocess", "kwip.data.sampleRows", "kwip.data.join", "kwip.data.aggregate", "kwip.data.union", "kwip.data.sort", "kwip.data.unique", "kwip.data.reshapeTable", "kwip.data.splitText", "kwip.data.outputDelimited"].contains(node.operatorID) { Text(context.operatorTitle).font(.caption).foregroundStyle(MLTheme.muted) }
                        if !["kwip.data.decide", "kwip.data.gameRules", "kwip.data.classifyAudio", "kwip.data.transcribeAudio", "kwip.data.extractText", "kwip.data.prepareMedia", "kwip.data.sequenceWindows", "kwip.data.textEncode", "kwip.data.sourceCSV", "kwip.data.sourceFiles", "kwip.data.view", "kwip.data.columns", "kwip.data.filterRows", "kwip.data.formula", "kwip.data.partition", "kwip.data.preprocess", "kwip.data.sampleRows", "kwip.data.join", "kwip.data.aggregate", "kwip.data.union", "kwip.data.sort", "kwip.data.unique", "kwip.data.reshapeTable", "kwip.data.splitText", "kwip.data.outputDelimited"].contains(node.operatorID) { DataSchemaDisclosureView(disclosure: context.schemaDisclosure) }
                        configuration(node).disabled(context.isBusy && node.operatorID != GameRuleConfiguration.operatorID)
                    }.id(context.contextIdentity)
                    InspectorInformationSection(componentType: node.operatorID) {
                        if node.operatorID == SequenceWindowsConfiguration.operatorID {SequenceWindowsInformationView()}
                        else if node.operatorID == "kwip.data.textEncode" {
                            TextEncodingInformationView()
                        } else if node.operatorID == "kwip.data.sourceCSV" {
                            TableInputInformationView(configuration: node.config)
                        } else if node.operatorID == GameRuleConfiguration.operatorID {
                            Text(GameRuleLessonContent.boundary)
                        } else if node.operatorID == "kwip.data.decide" {
                            Text("Score choice, ordinal and yes/no questions over private records with a verified typed-decision model. Decisions and all-option Scores are standard tables. Question revisions, state mapping and calibration stay captured; no question executes a tool or grants permission to write.")
                            Text("Top-option confidence is not the probability that an action is safe. Uncalibrated or unsupported decisions require review. Fit calibration in Train using separate source groups for selection and audit; insufficient audit evidence keeps review required. Evaluate uses the frozen calibration. Scores retains every option.")
                            Text("The encoder and decision head have separate token limits. Inputs must fit both limits; oversized records are rejected rather than silently truncated.")
                        } else if node.operatorID == "kwip.data.classifyAudio" {
                            Text("Prepare media supplies mono float32 waveform at 16,000 Hz. Load model provides the verified AST processor and class order; generic spectra cannot replace its processor.")
                            Text("Windows retain source-time coverage, including overlap. Padding never claims additional source audio, and window boundaries are not exact sound-event boundaries. Scores are independent sigmoid values, not calibrated certainty; reporting limits never remove classes from Scores.")
                            Text("Predictions retains source columns and the top label per window. Scores retains every class, rank, reporting status and captured audio. Connect either table to View or Table Output. Existing run time, memory and storage limits apply.")
                            Text(AudioClassificationLessonContent.boundary)
                        } else if node.operatorID == "kwip.data.transcribeAudio" {
                            Text("Prepare media supplies mono float32 waveform at 16,000 Hz. Load model provides the verified Whisper processor and tokenizer. Generic spectra are for analysis, not this model input.")
                            Text("Chunks run sequentially with source offsets. Boundaries can split phrases. Model text and timing can be wrong; compare results with the original recording. Transcripts retains one row per observation; Segments retains text, source times and captured audio. Connect either table to View or Table Output.")
                            Text(SpeechTranscriptionLessonContent.boundary)
                        } else if node.operatorID == "kwip.data.extractText" {
                            Text("Extract printed English text from captured images using a verified optional CPU pack. Documents is grouped text; Regions is a complete table of recognized text, model confidence and source-image pixel coordinates. Review flags preserve text. Apply saves settings; Run Flow performs OCR. Downloads are separate in Settings → Models & assets. View and History never rerun OCR.")
                            Text("Start with a sharp original. Detection magnification enlarges the detector input up to its maximum side; it does not add image detail. Small print, spaced headings and table punctuation can be misread. Confidence flags are review aids, not accuracy scores. Inspect Regions against the source before relying on extracted values. For a PDF with selectable text, use File Input’s PDF text reader instead of rasterizing it for OCR.")
                        } else if node.operatorID == "kwip.data.prepareMedia" {
                            Text("Prepare captured images or PCM WAV as tensors. Original media remains retained. Apply saves settings; Run Flow prepares the selected population. Image size, color, alpha and scaling changes are explicit. Audio windows retain sample offsets, valid masks and original observation identity. Partition files before window expansion; no fitted normalization, speech model or hidden download is added.")
                        } else if node.operatorID == "kwip.data.sourceFiles" {
                            FileInputInformationView()
                        } else if node.operatorID == "kwip.data.outputDelimited" {
                            TableOutputInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.sort" {
                            SortInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.unique" {
                            UniqueInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.reshapeTable" {
                            ReshapeTableInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.splitText" {
                            SplitTextInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.union" {
                            UnionInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.aggregate" {
                            SummarizeInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.join" {
                            Text("Pair exact scalar columns from Left and Right. Every pair must match; paired types must agree. Use Columns for explicit casts. Inner keeps matches; Left/Right retain all rows on that side; Full retains both sides. Unmatched fields are missing values. Missing keys can never match, match each other, or stop the run.")
                            Text("Use Keep, output names and grips/arrows to select, rename and order columns. Matching keys remain usable even when excluded from output. Search filters the grid; Select all and Clear affect every row. Names apply together, so swaps are supported. Image columns can pass through but cannot be matching keys. Prefixes apply to overlapping source names and update unchanged default names; custom names stay intact.")
                            Text("The key relationship checks uniqueness before expansion. Repeated keys can multiply rows; the maximum output row, cell and serialized-size guards are checked before materializing. Join currently accepts at most 100,000 rows per input and output. An empty result stops the run. Join sources before Partition or fitted preprocessing to preserve evaluation isolation.")
                            Text("New source columns can be included unchanged, excluded, or require explicit grid choices. Missing configured columns require repair. Apply saves the entire draft as one undoable change; Revert restores saved settings. Existing projects keep their original column order and collision names until edited. Input indicators show known schemas, not proof of current source validity.")
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.sampleRows" {
                            Text("Choose a reproducible subset without replacement, preserving source row order. Increasing the count with the same seed keeps the smaller subset. Rows to retain must be between 1 and 100,000 and cannot exceed the available input rows. Wide samples also have an in-memory cell limit.")
                            Text("Place before Partition to reduce its population, or on a training branch to keep held-out data fixed. Sampling is not stratified and may omit rare classes. It operates on the incoming table after Working data is applied; it does not reduce the source reader’s work. Change the seed to choose a different subset.")
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.preprocess" {
                            PreprocessInformationView()
                        } else if node.operatorID == "kwip.data.formula" {
                            FormulaInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.partition" {
                            PartitionInformationView()
                            DataSchemaDisclosureView(disclosure:context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.filterRows" {
                            Text("Choose Simple conditions, a typed Expression, or a Text pattern. Apply saves one undoable draft; Revert restores saved controls. All/Any combines conditions using three-valued logic: missing predicates use the chosen policy only when the final result is unknown. Is missing and Is not missing test directly.")
                            Text("Number, Text and Boolean values are explicit; Filter does not cast columns. Membership values use one line per value. Text case is chosen per condition. Expressions use the bounded column/operator language, never arbitrary Python. Pattern flags follow the supervised reader's syntax and resource limits.")
                            Text("Matching and nonmatching are complementary table outputs preserving source order, row identities, image content and existing partition lineage. They do not create training/validation roles. Results displays captured tables; Diagnostics identifies attributed failures. Syntax and observed data constraints are checked during execution.")
                            DataSchemaDisclosureView(disclosure: context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.columns" {
                            Text("Use Keep to select columns, edit output names and types in the grid, and drag a grip or use the arrows to order rows. Apply saves the whole draft as one undoable edit; Revert restores saved choices. Search filters the grid, while Select all and Clear affect every row.")
                            Text("Keep preserves the original type and values. Image retains an existing captured image value; it never decodes text into pixels. Trimming, missing-value and cast-failure policies affect explicit scalar conversions only. Stop run rejects invalid conversions; Use missing value records a null instead. Image values require an image adapter and cannot be cast to scalar types.")
                            Text("New source columns can pass through unchanged after configured columns, be excluded, or require explicit grid choices. Missing configured columns and duplicate output names require correction. Name swaps apply simultaneously. Retired Select / Name and Cast settings require an exact project upgrade; incomplete input-schema evidence requires explicit repair.")
                            DataSchemaDisclosureView(disclosure: context.schemaDisclosure)
                        } else if node.operatorID == "kwip.data.view" {
                            Text("View requests complete upstream data. Run Flow, then explore all captured rows in Results → Output. Search and sort operate on the captured table; they do not reread the original source. Double-click an overflowing scalar cell to inspect its captured text above the table. Long text uses bounded pages with character coverage; Copy page copies only the displayed portion. Close dismisses the detail area; Escape closes it while its text area is focused.")
                            DataSchemaDisclosureView(disclosure: context.schemaDisclosure)
                        } else { Text("Column choices follow incoming connectors. Observed columns are the names and types actually read by a run. If the source or flow changes, earlier observations become stale and cannot silently repair a missing column mapping.") }

                    }.id(context.contextIdentity)
                    if showsOutputs { DataNodeOutputsView(context:context) }
                } else { Text(context.usesFixtures ? "Select a block to configure its columns and connections." : "Select a block to configure it, inspect outputs, or choose its local source.").font(.caption).foregroundStyle(MLTheme.muted) }
            }.padding(16)
        }
    }
    @ViewBuilder private func configuration(_ node: DataFlowNode) -> some View {
        switch node.operatorID {
        case "kwip.data.sourceCSV":
            if context.usesFixtures {
                Text("Fixture observations").font(.caption.weight(.semibold))
                Text("The lesson adapter supplies these columns as strings. Cast them to numeric values before fitting statistics.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(context.fixtureColumns,id: \.name) { column in
                    HStack { Text(column.name);Spacer();Text(column.dtype).foregroundStyle(MLTheme.muted) }.font(.system(size:11,design:.monospaced))
                }
            } else { DataFlowSourceConfigurationView(context: context, node: node) }
        case GameRuleConfiguration.operatorID:
            GameRuleConfigurationView(context:context,node:node)
        case "kwip.data.decide":DecisionConfigurationView(context:context,node:node)
        case "kwip.data.classifyAudio":
            AudioClassificationConfigurationView(context:context,node:node)
        case "kwip.data.transcribeAudio":
            SpeechTranscriptionConfigurationView(context:context,node:node)
        case "kwip.data.extractText":
            OCRConfigurationView(context:context,node:node)
        case "kwip.data.prepareMedia":
            PrepareMediaConfigurationView(context:context,node:node)
        case "kwip.data.sourceFiles":
            FileInputConfigurationView(context: context, node: node)
        case "kwip.data.view":
            EmptyView()
        case "kwip.data.preprocess":
            PreprocessConfigurationView(context:context,node:node)
        case "kwip.data.sequenceWindows": SequenceWindowsConfigurationView(context:context,node:node)
        case "kwip.data.textEncode":
            TextEncodingConfigurationView(context:context,node:node)
        case "kwip.data.formula":
            FormulaConfigurationView(context:context,node:node)
        case "kwip.data.partition":
            PartitionToolConfigurationView(context:context,node:node)
        case "kwip.data.outputDelimited":
            TableOutputConfigurationView(context:context,node:node)
        case "kwip.data.sort":
            SortConfigurationView(context:context,node:node)
        case "kwip.data.unique":
            UniqueConfigurationView(context:context,node:node)
        case "kwip.data.reshapeTable":
            ReshapeTableConfigurationView(context:context,node:node)
        case "kwip.data.splitText":
            SplitTextConfigurationView(context:context,node:node)
        case "kwip.data.union":
            UnionConfigurationView(context:context,node:node)
        case "kwip.data.aggregate":
            AggregateConfigurationView(context:context,node:node)
        case "kwip.data.join":
            JoinConfigurationView(context:context,node:node)
        case "kwip.data.filterRows":
            FilterRowsConfigurationView(context: context, node: node)
        case "kwip.data.columns":
            ColumnsConfigurationView(context: context, node: node)
        case "kwip.data.sampleRows":
            SampleRowsConfigurationView(context:context)
        default:
            if RetiredDataOperatorMigration.retired.contains(node.operatorID) {
                Text("This retired tool requires an exact project upgrade or explicit repair with current tools. Its original settings and connections remain preserved. Configuration and execution are unavailable until repair is complete.").font(.caption).foregroundStyle(MLTheme.rose)
            } else {EmptyView()}
        }
    }

}


/// Shared output presentation; the universal workspace hosts this in its bottom dock.
struct DataNodeOutputsView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    var executionBusy=false
    @State private var apiTab="table"
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if context.node?.operatorID=="kwip.data.sourceCSV",context.node?.config["format"] == .string("api") {
                HStack {Button("Table") {apiTab="table"}.buttonStyle(FlatButtonStyle(prominent:apiTab=="table"));Button("Fetch receipt") {apiTab="receipt"}.buttonStyle(FlatButtonStyle(prominent:apiTab=="receipt"));Spacer()}
                if apiTab=="receipt" {DatabaseReceiptViewport {if let receipt=context.apiFetchReceipt {Text("Complete API snapshot").fontWeight(.semibold);ForEach(["rows","pages","responseBytes","capturedAt"],id:\.self) {key in HStack {Text(key=="responseBytes" ? "Response bytes":key=="capturedAt" ? "Captured":key.capitalized).foregroundStyle(MLTheme.muted);Text(receipt[key]?.integerValue.map(String.init) ?? receipt[key]?.stringValue ?? "—").textSelection(.enabled)}};Text("Recorded capture; viewing this receipt never fetches the endpoint.").foregroundStyle(MLTheme.muted)}else {Text("Fetch a snapshot or refresh its local preview to inspect its recorded fetch receipt.").foregroundStyle(MLTheme.muted)}}}
                else if let outputs=context.outputs,outputs.keys.contains(where:{context.capturedTableViewSession(port:$0) != nil}) {CapturedNodeTableOutputsView(context:context,executionBusy:executionBusy)}else {TableInputOutputView(context:context)}
            } else if context.node?.operatorID=="kwip.data.outputDelimited",let config=context.node?.config,APITableSettings.isExternal(config),let store=context.databaseOutputs {
                DatabaseOutputResultsView(context:context,store:store,executionBusy:executionBusy)
            } else if !context.usesFixtures, let outputs=context.outputs, outputs.keys.contains(where:{context.capturedTableViewSession(port:$0) != nil}) {
                CapturedNodeTableOutputsView(context:context,executionBusy:executionBusy)
            } else if context.node?.operatorID == "kwip.data.view", !context.usesFixtures {
                if let session = context.fullTableViewSession { TableViewOutputView(session: session,executionBusy:executionBusy) }
                else { Text("Run this View to explore its complete recorded table.").font(.caption).foregroundStyle(MLTheme.muted) }
            } else if (["kwip.data.partition","kwip.data.formula","kwip.data.sampleRows"].contains(context.node?.operatorID ?? "") || (context.node?.operatorID == "kwip.data.preprocess" && context.node?.config["mode"] != .string("table"))), !context.usesFixtures {
                if let outputs=context.outputs,!outputs.isEmpty {CapturedNodeTableOutputsView(context:context,executionBusy:executionBusy)}
                else {Text("Run this tool to explore its captured output tables.").font(.caption).foregroundStyle(MLTheme.muted)}
            } else if ["kwip.data.sourceCSV", "kwip.data.sourceFiles", "kwip.data.columns", "kwip.data.filterRows", "kwip.data.join", "kwip.data.aggregate", "kwip.data.union", "kwip.data.sort", "kwip.data.unique", "kwip.data.reshapeTable", "kwip.data.splitText", "kwip.data.preprocess"].contains(context.node?.operatorID ?? ""), !context.usesFixtures {
                TableInputOutputView(context: context)
            } else {
            Text("Outputs · \(context.outputStatus)").font(.system(size:12,weight:.semibold))
            if let disclosure=context.outputDisclosure {
                ResultsSection(title:"Captured request",symbol:"doc.text") {
                    Text(disclosure).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                }.font(.caption)
            }
            if let outputs=context.outputs {
                ForEach(outputs.keys.sorted(),id:\.self) {port in
                    if let session=context.capturedTableViewSession(port:port) { TableViewOutputView(session:session,executionBusy:executionBusy) }
                    else { DataPortPreviewView(port:port,output:outputs[port] as? [String:Any] ?? [:], imageResolver: context.imageArtifactResolver) }
                }
            } else {Text(context.emptyOutputMessage).font(.caption).foregroundStyle(MLTheme.muted)}
            }
        }
    }
}

/// One selected port at a time avoids concurrent paging requests and keeps state outputs accessible.
private struct CapturedNodeTableOutputsView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    var executionBusy=false
    @State private var selectedPort:String?
    var body:some View {
        let order=["training","validation","test","exclusions","state"]
        let ports=(context.outputs ?? [:]).keys.sorted {a,b in
            if ["kwip.data.partition","kwip.data.preprocess"].contains(context.node?.operatorID ?? "") {return (order.firstIndex(of:a) ?? 4) < (order.firstIndex(of:b) ?? 4)}
            return a < b
        }
        let active=selectedPort.flatMap { ports.contains($0) ? $0:nil } ?? (ports.contains("table") ? "table":ports.first)
        VStack(alignment:.leading,spacing:8) {
            if ports.count > 1 {
                FlatChoiceControl(title:"Output port",selection:active ?? "",options:ports.map {.init(id:$0,title:$0 == "state" && context.node?.operatorID == "kwip.data.preprocess" ? "Learned preprocessing":["kwip.data.partition","kwip.data.preprocess"].contains(context.node?.operatorID ?? "") ? $0.capitalized:$0)}) { selectedPort=$0 }
            }
            if let active,let session=context.capturedTableViewSession(port:active) {
                TableViewOutputView(session:session,executionBusy:executionBusy).id(ObjectIdentifier(session))
            } else if let active {
                let output=context.outputs?[active] as? [String:Any] ?? [:]
                if let preview=output["preview"] as? [String:Any],preview["columns"] is [[String:Any]],preview["rows"] is [[Any]] {
                    CapturedDataGridView(grid:.init(preview:preview),tableOnly:true,
                        evidenceLabel:(context.outputStatus.lowercased().contains("partial") ? "Partial · ":"")+(context.outputIsCurrent ? "Captured preview":"Out of date · captured preview"),imageResolver:context.imageArtifactResolver)
                } else {DataPortPreviewView(port:active,output:output,imageResolver:context.imageArtifactResolver)}
            }
        }.onChange(of:context.contextIdentity) {_,_ in selectedPort=nil}
    }
}
