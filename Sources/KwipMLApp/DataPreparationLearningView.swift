import SwiftUI
import KwipMLCore

/// Reading and editable canvas practice, deliberately separate from scored lesson receipts.
enum DataPreparationLessonContent {
    struct Topic: Identifiable {
        let id: String
        let title: String
        let objective: String
        let terms: String
        let steps: [String]
        let inspect: String
        let operators: [String]
        let exampleIDs: [String]
        var isAssessed: Bool { false }
    }
    static let version = 7
    static let authoredAt = "2026-10-18"
    static let disclosure = "Editable practice, not scored assessment. Open a sample, change its components and inspect measured results after an actual run. Reading these steps or opening a sample does not award completion. Examples are starting points; they do not contain every tool described below."
    static var operatorCoverage: Set<String> { Set(topics.flatMap(\.operators)) }
    static let topics: [Topic] = [
        Topic(id: "read-tables", title: "1 · Read a source as a table",
              objective: "Choose a reader that understands the source format, then check the actual columns and values before connecting preparation tools.",
              terms: "**Reader:** the tool that turns one source into rows and named columns. **Parsing:** interpreting delimiters, records or stored file structure. **Null:** a missing value, which differs from empty text. **Source snapshot:** the bytes or database rows captured for a particular inspection or run.",
              steps: [
                "Open table exploration practice. It contains Table Input connected directly to View, with 64 bundled synthetic observations. Select Table Input and open Results → Output to inspect its bounded source preview; an unverified file link is not a completed read.",
                "CSV / delimited text uses a separator between fields and quoting around fields that contain that separator. Check the comma delimiter and first-record header in this example. Choose the reader profile before adjusting its options: native Polars has a one-byte delimiter/quote and doubled quote escaping; compatibility parsing provides custom escaping, other encodings and record-ending rules.",
                "Decide what an empty field and each explicit null marker mean. Null markers are case-sensitive. Apply & preview saves changes and checks a new sample. A successful source preview does not validate every later record or establish a full row count.",
                "For your own source, choose Parquet, JSON document, JSON Lines, Excel, Markdown table, XML records, HTML table or Database. Use that format's selection, schema and read options. Spreadsheet formulas are not recalculated; nested JSON/XML is not automatically flattened. Database access uses an exact selected table with private credentials, not arbitrary workflow SQL. Read the format-specific Information before importing unfamiliar data.",
                "Keep source parsing separate from preparation. Table Input reads one source; downstream Columns, Filter and Join tables make conversions, row choices and combinations explicit. Image references in a table remain references rather than silently loading another kind of data."
              ],
              inspect: "Run Flow and inspect Table Input and View. This example has signal, context and label columns and 64 actual records. Compare leading zeros, empty text, null indicators and value types when trying another source. A configured source row limit intentionally restricts the table even when a downstream consumer requests complete data. Reading errors belong in Diagnostics; changing a setting is not evidence that the error has been resolved until another inspection or run succeeds.",
              operators: ["kwip.data.sourceCSV"], exampleIDs: ["table-view"]),
        Topic(id: "clean-and-check", title: "2 · Make the table usable",
              objective: "Choose useful columns, decide what missing values mean and check which rows survive your choices.",
              terms: "**Observation:** one row, such as one purchase or measurement. **Feature:** a value the model will use to make a prediction. **Target:** the answer you want it to learn to predict. **Schema:** the column names and the kind of value each column accepts.",
              steps: [
                "Open the prepared example. Select Table Input and inspect its column names. This example uses CSV text; other formats and explicit CSV schema/inference settings can already produce typed numbers or Boolean values. Check the observed schema instead of assuming every source is text.",
                "Open Columns practice to select, order, rename and cast in one grid. Keep selects columns, row order sets output column order, Output name renames and Type converts values. Keep identifiers as text when leading zeros matter.",
                "Use Preprocess in Fixed table rules mode to reject missing cells, exclude affected rows or fill them with a typed constant. Formula coalesce can express custom replacements; Filter can route missing rows. A fixed value is different from a statistic learned from training rows. Check how many rows remain.",
                "Open Filter practice to choose simple conditions, an expression or a text pattern. Matching and Nonmatching remain separate output tables in every mode; inspect both branches and choose an explicit missing-value policy. Sample rows selects a smaller count for a quick trial; its seed is the starting number that makes the selection repeatable. Use the same seed when comparing two changes."
              ],
              inspect: "Run Flow, then select the source or a changed component and open Results → Output. For a specific output port, open Results → Ports & connections and choose Inspect beside that port. Compare column names, value types, row counts and a few actual cells. A filter may remove rows; a cast changes their value type. A small sample is useful for iteration but is not proof that the full dataset will behave identically.",
              operators: ["kwip.data.formula", "kwip.data.columns", "kwip.data.filterRows", "kwip.data.preprocess", "kwip.data.sampleRows"], exampleIDs: [ColumnsPracticeExample.exampleID, FilterPracticeExample.exampleID, "prepared"]),
        Topic(id: "explore-complete-table", title: "3 · Explore complete data with View",
              objective: "Inspect a captured table without confusing a small working sample with the complete table used by a demanding consumer.",
              terms: "**Working data:** the canvas's development sampling policy. **Prefix sample:** the first records in source order, which can miss later groups or classes. **Seeded sample:** a reproducible subset chosen by Sample rows. **Page:** a bounded set of rows loaded into the viewer from an already captured table.",
              steps: [
                "Open table exploration practice and Run Flow. Select View, then Results → Output. Its table viewer reads bounded pages from the retained capture; it does not reload a mutable source file or hold every row in the interface.",
                "Use Find in table to search the captured values. Clear the search and use the previous/next page arrows when a result has more rows than one page. A viewer search changes the displayed rows; it does not filter the table passed to a downstream component. Add Filter when that is your intended workflow operation.",
                "Change Working data to Record count and apply four records, then run again. View requests complete upstream data for its branch, so this example still contains all 64 records. A file output or training consumer likewise requests complete dependencies. Complete means all data permitted by the explicit source settings and resource budget, not unlimited memory or access to another source.",
                "For comparison, remove View from this editable example, leaving Table Input, and run with the same working-data count. The branch can now use a source-order prefix. Inspect the recorded coverage indicator rather than inferring completeness from the currently displayed table page. Reopen the example to restore its original two-tool flow.",
                "Sample rows is different: it deliberately changes the downstream dataset to a seeded subset without replacement. Adding View after Sample rows explores that selected subset; it does not undo your sampling component. Canvas percentage sampling is also a prefix, may require a cardinality scan and is not a claim of statistical representativeness."
              ],
              inspect: "The source and View should report 64 captured records in the original example, even with a four-record working-data policy. The source-only comparison should be marked sampled when it uses fewer records. Find affects visible matches, not the recorded table's total or provenance. Full data requests can take longer and reach a resource limit; a refusal is preferable to silently presenting a partial table as complete. This editable practice has no scored pass or mastery award.",
              operators: ["kwip.data.view"], exampleIDs: ["table-view"]),
        Topic(id: "combine-and-summarize", title: "4 · Combine and summarize observations",
              objective: "Combine related tables without unexpectedly multiplying rows, then summarize repeated measurements at the right level.",
              terms: "**Key:** a column, or set of columns, used to match related records. **Join:** matching rows from two tables using those keys. **Aggregate:** making one summary row from several rows, such as their count, total or average. **Mean:** the total divided by the number of values.",
              steps: [
                "Open the joined example. Select Join tables. In Matching keys, pair one left column with one right column; add another pair when both fields must agree. Types must match: use Columns for casts. Predict which records should match before running.",
                "Choose whether unmatched rows should be retained. The Key relationship setting checks which side must have unique keys. For example, one-to-one expects one row on each side; repeated keys on both sides can multiply the output.",
                "In Join’s Output columns grid, clear an unneeded duplicate key, rename a kept column, and use its grip or arrows to change output order. Find column searches the grid; Select all and Clear affect every row. Apply saves one undoable edit; Revert restores saved choices. Check Results → Output after running. Excluding a key from output does not change matching. Keep image columns unchanged and choose scalar metadata for matching keys.",
                "Open the aggregation example. Select Summarize and inspect its Group by and Aggregations grids. Count rows counts measurements; mean calculates their average. Decide how missing values should affect each summary. For order-line text, use Join text with the Newline separator instead of Sum; open Summarize practice for a dedicated example.",
                "Run the example and check its label summaries as well as its feature summaries. Minimum and maximum labels can reveal conflicting answers within a group; the summary tool does not automatically repair those conflicts."
              ],
              inspect: "After running, select the components before and after Join tables and open Results → Output to compare row counts and matched keys. In the bundled aggregation example, 192 measurements become 64 subject summaries. Those numbers describe this sample only; inspect your actual output for other data. Confirm the measurement-count column and compare minimum and maximum labels before trusting a summary.",
              operators: ["kwip.data.join", "kwip.data.aggregate"], exampleIDs: ["joined", "aggregated"]),
        Topic(id: "protect-splits", title: "5 · Keep evaluation data separate",
              objective: "Choose a split that tests the prediction you actually care about: new rows, new people or future observations.",
              terms: "**Training:** the rows used to adjust learned values. **Validation:** separate rows used to compare choices while developing the model. **Test:** rows reserved for the final check. **Leakage:** information from evaluation data slipping into training and making the result look better than it should.",
              steps: [
                "In the prepared example, select Partition. Its seed makes the training/validation assignment repeatable. A random row split is useful when rows are sufficiently independent.",
                "Open the subject holdout example when one person or entity has several measurements. Partition’s By entity mode keeps all rows for a chosen key together. A held-out person should not also appear in training.",
                "Open the chronological example when the task is to predict future observations. Partition’s By time mode uses your chosen cutoffs; its gap leaves space between periods to reduce unwanted overlap near a boundary.",
                "Run and inspect the actual partition outputs. Keep training, validation and test connections distinct. Splitting by time alone does not prove that repeated people or other related observations are independent."
              ],
              inspect: "Inspect each partition's row count and its relevant keys or timestamps. For an entity split, look for the same entity in more than one partition. For a time split, compare the dates with the configured cutoffs and gap. A held-out score answers only the question represented by this split; it is not a guarantee about every future population.",
              operators: ["kwip.data.partition"], exampleIDs: ["prepared", "group", "temporal"]),
        Topic(id: "learn-from-training", title: "6 · Learn preparation from training only",
              objective: "Learn reusable preparation rules from training rows, then apply those same rules to each separate evaluation partition.",
              terms: "**Fit:** learn values from the connected training rows. **Transform:** apply those saved values to a table. **Standard scaling:** express a value relative to the training average and spread. **Learned replacement:** a missing-value setting in Preprocess that learns from training rows. **Categorical encoding:** representing categories with separate columns containing zero or one.",
              steps: [
                "Open Preprocess practice. Follow the existing Training, Validation and Test connections from Partition into one Preprocess; each prepared output retains its own population.",
                "Configure missing replacements, encoding and scaling in its column grid. Learned statistics and categories come only from retained training rows and are applied unchanged to connected Validation and Test.",
                "Use Fixed table rules for constants, missing-value checks or row exclusion before Partition. Formula coalesce supports custom replacements; Filter can route missing rows.",
                "Try an unknown category under Stop run versus All-zero encoding. Review the attributed diagnostic, then restore the intended policy. Use one Preprocess so every prepared population uses the same training-owned state."

              ],
              inspect: "After an actual run, inspect transformed training and validation tables. Confirm both reference the same saved preparation state while retaining their own rows. Scaled training values may be centered near zero; validation values need not be. Check encoder output columns and missing-value behavior rather than assuming every newly added tool is already configured by the sample.",
              operators: ["kwip.data.preprocess"], exampleIDs: ["preprocess"]),
        Topic(id: "protect-final-test", title: "7 · Check the saved model on unseen data",
              objective: "Evaluate a chosen saved model on unseen test rows after fixing your model and preparation choices.",
              terms: "**Checkpoint:** saved learned values and the information needed to identify or continue that training run. **Saved checkpoint:** a verified completed model selected for prediction without updating it. **Accuracy:** the fraction of evaluated answers that match the targets; it does not explain every kind of error.",
              steps: [
                "Open the subject holdout or chronological example. Train with its training and validation connections, keeping the protected test branch separate.",
                "Select Saved checkpoint and choose the intended completed run in Configuration. Keep the test preparation connected to the state learned from training, rather than fitting a new preparation rule on test rows.",
                "After fixing your development choices, choose Test in Evaluate and run it once for the final check. Choose Validation instead for development checks. The application records protected Test exposure; repeatedly using this result to choose a model turns the test into another development set.",
                "Open History and compare the captured checkpoint identity, evaluation partition and preparation source. Describe what this result shows and what a new independent dataset would still need to test."
              ],
              inspect: "Results and History should identify the actual saved model and separate test evaluation. Compare the evaluated row count and recorded outputs; a high accuracy on very few rows can change sharply after a single wrong answer. If no compatible checkpoint or retained evidence exists, the result is unavailable rather than a guessed score. This practice does not award an assessment pass.",
              operators: ["kwip.data.partition", "kwip.data.preprocess"], exampleIDs: ["group", "temporal"]),
        Topic(id: "read-file-contents", title: "8 · Read files and their contents",
              objective: "Turn selected local files into observations while keeping file identity, extracted text and image pixels distinct.",
              terms: "**File collection:** the selected files captured for one run. **Reader:** the rules for extracting a file's content. **Image reference:** a verified captured pixel artifact, not a live path or a model-ready tensor.",
              steps: [
                "Open file exploration practice. File Input selects twelve synthetic files and connects to View. Choose a single file or directory; directory file types and discovery settings control which files enter the collection.",
                "Select the appropriate reader settings. Text can emit one row per file or line, documents can emit pages or paragraphs, and images can emit selected frames. Use metadata-only reading when you do not need content.",
                "Apply & preview checks a bounded sample. Images in this preview are metadata-only. Run Flow with View to capture complete data within the explicit source settings and resource limits, then open an image cell to inspect its verified pixels.",
                "Working-data sampling chooses files before extraction; one file can produce several rows. Reading pixels does not prepare them for model training. Keep later feature extraction and image-to-model conversion explicit."
              ],
              inspect: "The default practice produces twelve records. Switch text to line units and compare the observed row count, or exclude a file type and run again. Inspect reader status and Diagnostics for refusals. History retains captured text and pixels even after changing the live source. This practice is not scored and does not certify model readiness.",
              operators: ["kwip.data.sourceFiles"], exampleIDs: [FileInputViewPracticeExample.exampleID])
    ]
    static var railSteps:[LessonContentStep] {
        var result=[LessonContent.step("goal","Prepare a table before training", "A model learns from the table you give it. This course follows that table from a source through cleanup, row selection, combination, partitioning and preparation learned only from Training.\n\nThis course opens an empty lesson canvas. The examples live in Exercise choices & reference below. Open the named example when a step asks for it; different examples illustrate different parts of the flow. Save a canvas before replacing it.\n\nStart with Read a source as a table → Open table exploration practice. It contains Table Input → View and 64 synthetic signal/context/label records. No private file is required.")]
        for topic in topics {
            result.append(.init(id:topic.id+".goal",title:topic.title,blocks:[
                .init(id:topic.id+".purpose",kind:.explanation,title:"Understand the idea",text:topic.objective),
                .init(id:topic.id+".terms",kind:.detail,title:"Words to know",text:topic.terms)]))
            result += topic.steps.enumerated().map {index,text in
                LessonContent.step(topic.id+".try.\(index)",topic.title+" · \(index+1)",text,kind:.canvasAction)
            }
            result.append(LessonContent.step(topic.id+".inspect","Compare the actual result",topic.inspect,kind:.observation))
        }
        result.append(LessonContent.step("inspect","Explain your preparation choices","Which rules learned values from Training, and which were fixed choices? Explain why a held-out person's observations must stay separate. Name one source setting that could still restrict a complete-data consumer.\n\nThis is editable, unscored practice. A successful run is evidence about the actual selected data and settings, not an assessment pass.",kind:.question))
        return result
    }
    static func exampleTitle(_ id: String) -> String {
        switch id {
        case "filter-rows": "Open Filter practice"
        case "columns-grid": "Open Columns practice"
        case "table-view": "Open table exploration practice"
        case "file-view": "Open file exploration practice"
        case "preparation": "Open missing/category practice"
        case "preprocess": "Open Preprocess practice"
        case "joined": "Open joined example"
        case "aggregated": "Open aggregation example"
        case "group": "Open subject holdout example"
        case "temporal": "Open chronological example"
        default: "Open prepared example"
        }
    }
}

struct DataPreparationLearningView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            LessonTextPanel(title: "Data preparation · editable canvas practice", text: DataPreparationLessonContent.disclosure)
            ForEach(DataPreparationLessonContent.topics) { topic in
                VStack(alignment: .leading, spacing: 14) {
                    Text(topic.title).font(.system(size: 16, weight: .semibold))
                    if !inLessonRail {
                    LessonRichText(text: topic.objective)
                    LessonTextPanel(title: "Words to know", text: topic.terms)
                    LessonTextPanel(title: "Try it on the canvas", text: topic.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n"))
                    LessonTextPanel(title: "Inspect the actual output", text: topic.inspect)
                    }
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 8) { exampleButtons(topic) }
                        VStack(alignment: .leading, spacing: 8) { exampleButtons(topic) }
                    }
                }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                    .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    .accessibilityIdentifier("learning.data-preparation." + topic.id)
            }
            DataEngineeringSharedGuidanceView()
            Text("Content v\(DataPreparationLessonContent.version) · Authored \(DataPreparationLessonContent.authoredAt) · Editable practice · Results require execution")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    @ViewBuilder private func exampleButtons(_ topic: DataPreparationLessonContent.Topic) -> some View {
        ForEach(topic.exampleIDs, id: \.self) { id in
            Button(DataPreparationLessonContent.exampleTitle(id)) { openExample(id) }
                .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.data-preparation." + topic.id + "." + id)
        }
    }
    private func openExample(_ id: String) {
        switch id {
        case "filter-rows": _ = store.loadComponentWorkflowExample(FilterPracticeExample.exampleID, device: "cpu")
        case "columns-grid": _ = store.loadComponentWorkflowExample(ColumnsPracticeExample.exampleID, device: "cpu")
        case "table-view": _ = store.loadComponentWorkflowExample(TableInputViewPracticeExample.exampleID, device: "cpu")
        case "file-view": _ = store.loadComponentWorkflowExample(FileInputViewPracticeExample.exampleID, device: "cpu")
        case "preparation": _ = store.loadPreparationPracticeExample()
        case "preprocess": _ = store.loadComponentWorkflowExample(PreprocessPracticeExample.exampleID,device:"cpu")
        case "joined": _ = store.loadJoinedWorkflowExample()
        case "aggregated": _ = store.loadAggregatedWorkflowExample()
        case "group": _ = store.loadGroupSplitWorkflowExample()
        case "temporal": _ = store.loadTemporalSplitWorkflowExample()
        default: _ = store.loadPreparedWorkflowExample()
        }
    }
}
