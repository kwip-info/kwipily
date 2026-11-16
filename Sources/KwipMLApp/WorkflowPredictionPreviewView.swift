import SwiftUI
import KwipMLCore

struct WorkflowPredictionPreviewView: View {
    @ObservedObject var execution: WorkflowExecutionSession
    var body: some View {
        if execution.phase == .completed,let preview=execution.bagPredictionPreview {
            BagResultSectionsView(preview:preview,evaluation:execution.objectiveEvaluation,provenance:"Current session result · request \(execution.request.requestIdentity.prefix(12))").id(execution.runID)
        } else if execution.phase == .completed,let preview=execution.objectivePredictionPreview,let evaluation=execution.objectiveEvaluation {
            ObjectiveResultSectionsView(preview:preview,evaluation:evaluation,provenance:"Current session result · request \(execution.request.requestIdentity.prefix(12))").id(execution.runID)
        } else if execution.phase == .completed, let preview = execution.predictionPreview {
            PredictionPreviewTable(preview: preview, provenance: "Current session preview").id(execution.runID)
        }
    }
}

struct PredictionPreviewTable: View {
    let preview: PredictionPreview
    let provenance: String
    @State private var rowPage = 0
    @State private var columnPage = 0

    var body: some View {
            VStack(alignment: .leading, spacing: 8) {
                Text("Held-out predictions").font(.caption.weight(.semibold))
                Text("\(provenance) · \(preview.rows.count) of \(preview.evaluationCount) evaluated rows · \(preview.dtype) · sample \(preview.sampleShape.map(String.init).joined(separator: " × "))")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
                if preview.captureStatus == .omittedByPolicy {
                    Text("Not captured by this run’s policy. Aggregate metrics include all evaluated rows.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                } else {
                Text("Input headings are tensor coordinates. Labels and predictions are class IDs.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
                let columns = Array(preview.coordinates.indices.dropFirst(columnPage * 8).prefix(8))
                let rows = Array(preview.rows.indices.dropFirst(rowPage * 20).prefix(20))
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 5) {
                    GridRow {
                        Text("Row")
                        ForEach(columns, id: \.self) { column in Text(preview.coordinates[column]) }
                        Text("Expected"); Text("Predicted"); Text("Match")
                    }.font(.caption.weight(.semibold))
                    ForEach(rows, id: \.self) { index in
                        let row = preview.rows[index]
                        GridRow {
                            Text(String(index + 1))
                            ForEach(columns, id: \.self) { column in
                                Text(row.inputs[column])
                                    .accessibilityLabel("Preview row \(index + 1), input \(preview.coordinates[column]), \(row.inputs[column])")
                            }
                            Text(String(row.label)); Text(String(row.predicted))
                            Text(row.label == row.predicted ? "Yes" : "No")
                                .foregroundStyle(row.label == row.predicted ? MLTheme.muted : MLTheme.rose)
                        }
                    }
                }.font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                HStack {
                    Button("Previous rows") { rowPage -= 1 }.accessibilityLabel("Previous prediction rows").disabled(rowPage == 0)
                    Text("\(rowPage * 20 + 1)–\(min((rowPage + 1) * 20, preview.rows.count)) of \(preview.rows.count)")
                    Button("Next rows") { rowPage += 1 }.accessibilityLabel("Next prediction rows").disabled((rowPage + 1) * 20 >= preview.rows.count)
                    if preview.coordinates.count > 8 {
                        Divider().frame(height: 16)
                        Button("Previous inputs") { columnPage -= 1 }.accessibilityLabel("Previous prediction input columns").disabled(columnPage == 0)
                        Text("Inputs \(columnPage * 8 + 1)–\(min((columnPage + 1) * 8, preview.coordinates.count)) of \(preview.coordinates.count)")
                        Button("Next inputs") { columnPage += 1 }.accessibilityLabel("Next prediction input columns").disabled((columnPage + 1) * 8 >= preview.coordinates.count)
                    }
                }.font(.caption).buttonStyle(FlatButtonStyle())
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityElement(children: .contain)
                .accessibilityLabel("Held-out prediction table")
                .onChange(of: preview) { _, _ in rowPage = 0; columnPage = 0 }
    }
}
