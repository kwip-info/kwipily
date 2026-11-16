import SwiftUI

/// Shared bounded data output presentation for live inspectors and archived results.
struct DataPortPreviewView: View {
    let port: String
    let output: [String: Any]
    var imageResolver: FileImageArtifactResolver? = nil
    @ViewBuilder var body: some View {
        Text(port).font(.system(size: 11, weight: .semibold)).foregroundStyle(MLTheme.teal)
        if let preview = output["preview"] as? [String: Any] {
            if !(preview["columns"] is [[String: Any]] && preview["rows"] is [[Any]]) {
                Text((preview["rowCount"] as? Int).map { "\($0) rows" } ?? "Row count not recorded").font(.caption)
            }
            if let diagnostics = preview["diagnostics"] as? [[String: Any]], !diagnostics.isEmpty {
                ResultsSection(title:"Diagnostics · \(preview["diagnosticCount"] as? Int ?? diagnostics.count)") {
                    ForEach(Array(diagnostics.enumerated()), id: \.offset) { _, diagnostic in
                        if let join = JoinResultSummary(evidence: diagnostic) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("Join · \(join.outputRows) output rows").fontWeight(.semibold)
                                Text("Left: \(join.leftRows) rows · \(join.leftUnmatched) unmatched")
                                Text("Right: \(join.rightRows) rows · \(join.rightUnmatched) unmatched")
                                Text("\(join.matchedPairs) matched pairs · \(join.retainedUnmatchedRows) unmatched rows retained · \(join.discardedUnmatchedRows) excluded by join mode")
                            }.font(.caption).textSelection(.enabled)
                            ResultsSection(title:"Join evidence") { CapturedMetadataView(values:diagnostic) }
                        } else if let aggregate = AggregateResultSummary(evidence: diagnostic) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("Aggregate · \(aggregate.outputRows) output groups").fontWeight(.semibold)
                                Text("\(aggregate.inputRows) input rows · \(aggregate.retainedRows) retained · \(aggregate.droppedKeyRows) excluded for missing keys")
                                ForEach(Array(aggregate.reducers.enumerated()), id: \.offset) { _, reducer in
                                    Text("\(reducer.output) · \(reducer.nullInputRows) missing input values among retained rows")
                                }
                            }.font(.caption).textSelection(.enabled)
                            ResultsSection(title:"Aggregation evidence") { CapturedMetadataView(values:diagnostic) }
                        } else if let partition = PartitionResultSummary(evidence: diagnostic) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(partition.grouped ? "Entity partition · measured allocation" : partition.temporal ? "Time partition · measured allocation":"Random partition · measured allocation").fontWeight(.semibold)
                                Text("\(partition.inputRows) input rows · \(partition.excludedRows) excluded" + (partition.temporal ? " by embargo" : ""))
                                ForEach(Array(partition.partitions.enumerated()), id: \.offset) { _, item in
                                    Text("\(item.role.capitalized): \(item.rows) rows" + (item.groups.map {" · \($0) groups"} ?? ""))
                                }
                                Text("Test rows remain separate from training and validation. These counts do not report final test performance.").foregroundStyle(MLTheme.muted)
                            }.font(.caption).textSelection(.enabled)
                            ResultsSection(title:"Partition evidence") { CapturedMetadataView(values:diagnostic) }
                        } else {
                            CapturedMetadataView(values:diagnostic)
                        }
                    }
                }
            }
            if preview["columns"] is [[String: Any]], preview["rows"] is [[Any]] {
                CapturedDataGridView(grid: CapturedDataGrid(preview: preview), tableOnly: true, evidenceLabel: "Table", imageResolver: imageResolver)
                    .frame(minHeight: 180, maxHeight: .infinity)
            } else {
                ResultsSection(title:"Captured preview · unstructured") {
                    CapturedMetadataView(values:preview)
                }
            }
        }
        if let summary = output["summary"] as? [String: Any] {
            if summary["operator"] as? String == "kwip.data.preprocess" {PreprocessLearnedStateView(summary:summary)}
            else {ResultsSection(title:"Output metadata",initiallyExpanded:true) {CapturedMetadataView(values:summary)}}
        }
        if let lineage = output["lineage"] as? [String: Any] {
            ResultsSection(title:"Lineage") { CapturedMetadataView(values:lineage) }
        }
    }
    private func pretty(_ value: [String: Any]) -> String { (try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])).flatMap { String(data: $0, encoding: .utf8) } ?? "Unavailable" }
}
