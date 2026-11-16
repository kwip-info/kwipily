import SwiftUI
import AppKit
import KwipMLCore

struct BackendFitResultsView: View {
    @ObservedObject var session: BackendFitSession
    @State private var exportMessage = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ResultsSection(title:"Overview",symbol:"info.circle",initiallyExpanded:true) {
                ViewThatFits(in:.horizontal) {
                    HStack(alignment:.top) {status;Spacer();runningControls}
                    VStack(alignment:.leading,spacing:8) {status;runningControls}
                }
                Text("\(session.plan.configuration.backendID) · CPU · captured canvas revision \(session.document.revision)")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            }
            if let manifest = session.manifest, case .object(let evaluation) = manifest["evaluation"] {
                if manifest["task"] == .string("prebuilt") {
                    PrebuiltResultsMetrics(manifest:manifest)
                } else if manifest["task"] == .string("documentClassification") {DocumentClassificationResults(manifest:manifest)} else {
                ResultsSection(title:"Metrics",symbol:"chart.bar",initiallyExpanded:true) {
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:140),alignment:.leading)],alignment:.leading,spacing:12) {
                        metric("Accuracy", evaluation["accuracy"])
                        metric("Log loss", evaluation["logLoss"] ?? evaluation["loss"])
                        metric("Validation rows", evaluation["count"] ?? evaluation["samples"])
                    }
                }
                ResultsSection(title:"Predictions",symbol:"tablecells") {
                    if case .object(let prediction) = manifest["predictions"], case .array(let classes) = prediction["classes"] {
                        let scores: [JSONValue] = { if case .array(let values) = prediction["decisionScores"] { return values }; if case .array(let values) = prediction["logits"] { return values }; return [] }()
                        Text("Captured validation sample · first \(classes.count) rows, not the full evaluation").font(.caption2).foregroundStyle(MLTheme.muted)
                        ThemedScrollView(.horizontal) {
                            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
                                GridRow { Text("Row"); Text("Class"); Text("Score / logits") }.font(.caption.weight(.semibold))
                                ForEach(classes.indices, id: \.self) { index in
                                    GridRow { Text(String(index + 1)); Text(render(classes[index])); Text(index < scores.count ? render(scores[index]) : "—") }
                                }
                            }.font(.system(size: 11, design: .monospaced)).textSelection(.enabled).padding(.vertical,6)
                        }
                    } else {
                        Text("Prediction sample capture was disabled. Aggregate metrics still evaluate the full validation partition.").font(.caption2).foregroundStyle(MLTheme.muted)
                    }
                }
                }
                ResultsSection(title:"Artifacts",symbol:"archivebox") {
                    if session.output != nil,session.manifest?["configuration"].flatMap({if case .object(let c)=$0{return c["algorithm"]};return nil}) != .string("dbscan") {
                        Button("Reveal portable model package") {
                            exportMessage = "Verifying saved output…"
                            Task {
                                do {
                                    let output = try await session.verifiedOutput()
                                    NSWorkspace.shared.activateFileViewerSelecting([output.appendingPathComponent("portable")])
                                    exportMessage = "Verified portable model package."
                                } catch { exportMessage = error.localizedDescription }
                            }
                        }.buttonStyle(FlatButtonStyle())
                        if !exportMessage.isEmpty { Text(exportMessage).font(.caption2).foregroundStyle(MLTheme.muted) }
                    } else {
                        Text("No predictive model package is available. DBSCAN analysis remains in complete output tables and History.").font(.caption2).foregroundStyle(MLTheme.muted)
                    }
                }
            } else {
                Text("No evaluation metrics are available for this fit yet.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ResultsSection(title:"Messages",symbol:"text.alignleft") {
                if session.messages.isEmpty { Text("No captured backend messages.").font(.caption2).foregroundStyle(MLTheme.muted) }
                else {
                    Text("Latest 12 captured messages").font(.caption2).foregroundStyle(MLTheme.muted)
                    ForEach(Array(session.messages.suffix(12).enumerated()), id: \.offset) { _, message in
                        Text(message).font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                            .padding(.vertical,6).frame(maxWidth:.infinity,alignment:.leading)
                            .overlay(alignment:.bottom){Rectangle().fill(MLTheme.rule).frame(height:1)}
                    }
                }
            }
        }
    }
    private var status:some View {
        Text(session.status).font(.caption.weight(.semibold)).textSelection(.enabled)
    }
    @ViewBuilder private var runningControls:some View {
        if session.isRunning {
            HStack {ProgressView().controlSize(.small);Button("Cancel fit") {session.cancel()}.buttonStyle(FlatButtonStyle())}
        }
    }
    private func metric(_ title: String, _ value: JSONValue?) -> some View {
        VStack(alignment: .leading, spacing: 3) { Text(title).font(.caption2).foregroundStyle(MLTheme.muted); Text(value.map(render) ?? "—").font(.system(size: 13, design: .monospaced)) }
    }
    private func render(_ value: JSONValue) -> String {
        switch value { case .number(let v): return String(format: "%.6g", v); case .integer(let v): return String(v); case .array(let values): return values.map(render).joined(separator: ", "); default: return value.stringValue ?? "—" }
    }
}
