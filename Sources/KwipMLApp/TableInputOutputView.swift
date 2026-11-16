import SwiftUI

/// One evidence surface: current captured output, otherwise a verified preview,
/// otherwise the old captured table explicitly marked out of date.
struct TableInputOutputPresentation {
    let grid: CapturedDataGrid
    let label: String
    init(run: [String: Any]?, runIsCurrent: Bool, preview: SourceInspectionResult?, retainedPreview: SourceInspectionResult? = nil, preferNewerRun: Bool = false) {
        if let run, runIsCurrent, preferNewerRun {
            grid = CapturedDataGrid(preview: run); label = "Run output"
        } else if let preview, preview.parsing?.sampleOnly == true {
            // A refreshed prefix cannot certify that a previous full run is still current.
            grid = DataSourcePreviewPresentation(result: preview).grid; label = Self.previewLabel(preview)
        } else if let run, runIsCurrent {
            grid = CapturedDataGrid(preview: run); label = "Run output"
        } else if let preview {
            grid = DataSourcePreviewPresentation(result: preview).grid; label = Self.previewLabel(preview)
        } else if let run {
            grid = CapturedDataGrid(preview: run); label = "Out of date · run output"
        } else if let retainedPreview {
            grid = DataSourcePreviewPresentation(result: retainedPreview).grid; label = "Out of date · " + Self.previewLabel(retainedPreview).lowercased()
        } else {
            grid = CapturedDataGrid(preview: [:]); label = "Preview"
        }
    }
    private static func previewLabel(_ preview: SourceInspectionResult?) -> String {
        guard preview?.collection?.readMode == "content" else { return "Preview" }
        return "Preview · images are metadata"
    }
}

struct TableInputOutputView<Context: DataNodeEditing>: View {
    @ObservedObject var context: Context
    @State private var retainedPreview: SourceInspectionResult?
    @State private var retainedNodeID: String?
    var body: some View {
        let output = context.outputs?["table"] as? [String: Any]
        let currentPreview = context.sourceInspectionPreview
        let runDigest = (output?["lineage"] as? [String: Any])?["sourceDigest"] as? String
        let newerRun = context.outputCapturedAt.map { $0 > (context.sourcePreviewCompletedAt ?? .distantPast) } ?? false
        let digestMatches = currentPreview == nil || runDigest == currentPreview?.sourceSHA256 || (newerRun && currentPreview?.parsing?.sampleOnly == true)
        let presentation = TableInputOutputPresentation(run: output?["preview"] as? [String: Any], runIsCurrent: context.outputIsCurrent && digestMatches, preview: currentPreview, retainedPreview: retainedNodeID == context.node?.id ? retainedPreview : nil, preferNewerRun: newerRun)
        VStack(alignment: .leading, spacing: 8) {
            if presentation.label != "Run output", let status = context.sourceInspectionStatus, status.isProblem {
                Label("Source inspection failed. See Diagnostics for details.", systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(MLTheme.rose)
            }
            if presentation.grid.columns.isEmpty {
                Text(["kwip.data.columns", "kwip.data.filterRows", "kwip.data.join", "kwip.data.aggregate", "kwip.data.union", "kwip.data.sort", "kwip.data.unique", "kwip.data.reshapeTable", "kwip.data.splitText", "kwip.data.preprocess"].contains(context.node?.operatorID ?? "") ? "Run Flow to capture this component’s output." : context.node?.operatorID == "kwip.data.sourceFiles" ? "Choose a file or folder and refresh its preview, or run the flow." : "Choose a file and refresh its preview, or run the flow to capture output.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
                CapturedDataGridView(grid: presentation.grid, tableOnly: true, evidenceLabel: presentation.label == "Run output" ? (context.outputPopulationLabel ?? presentation.label) : presentation.label, imageResolver: presentation.label.contains("run output") || presentation.label == "Run output" ? context.imageArtifactResolver : nil)
            }
        }
        .onAppear { rememberPreview() }
        .onChange(of: context.node?.id) { _, _ in retainedPreview = nil; rememberPreview() }
        .onChange(of: context.sourceInspectionPreview) { _, _ in rememberPreview() }
    }
    private func rememberPreview() {
        if let preview = context.sourceInspectionPreview { retainedPreview = preview; retainedNodeID = context.node?.id }
    }
}
