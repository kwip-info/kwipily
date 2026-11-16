import SwiftUI
import UniformTypeIdentifiers

private struct SupportSummaryDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    let bytes: Data
    init(bytes: Data) { self.bytes = bytes }
    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents, data.count <= 16_384 else {
            throw WorkflowArchiveError("Invalid support summary document.")
        }
        bytes = data
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper { FileWrapper(regularFileWithContents: bytes) }
}

struct WorkflowSupportSummaryView: View {
    let verified: VerifiedWorkflowOutcome
    let isActive: Bool
    @State private var includePolicy = true
    @State private var showingExporter = false
    @State private var document: SupportSummaryDocument?
    @State private var message: String?
    private var summary: Result<Data, Error> {
        Result { try WorkflowSupportSummary(outcome: verified.outcome, budget: verified.capture.executionBudget,
            sampleCapture: verified.capture.sampleCapture, includeExecutionPolicy: includePolicy).encoded() }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Support summary")
            Text("Review exactly what will be saved. This local JSON contains status, terminal phase, artifact count and any recorded failure category. Optional policy details contain limits and capture choices. Missing historical policy remains omitted.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title: "Support policy details", selection: includePolicy ? "include" : "omit", options: [
                .init(id: "include", title: "Include execution policy"),
                .init(id: "omit", title: "Omit execution policy")]) { includePolicy = $0 == "include" }
                .disabled(showingExporter)
            Text("Excludes names, identifiers, paths, timestamps, raw errors/logs, source rows, tensor values, graph/code and weights. This is a limited summary, not a full diagnostic bundle. Nothing is uploaded or sent.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            switch summary {
            case .success(let bytes):
                ThemedScrollView {
                    Text(String(decoding: bytes, as: UTF8.self)).font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(12)
                }.overlay(Rectangle().stroke(MLTheme.rule))
                Button("Save reviewed summary…") {
                    guard isActive else { return }
                    document = .init(bytes: bytes); showingExporter = true; message = nil
                }.buttonStyle(FlatButtonStyle(prominent: true)).disabled(!isActive || showingExporter)
            case .failure(let error): Text(error.localizedDescription).font(.caption).foregroundStyle(MLTheme.rose)
            }
            if let message { Text(message).font(.caption).foregroundStyle(MLTheme.muted) }
        }.fileExporter(isPresented: $showingExporter, document: document, contentType: .json,
            defaultFilename: "kwipml-support-summary") { result in
                switch result {
                case .success: message = "Summary saved locally."
                case .failure(let error): message = "Summary was not saved. \(error.localizedDescription)"
                }
            }
    }
}
