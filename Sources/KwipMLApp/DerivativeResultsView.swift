import SwiftUI

/// Public verified History results only. The caller supplies the verified result projection.
struct DerivativeResultsView: View {
    let presentation: DerivativeResultsPresentation
    @State private var requestIndex = 0
    @State private var targetIndex = 0
    private var selectedRequest: DerivativeResultsPresentation.Request? {
        guard !presentation.requests.isEmpty else { return nil }
        return presentation.requests[min(requestIndex, presentation.requests.count - 1)]
    }
    var body: some View {
        ResultsSection(title: "Requested small-change effects", initiallyExpanded: true) {
            VStack(alignment: .leading, spacing: 12) {
                Text(DerivativeResultsPresentation.explanation).font(.caption).foregroundStyle(MLTheme.muted)
                if let request = selectedRequest {
                    FlatChoiceControl(title: "Request", selection: request.id,
                        options: presentation.requests.map { .init(id: $0.id, title: $0.id + " · " + $0.output) }) { id in
                        if let index = presentation.requests.firstIndex(where: { $0.id == id }) {
                            requestIndex = index; targetIndex = 0
                        }
                    }
                    Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 5) {
                        summary("Result port", request.output)
                        summary("Phase", request.phase)
                        summary("Selected observations", request.partition.capitalized + " · rows \(request.rowStart + 1)–\(request.rowStart + request.rowCount) · \(request.rowCount) observations")
                        summary("Result shape", "[" + request.outputShape.map(String.init).joined(separator: ", ") + "]")
                        summary("Device", presentation.device.rawValue.uppercased())
                    }.textSelection(.enabled)
                    Text("Targets · select one to inspect its values or unavailable reason")
                        .font(.system(size: 11, weight: .semibold))
                    ThemedScrollView(.horizontal) {
                        Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 7) {
                            GridRow {
                                Text("Target"); Text("Status"); Text("Shape"); Text("Type")
                            }.font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted)
                            ForEach(Array(request.targets.enumerated()), id: \.offset) { index, target in
                                GridRow {
                                    Button { targetIndex = index } label: {
                                        HStack(spacing: 5) {
                                            Image(systemName: targetIndex == index ? "circle.inset.filled" : "circle")
                                            Text(target.name)
                                        }
                                    }.buttonStyle(.borderless).accessibilityLabel("Inspect " + target.name)
                                    Text(target.statusText)
                                    Text("[" + target.shape.map(String.init).joined(separator: ", ") + "]")
                                    Text(target.dtype.rawValue)
                                }.font(.system(size: 11, design: .monospaced))
                            }
                        }
                    }
                    if !request.targets.isEmpty {
                        let target = request.targets[min(targetIndex, request.targets.count - 1)]
                        Text(target.explanation).font(.caption).foregroundStyle(MLTheme.muted)
                        if let tensor = target.tensor {
                            LearningTensorTable(tensor: tensor, title: target.name + " · measured small-change effect", signed: true)
                                .id(presentation.runID + ":" + request.id + ":" + target.name)
                        }
                    }
                } else {
                    Text("This run has no requested derivative results.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }.padding(.top, 8)
        }
    }
    @ViewBuilder private func summary(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(MLTheme.muted)
            Text(value).frame(maxWidth: .infinity, alignment: .leading)
        }.font(.system(size: 11))
    }
}
