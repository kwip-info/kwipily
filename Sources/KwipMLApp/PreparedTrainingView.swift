import SwiftUI
import CryptoKit
import KwipMLCore

/// Explicitly maps real preparation outputs; never infers a target or re-splits rows.
struct PreparedTrainingView: View {
    @ObservedObject var preparation: DataFlowWorkspaceStore
    @State private var training = ""
    @State private var validation = ""
    @State private var first = ""
    @State private var second = ""
    @State private var target = ""
    @State private var message: String?
    private struct OutputChoice: Identifiable {
        var node: String; var port: String; var title: String; var role: String; var columns: [String]
        var id: String { node + "\u{1f}" + port }
    }
    private var outputs: [OutputChoice] {
        preparation.previews.keys.sorted().flatMap { node in
            (preparation.previews[node] ?? [:]).keys.sorted().compactMap { port in
                guard let output = preparation.previews[node]?[port] as? [String: Any], output["kind"] as? String == "table",
                      let preview = output["preview"] as? [String: Any], let columns = preview["columns"] as? [[String: Any]],
                      let lineage = output["lineage"] as? [String: Any], let role = lineage["role"] as? String else { return nil }
                return OutputChoice(node: node, port: port, title: "\(preparation.project.editor[node]?.name ?? node) · \(port)", role: role, columns: columns.compactMap { $0["name"] as? String })
            }
        }
    }
    private var columns: [String] { outputs.first { $0.id == training }?.columns ?? [] }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Use prepared data for training").font(.headline)
            Text("Choose exact output partitions and column roles. Training preserves these rows and fitted transforms; model split settings are not applied.").font(.caption).foregroundStyle(MLTheme.muted)
            choice("Training output", value: training, options: outputs.filter { $0.role == "training" }.map { FlatChoiceOption(id: $0.id, title: $0.title) }) { training = $0; first = ""; second = ""; target = "" }
            choice("Validation output", value: validation, options: outputs.filter { $0.role == "validation" }.map { FlatChoiceOption(id: $0.id, title: $0.title) }) { validation = $0 }
            choice("Feature 1", value: first, options: columns.map { FlatChoiceOption(id: $0, title: $0) }) { first = $0 }
            choice("Feature 2", value: second, options: columns.map { FlatChoiceOption(id: $0, title: $0) }) { second = $0 }
            choice("Target", value: target, options: columns.map { FlatChoiceOption(id: $0, title: $0) }) { target = $0 }
            Text("This classifier accepts two numeric features and an integer binary target. Cast the target to int64; apply the same fitted scaling state to both partitions.").font(.caption).foregroundStyle(MLTheme.muted)
            if let message { Text(message).font(.caption).foregroundStyle(MLTheme.teal) }
            Button("Use these partitions") { apply() }
                .buttonStyle(FlatButtonStyle()).disabled(!preparation.resultsCurrent || preparation.isBusy || training.isEmpty || validation.isEmpty || first.isEmpty || second.isEmpty || target.isEmpty)
            if !preparation.resultsCurrent { Text("Prepare the current graph and sources before selecting outputs.").font(.caption).foregroundStyle(.orange) }
        }.padding(20).frame(width: 440).background(MLTheme.panel)
    }
    private func choice(_ title: String, value: String, options: [FlatChoiceOption], select: @escaping (String) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: value, options: options, placeholder: "Select…", onSelect: select)
        }
    }
    private func apply() {
        do {
            try preparation.requirePreparationAccess()
            guard preparation.resultsCurrent, let url = preparation.manifestURL,
                  let train = outputs.first(where: { $0.id == training }), let heldout = outputs.first(where: { $0.id == validation }) else { throw DataFlowContractError("Prepare data and choose both output partitions first.") }
            let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
            let data = try handle.read(upToCount: 2_000_001) ?? Data()
            guard data.count <= 2_000_000, let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any], let digest = manifest["graphDigest"] as? String else { throw DataFlowContractError("Preparation manifest is unreadable.") }
            let selection = PreparedTrainingSelection(manifestPath: url.path, manifestSHA256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(), graphDigest: digest, trainingNode: train.node, trainingPort: train.port, validationNode: heldout.node, validationPort: heldout.port, featureColumns: [first, second], targetColumn: target)
            try preparation.workspace.usePreparedTraining(selection)
            message = "Selected. Run the model from the workbench; this preparation snapshot is retained with the run."
        } catch { message = error.localizedDescription }
    }
}
