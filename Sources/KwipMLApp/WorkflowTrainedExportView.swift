import SwiftUI
import AppKit

struct WorkflowTrainedExportView: View {
    @ObservedObject var export: WorkflowTrainedExportStore
    @ObservedObject var workspace: WorkspaceStore
    @State private var choosingDestination = false
    @State private var folderName = ""
    @State private var destinationError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("TRAINED MODEL PACKAGE").font(.system(size: 10, weight: .medium)).foregroundStyle(MLTheme.muted)
                Spacer()
                if export.isBusy {
                    ProgressView().controlSize(.small)
                    Button("Cancel export") { export.cancel() }.buttonStyle(FlatButtonStyle())
                } else {
                    Button("Prepare again") { export.prepare() }.buttonStyle(FlatButtonStyle()).disabled(workspace.isBusy)
                    Button("Choose destination…") { chooseDestination() }.buttonStyle(FlatButtonStyle(prominent: true))
                        .disabled(export.package == nil || workspace.isBusy || choosingDestination || !validFolderName)
                }
            }
            Text(export.statusMessage).font(.caption).textSelection(.enabled)
            HStack {
                Text("Package folder").font(.caption).foregroundStyle(MLTheme.muted)
                TextField("Folder name", text: $folderName).textFieldStyle(.plain).font(.system(size: 12))
                    .padding(8).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    .frame(maxWidth: 420).disabled(export.isBusy || choosingDestination)
            }
            if let destinationError { Text(destinationError).font(.caption).foregroundStyle(MLTheme.rose) }
            if !folderName.isEmpty && !validFolderName {
                Text("Use a single folder name, up to 128 bytes, without path separators or control characters.").font(.caption).foregroundStyle(MLTheme.rose)
            }
            if let package = export.package {
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Use this trained model outside KwipML").font(.headline)
                        Text("Includes the captured model source, trained checkpoint and a standalone Python inference helper. Canvas edits do not change this package.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Input contract").font(.subheadline.bold())
                            Text(package.inputContract.reviewSummary).font(.caption)
                            Text("Column order: " + package.inputContract.columns.joined(separator: " → ")).font(.caption).textSelection(.enabled)
                            Text(package.inputContract.valueGuidance).font(.caption).foregroundStyle(MLTheme.muted)
                            Text(package.inputContract.outputReviewSummary).font(.caption).foregroundStyle(MLTheme.muted)
                            Text("Apply the same preprocessing used during training before inference. This package does not include source rows or fitted preprocessing.")
                                .font(.caption).foregroundStyle(MLTheme.muted)
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Package contents").font(.subheadline.bold())
                            ForEach(package.files, id: \.path) { file in
                                HStack {
                                    Text(file.path).font(.system(size: 11, design: .monospaced))
                                    Spacer()
                                    Text(ByteCountFormatter.string(fromByteCount: Int64(file.bytes), countStyle: .file)).font(.caption).foregroundStyle(MLTheme.muted)
                                }
                            }
                            HStack {
                                Text("export-manifest.json").font(.system(size: 11, design: .monospaced))
                                Spacer()
                                Text(ByteCountFormatter.string(fromByteCount: Int64(package.manifestBytes.count), countStyle: .file)).font(.caption).foregroundStyle(MLTheme.muted)
                            }
                        }.frame(maxWidth: 620, alignment: .leading)
                        Text("The checkpoint includes weights, optimizer state, random-generator state and provenance hashes. Model names and topology are included. Private source rows and local source bindings are excluded.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Try the exported model").font(.subheadline.bold())
                            Text("python -m pip install -r requirements.txt\npython inference.py --example --batch-size 256")
                                .font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                            Text("Run inside the saved folder using Python with the recorded PyTorch version. The example uses synthetic prepared inputs. README.md explains custom inputs, batch sizes and loading once for repeated predictions.")
                                .font(.caption).foregroundStyle(MLTheme.muted)
                        }
                        DisclosureGroup("Verified identity and file hashes") {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Run: \(package.runID)\nRequest: \(package.requestIdentity)\nModel manifest: \(package.modelManifestSHA256)")
                                ForEach(package.files, id: \.path) { file in Text("\(file.path)\n\(file.sha256)") }
                            }.font(.system(size: 10, design: .monospaced)).textSelection(.enabled)
                        }
                        if let saved = export.savedURL {
                            Text("Saved to \(saved.path)").font(.caption).textSelection(.enabled)
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([saved]) }.buttonStyle(FlatButtonStyle())
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(.trailing, 8)
                }
            } else { Spacer() }
        }.onAppear {
            if folderName.isEmpty, let package = export.package { folderName = "KwipML-model-" + String(package.runID.prefix(8)) }
        }.onChange(of: export.package?.runID) { _, runID in
            if folderName.isEmpty, let runID { folderName = "KwipML-model-" + String(runID.prefix(8)) }
        }
    }
    private var validFolderName: Bool {
        (try? WorkflowTrainedExportPackage.destination(in: URL(fileURLWithPath: "/"), folderName: folderName)) != nil
    }
    private func chooseDestination() {
        guard let package = export.package, !export.isBusy, !workspace.isBusy, !choosingDestination else { return }
        let panel = WorkflowExportParentChooser.panel(title: "Choose export destination", message: "The reviewed package will be saved in a new folder named “" + folderName + "”. Existing folders will not be replaced.", prompt: "Save package")
        let name = folderName
        choosingDestination = true
        destinationError = nil
        panel.begin { response in
            choosingDestination = false
            guard response == .OK, let parent = panel.url else { return }
            do {
                let destination = try WorkflowTrainedExportPackage.destination(in: parent, folderName: name)
                guard export.package?.runID == package.runID else { return }
                export.save(to: destination, access: parent)
            } catch { destinationError = error.localizedDescription }
        }
    }
}
