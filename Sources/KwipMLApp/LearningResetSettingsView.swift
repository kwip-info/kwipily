import SwiftUI
import Combine
import KwipMLCore

struct LearningResetTarget: Identifiable {
    let id: String
    let title: String
    let detail: String
}

/// One reviewed target at a time: no multi-store partial reset or broad folder deletion.
@MainActor final class LearningResetSettingsModel: ObservableObject {
    @Published private(set) var targets: [LearningResetTarget] = []
    @Published private(set) var previewAttemptIDs: [String] = []
    @Published private(set) var previewRecords: [LearningAttempt] = []
    @Published private(set) var previewTitle: String?
    @Published private(set) var message = ""
    private enum Plan { case objective(LearningProgressResetPlan), tensor(TensorLearningResetPlan) }
    private var plan: Plan?
    private var observations = Set<AnyCancellable>()
    private let workspace: WorkspaceStore
    private let blocked: () -> Bool
    private var directory: URL { workspace.workspaceDirectory.appendingPathComponent("learning") }
    private var progress: LearningProgressStore { LearningProgressStore(directory: directory) }
    private var receipts: TensorLearningReceiptRepository { TensorLearningReceiptRepository(directory: directory.appendingPathComponent("projects")) }
    init(workspace: WorkspaceStore, blocked: (() -> Bool)? = nil) {
        self.workspace = workspace
        self.blocked = blocked ?? { [weak workspace] in
            workspace?.isBusy != false || workspace?.lessonContext != nil ||
                ProjectWindowRegistry.shared.openControllers.contains {
                    $0.session.workspace.isBusy || $0.session.workspace.lessonContext != nil
                }
        }
        workspace.objectWillChange.sink { [weak self] _ in
            DispatchQueue.main.async { self?.objectWillChange.send() }
        }.store(in: &observations)
        ProjectWindowRegistry.shared.objectWillChange.sink { [weak self] _ in
            DispatchQueue.main.async { self?.objectWillChange.send() }
        }.store(in: &observations)
    }
    var canApply: Bool { plan != nil && !blocked() }
    func refresh() {
        plan = nil; previewTitle = nil; previewAttemptIDs = []; previewRecords = []; targets = []
        do {
            let groups = Dictionary(grouping: try progress.attempts(), by: \.objectiveID)
            targets = groups.keys.sorted().map { id in
                LearningResetTarget(id: "objective:" + id,
                    title: id.lowercased().contains("normal") ? "Training-only standardization" : "Model mathematics · " + (id.split(separator: ".").last.map(String.init) ?? id).replacingOccurrences(of: "-", with: " ").capitalized,
                    detail: "\(groups[id]!.count) recorded attempts · objective progress")
            }
            workspace.refreshProjects()
            for project in workspace.lessonProjects {
                if let value = try receipts.previewReset(projectID: project.id) {
                    targets.append(.init(id: "project:" + project.id, title: project.name,
                        detail: "1 current saved check · " + value.objectiveID))
                }
            }
            message = targets.isEmpty ? "No recorded checks to reset. Saved lessons and drafts remain available in Learn." : "Choose one lesson or objective, then review its affected checks."
        } catch { targets = []; message = "Could not read progress safely: " + error.localizedDescription }
    }
    func preview(id: String) {
        plan = nil; previewTitle = nil; previewAttemptIDs = []; previewRecords = []
        guard !blocked() else { message = "Finish operations and close lesson tabs before resetting progress. Your saved work is preserved."; return }
        guard let target = targets.first(where: { $0.id == id }) else { return }
        do {
            if id.hasPrefix("objective:") {
                let value = try progress.previewReset(objectiveID: String(id.dropFirst(10)))
                guard !value.attemptIDs.isEmpty else { throw LearningContractError("Progress changed. Refresh the list.") }
                plan = .objective(value); previewAttemptIDs = value.attemptIDs.sorted()
                previewRecords = try progress.attempts().filter { value.attemptIDs.contains($0.id) }
            } else {
                guard let value = try receipts.previewReset(projectID: String(id.dropFirst(8))) else {
                    throw LearningContractError("Saved check changed. Refresh the list.")
                }
                plan = .tensor(value); previewAttemptIDs = [value.attemptID]
                previewRecords = [try receipts.load(projectID: value.projectID, attemptID: value.attemptID).attempt]
            }
            previewTitle = target.title
            message = "Review the exact affected records below. This cannot be undone; recheck the lesson to earn completion again."
        } catch { message = error.localizedDescription }
    }
    func cancelPreview() { plan = nil; previewTitle = nil; previewAttemptIDs = []; previewRecords = [] }
    @discardableResult func apply() -> Bool {
        guard !blocked() else { message = "Finish operations and close lesson tabs before resetting progress."; return false }
        guard let plan else { return false }
        do {
            switch plan {
            case .objective(let value): try progress.applyReset(value)
            case .tensor(let value): try receipts.applyReset(value)
            }
            refresh()
            message = "Progress reset. Saved canvases, drafts, immutable check history, weights and source files are preserved."
            NotificationCenter.default.post(name: .learningProgressDidReset, object: workspace.workspaceDirectory.standardizedFileURL.path)
            return true
        } catch { cancelPreview(); message = error.localizedDescription; return false }
    }
}

extension Notification.Name { static let learningProgressDidReset = Notification.Name("KwipML.learningProgressDidReset") }

struct LearningResetSettingsView: View {
    @StateObject private var model: LearningResetSettingsModel
    init(store: WorkspaceStore) { _model = StateObject(wrappedValue: LearningResetSettingsModel(workspace: store)) }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionLabel(title: "Saved learning progress")
            Text("Reset one lesson’s current saved check or one mathematics objective’s recorded attempts. Editable lessons and immutable check history stay on this Mac; they do not count as completion after reset.")
                .font(.callout).fixedSize(horizontal: false, vertical: true)
            Text("Finish operations and close lesson tabs first. This prevents an open lesson from continuing to show an old grade. Reading and guided examples have no assessed completion to clear.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            HStack { Text("Recorded checks").fontWeight(.semibold); Spacer(); Button("Refresh") { model.refresh() } }
            ForEach(model.targets) { target in
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(target.title).fontWeight(.medium)
                        Text(target.detail).font(.caption).foregroundStyle(MLTheme.muted)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                    Button("Review reset") { model.preview(id: target.id) }
                }.padding(12).overlay(Rectangle().stroke(MLTheme.rule))
            }
            if let title = model.previewTitle {
                VStack(alignment: .leading, spacing: 10) {
                    SectionLabel(title: "Review reset")
                    Text(title).fontWeight(.semibold)
                    Text("\(model.previewAttemptIDs.count) affected records").font(.callout)
                    HStack {
                        Text("Mode").frame(width: 90, alignment: .leading)
                        Text("Result").frame(width: 100, alignment: .leading)
                        Text("Saved").frame(maxWidth: .infinity, alignment: .leading)
                        Text("Record").frame(width: 85, alignment: .leading)
                    }.font(.caption).foregroundStyle(MLTheme.muted)
                    ForEach(model.previewRecords.prefix(100)) { record in
                        HStack {
                            Text(record.mode.rawValue.capitalized).frame(width: 90, alignment: .leading)
                            Text(record.grade.status.rawValue.capitalized).frame(width: 100, alignment: .leading)
                            Text(record.timestamp, format: .dateTime.year().month().day().hour().minute()).frame(maxWidth: .infinity, alignment: .leading)
                            Text(String(record.id.prefix(8))).font(.system(size: 11, design: .monospaced)).frame(width: 85, alignment: .leading).help(record.id)
                        }.font(.caption)
                        Divider()
                    }
                    if model.previewRecords.count > 100 { Text("Showing the first 100 of \(model.previewRecords.count) attempts in this objective. All \(model.previewRecords.count) are included in the reviewed reset.").font(.caption) }
                    HStack {
                        Button("Cancel") { model.cancelPreview() }
                        Spacer()
                        Button("Reset reviewed progress") { model.apply() }.disabled(!model.canApply)
                            .accessibilityIdentifier("settings.learning.applyReset")
                    }
                }.padding(14).overlay(Rectangle().stroke(MLTheme.rose))
            }
            Text(model.message).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("settings.learning.status")
        }.onAppear { model.refresh() }
    }
}
