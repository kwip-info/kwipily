import SwiftUI
import KwipMLCore

/// One editable child draft per parent; closing a window never discards its saved work.
@MainActor final class ModelChildEditorSession: ObservableObject {
    let owner: any ModelCanvasAuthority
    @Published private(set) var editor: ModelCanvasAdapter?
    @Published private(set) var error: String?
    private var active = true
    var presentation = CanvasPresentationSettings()
    var configurePresentation: ((ModelCanvasAdapter) -> Void)?
    var persistPresentation: (() throws -> Void)?
    init(owner: any ModelCanvasAuthority) { self.owner = owner }
    @discardableResult func open(path: [String]) -> Bool {
        guard active, !owner.isBusy else { error = "Stop the active execution before opening this Block."; return false }
        if let editor, editor.path == path, editor.editingDraft { error = nil; return true }
        guard flushPresentation() else { return false }
        guard owner.beginDraft(path: path, lease: owner.lease()) else { error = owner.error; return false }
        do {
            let next = try ModelCanvasAdapter(owner: owner, path: path, presentation: presentation)
            next.onOpenBlock = { [weak self] path in _ = self?.open(path: path) }
            editor?.invalidate(); editor = next; configurePresentation?(next); error = nil; return true
        } catch { self.error = error.localizedDescription; return false }
    }
    func restorePresentation() { if active, let editor { configurePresentation?(editor) } }
    @discardableResult func flushPresentation() -> Bool {
        guard active, editor != nil else { return true }
        do { try persistPresentation?(); error = nil; return true }
        catch {
            self.error = "Block layout could not be saved: \(error.localizedDescription)"
            editor?.statusMessage = self.error!
            return false
        }
    }
    private func tearDownEditor() { editor?.invalidate(); editor = nil }
    @discardableResult func close() -> Bool {
        guard flushPresentation() else { return false }
        tearDownEditor(); return true
    }
    func invalidate() { active = false; tearDownEditor(); configurePresentation = nil; persistPresentation = nil }
    @discardableResult func apply() -> Bool {
        guard active, let editor else { return false }
        guard flushPresentation() else { return false }
        guard owner.applyDraft(lease: owner.lease(path: editor.path)) else { error = owner.error; return false }
        tearDownEditor(); error = nil; return true
    }
    @discardableResult func discard() -> Bool {
        guard active, let editor else { return false }
        guard flushPresentation() else { return false }
        guard owner.discardDraft(lease: owner.lease(path: editor.path)) else { error = owner.error; return false }
        tearDownEditor(); error = nil; return true
    }
}
