import AppKit
import Combine
import SwiftUI

@MainActor protocol EditorHistory: AnyObject {
    var historyChanges: AnyPublisher<Void, Never> { get }
    var canUndo: Bool { get }
    var canRedo: Bool { get }
    var historyContextID: String { get }
    var historyContextIsActive: Bool { get }
    func undo()
    func redo()
}

enum EditorHistoryAction { case undo, redo }

extension EditorHistory where Self: ObservableObject {
    var historyChanges: AnyPublisher<Void, Never> { objectWillChange.map { _ in () }.eraseToAnyPublisher() }
}

/// Weak, origin-bound observable capabilities. Notifications are deferred until a history mutation finishes.
@MainActor final class EditorHistoryContext: ObservableObject {
    @Published private(set) var revision = 0
    private weak var editor: (any EditorHistory)?
    let identity: String
    private var observation: AnyCancellable?
    private var valid = true
    private var undoAvailable = false
    private var redoAvailable = false
    private var scheduled = false
    init(_ editor: any EditorHistory) {
        self.editor = editor; identity = editor.historyContextID
        undoAvailable = canPerform(.undo); redoAvailable = canPerform(.redo)
        observation = editor.historyChanges.sink { [weak self] in self?.scheduleRefresh() }
    }
    func represents(_ editor: any EditorHistory) -> Bool { valid && self.editor === editor && identity == editor.historyContextID }
    func canPerform(_ action: EditorHistoryAction) -> Bool {
        guard valid, let editor, editor.historyContextIsActive, editor.historyContextID == identity else { return false }
        return action == .undo ? editor.canUndo : editor.canRedo
    }
    func perform(_ action: EditorHistoryAction) {
        guard canPerform(action), let editor else { return }
        if action == .undo { editor.undo() } else { editor.redo() }
    }
    func invalidate() { valid = false; observation = nil; refresh() }
    private func scheduleRefresh() {
        guard !scheduled else { return }; scheduled = true
        DispatchQueue.main.async { [weak self] in self?.scheduled = false; self?.refresh() }
    }
    private func refresh() {
        let undo = canPerform(.undo), redo = canPerform(.redo)
        guard undo != undoAvailable || redo != redoAvailable else { return }
        undoAvailable = undo; redoAvailable = redo; revision &+= 1
    }
}

/// Stable per-canvas owner. Rebind only on editor/origin replacement, not on ordinary body updates.
@MainActor final class EditorHistoryProvider: ObservableObject {
    @Published private(set) var context: EditorHistoryContext?
    private var contextObservation: AnyCancellable?
    func attach(_ editor: any EditorHistory) {
        guard context?.represents(editor) != true else { return }
        contextObservation = nil; context?.invalidate(); context = EditorHistoryContext(editor)
        contextObservation = context?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    func detach() { contextObservation = nil; context?.invalidate(); context = nil }
}

/// Presence means text owns history, even when it has no manager or no undoable edits.
struct EditorTextHistory { let manager: UndoManager? }
@MainActor enum EditorHistoryDispatch {
    static func canPerform(_ action: EditorHistoryAction, context: EditorHistoryContext?, text: EditorTextHistory?) -> Bool {
        if let text { return action == .undo ? text.manager?.canUndo == true : text.manager?.canRedo == true }
        return context?.canPerform(action) == true
    }
    static func perform(_ action: EditorHistoryAction, context: EditorHistoryContext?, text: EditorTextHistory?) {
        guard canPerform(action, context: context, text: text) else { return }
        if let text {
            if action == .undo { text.manager?.undo() } else { text.manager?.redo() }
        } else { context?.perform(action) }
    }
}

extension WorkspaceStore: EditorHistory {
    var historyContextID: String { "model:\(activeProjectID ?? "unsaved"):\(historySessionRevision)" }
    var historyContextIsActive: Bool { activeProjectID != nil && !isBusy }
}
extension DataFlowWorkspaceStore: EditorHistory {
    var historyContextID: String { "data:\(projectID ?? "unsaved"):\(historySessionRevision)" }
    var historyContextIsActive: Bool { projectID != nil && projectID == workspace.activeProjectID && !isBusy && !workspace.isBusy }
    var canUndo: Bool { historyContextIsActive && history.canUndo }
    var canRedo: Bool { historyContextIsActive && history.canRedo }
}
extension CompositeWorkspaceStore: EditorHistory {
    var historyContextID: String { "composite:\(projectID ?? "unsaved"):\(historySessionRevision):\(contextKey)" }
    var historyContextIsActive: Bool { projectID != nil && projectID == workspace.activeProjectID && !workspace.isBusy }
    var canUndo: Bool { historyContextIsActive && history.canUndo }
    var canRedo: Bool { historyContextIsActive && history.canRedo }
}

/// Event-driven menu invalidation. No keyboard interception, timers, or polling.
@MainActor final class EditorHistoryFocus: ObservableObject {
    @Published private(set) var revision = 0
    private struct Snapshot: Equatable {
        var responder: ObjectIdentifier?
        var manager: ObjectIdentifier?
        var editing: Bool
        var canUndo: Bool
        var canRedo: Bool
    }
    private var snapshot: Snapshot?
    private var observations: [NSObjectProtocol] = []
    private let currentResponder: @MainActor () -> NSResponder?
    init(responder: (@MainActor () -> NSResponder?)? = nil) {
        currentResponder = responder ?? { NSApp.keyWindow?.firstResponder }
        // Window update is AppKit's notification after responder/focus changes. Compare state before publishing
        // so ordinary drawing updates cannot cause a SwiftUI invalidation loop.
        let names: [Notification.Name] = [NSWindow.didUpdateNotification, NSWindow.didBecomeKeyNotification, NSWindow.didResignKeyNotification,
            NSText.didBeginEditingNotification, NSText.didEndEditingNotification, NSText.didChangeNotification,
            Notification.Name.NSUndoManagerDidUndoChange, Notification.Name.NSUndoManagerDidRedoChange,
            Notification.Name.NSUndoManagerDidOpenUndoGroup, Notification.Name.NSUndoManagerDidCloseUndoGroup]
        observations = names.map { name in
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in DispatchQueue.main.async { self?.refresh() } }
        }
        refresh()
    }
    deinit { for observation in observations { NotificationCenter.default.removeObserver(observation) } }
    var textHistory: EditorTextHistory? {
        let responder = currentResponder()
        let text = (responder as? NSTextView) ?? (responder as? NSTextField)?.currentEditor() as? NSTextView
        guard let text, text.isEditable else { return nil }
        return EditorTextHistory(manager: text.undoManager)
    }
    func refresh() {
        let text = textHistory
        let next = Snapshot(responder: currentResponder().map(ObjectIdentifier.init), manager: text?.manager.map(ObjectIdentifier.init), editing: text != nil,
                            canUndo: text?.manager?.canUndo == true, canRedo: text?.manager?.canRedo == true)
        guard next != snapshot else { return }; snapshot = next; revision &+= 1
    }
    func perform(_ action: EditorHistoryAction, context: EditorHistoryContext?) {
        // Resolve current responder again; menu availability may have been evaluated before focus moved.
        EditorHistoryDispatch.perform(action, context: context, text: textHistory)
        refresh()
    }
}

struct EditorHistoryCommands: Commands {
    @FocusedObject private var provider: EditorHistoryProvider?
    private var editor: EditorHistoryContext? { provider?.context }
    @StateObject private var focus = EditorHistoryFocus()
    var body: some Commands {
        CommandGroup(replacing: .undoRedo) {
            Button("Undo") { focus.perform(.undo, context: editor) }
                .keyboardShortcut("z").disabled(!EditorHistoryDispatch.canPerform(.undo, context: editor, text: focus.textHistory))
            Button("Redo") { focus.perform(.redo, context: editor) }
                .keyboardShortcut("z", modifiers: [.command, .shift]).disabled(!EditorHistoryDispatch.canPerform(.redo, context: editor, text: focus.textHistory))
        }
    }
}
