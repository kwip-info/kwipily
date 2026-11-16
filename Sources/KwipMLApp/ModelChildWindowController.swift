import AppKit
import SwiftUI
import KwipMLCore

/// Retained by the parent workspace. Native windows borrow a session; they never own a document copy.
@MainActor final class ModelChildWindowController: NSObject, NSWindowDelegate {
    let session: ModelChildEditorSession
    private var window: NSWindow?
    var restorationWindow: NSWindow? { window }
    private weak var displayedEditor: ModelCanvasAdapter?
    var onDraftResolved: (() -> Void)?
    var capturedOutput: (([String], String) -> WorkflowCapturedDataOutput?)?
    private let appearance: AppAppearanceState
    init(session: ModelChildEditorSession, appearance: AppAppearanceState? = nil) { self.session = session; self.appearance = appearance ?? AppAppearanceState(); super.init() }
    @discardableResult func open(path: [String], present: Bool = true) -> Bool {
        guard session.open(path: path), let editor = session.editor else { return false }
        if let window, displayedEditor === editor { if present { window.makeKeyAndOrderFront(nil) }; return true }
        let isNewWindow = window == nil
        let window = window ?? NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1120, height: 760),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = editor.breadcrumbTitles.joined(separator: " › ")
        window.minSize = NSSize(width: 760, height: 520)
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.tabbingIdentifier = "info.kwip.KwipML.canvas-documents"; window.tabbingMode = .preferred
        editor.capturedOutput = capturedOutput
        window.contentView = NSHostingView(rootView: ModelScopeEditorView(store: editor,
            onApply: { [weak self] in _ = self?.applyAndClose() },
            onDiscard: { [weak self] in _ = self?.discardAndClose() })
            .buttonStyle(FlatButtonStyle()).scrollIndicators(.hidden).appAppearance(appearance))
        self.window = window; displayedEditor = editor
        if isNewWindow { window.center() }; if present { window.makeKeyAndOrderFront(nil) }
        // Route nested requests through the retained window owner, including diagnostics on failure.
        editor.onOpenBlock = { [weak self] path in _ = self?.open(path: path) }
        return true
    }
    @discardableResult func applyAndClose() -> Bool {
        guard session.apply() else { return false }
        window?.close(); onDraftResolved?(); return true
    }
    @discardableResult func discardAndClose() -> Bool {
        guard session.discard() else { return false }
        window?.close(); onDraftResolved?(); return true
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool { session.close() }
    func windowWillClose(_ notification: Notification) { session.close(); window = nil; displayedEditor = nil }
    func invalidate() { onDraftResolved = nil; session.invalidate(); window?.close(); window = nil; displayedEditor = nil }
}
