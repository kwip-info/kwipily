import AppKit
import SwiftUI

/// A configuration editor is modeless: it must never enter a sheet's modal event loop,
/// consume Quit, or require the first Close click merely to activate a panel.
typealias SelfPlayEditorWindowController = ExpandedConfigurationWindowController

@MainActor final class ExpandedConfigurationWindowController: NSWindowController, NSWindowDelegate {
    // An inspector can disappear while its editor still owns an unapplied draft.
    // Retain that editor until an explicit close decision, rather than dropping its owner.
    private static var editors:[ExpandedConfigurationWindowController]=[]
    private var confirmingClose=false
    var confirmDiscard:((NSWindow,@escaping (Bool)->Void)->Void)?
    var hasUnappliedDraft:(()->Bool)?
    static func prepareToTerminate(confirmDiscard:(()->Bool)?=nil)->Bool {
        let live=editors.filter{!$0.didClose}
        if live.contains(where:{$0.hasUnappliedDraft?() == true}) {
            if let confirmDiscard {guard confirmDiscard() else{return false}}
            else {
                let alert=NSAlert();alert.messageText="Discard unapplied configuration drafts and quit?";alert.informativeText="Applied settings and saved project content are preserved. Unapplied configuration editor changes will be discarded."
                alert.addButton(withTitle:"Discard drafts and continue quitting");alert.addButton(withTitle:"Cancel")
                guard alert.runModal() == .alertFirstButtonReturn else{return false}
            }
        }
        live.forEach{$0.window?.close()};return true
    }
    var discardDraft: (() -> Void)?
    private(set) var didClose = false

    init(title:String="Train · Self-play",identifier:String="info.kwip.KwipML.self-play-editor",content: (ExpandedConfigurationWindowController) -> AnyView) {
        let window = NSWindow(contentRect: .init(x: 0, y: 0, width: 940, height: 730),
            styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = title
        window.identifier = .init(identifier)
        window.isReleasedWhenClosed = false
        window.tabbingMode = .disallowed
        window.minSize = .init(width: 800, height: 640)
        // Configuration does not require initial text selection or a first focus commit.
        window.initialFirstResponder = nil
        super.init(window: window)
        window.delegate = self
        window.contentView = NSHostingView(rootView: content(self))
        window.center()
        Self.editors.append(self)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    func present() {
        guard !didClose else { return }
        let wasVisible=window?.isVisible == true
        showWindow(nil)
        window?.makeKeyAndOrderFront(nil)
        if !wasVisible {window?.makeFirstResponder(nil)}
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Button, Escape and native Cmd-W all end the same draft lifetime.
    override func close() { window?.performClose(nil) }

    func windowShouldClose(_ sender:NSWindow)->Bool {
        guard !didClose else{return true}
        guard !confirmingClose else{return false}
        guard hasUnappliedDraft?() == true else{return true}
        confirmingClose=true
        let finish:(Bool)->Void = { [self] discard in
            confirmingClose=false
            guard !didClose else{return}
            if discard {sender.close()}
            else {sender.makeKeyAndOrderFront(nil)}
        }
        if let confirmDiscard {confirmDiscard(sender,finish)}
        else {
            let alert=NSAlert()
            alert.messageText="Discard unapplied changes?"
            alert.informativeText="Keep editing to preserve this draft. Applied project settings are unchanged."
            alert.addButton(withTitle:"Keep editing")
            alert.addButton(withTitle:"Discard changes")
            alert.beginSheetModal(for:sender){response in finish(response == .alertSecondButtonReturn)}
        }
        return false
    }

    func windowWillClose(_ notification: Notification) {
        guard !didClose else { return }
        didClose = true
        let discard = discardDraft
        discardDraft = nil;hasUnappliedDraft=nil
        discard?()
        // Closing an NSWindow does not necessarily dismantle its SwiftUI subtree.
        // Explicitly release tasks/observations and the borrowed context afterwards.
        window?.contentView = nil
        window?.delegate = nil
        window?.windowController = nil
        confirmDiscard = nil
        Self.editors.removeAll{$0 === self}
    }
}
