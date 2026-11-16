import AppKit
import SwiftUI
import Combine
import KwipMLCore

@MainActor final class FlowCanvasWindowRegistry {
    static let shared = FlowCanvasWindowRegistry()
    private var windows: [URL: FlowCanvasWindowController] = [:]
    private struct Identity: Hashable { let locationID: UUID?; let documentID: UUID }
    private var identities: [Identity: URL] = [:]
    private weak var workspaceWindow: NSWindow?
    private let tabGroupIdentifier = "info.kwip.KwipML.canvas-documents"
    func registerWorkspaceWindow(_ window: NSWindow, title: String) {
        workspaceWindow = window
        window.tabbingIdentifier = tabGroupIdentifier
        window.tabbingMode = .preferred
        window.tab.title = title
    }
    func attachAsTab(_ controller: FlowCanvasWindowController, to host: NSWindow) {
        guard let window = controller.window, window !== host else { return }
        window.tabbingIdentifier = tabGroupIdentifier
        window.tabbingMode = .preferred
        window.tab.title = controller.canvas.title
        if host.tabbedWindows?.contains(where: { $0 === window }) != true {
            host.addTabbedWindow(window, ordered: .above)
        }
        host.tabGroup?.selectedWindow = window
    }
    private func presentNew(_ controller: FlowCanvasWindowController) {
        let active = NSApp.keyWindow
        let host = active?.tabbingIdentifier == tabGroupIdentifier ? active : workspaceWindow
        if let host { attachAsTab(controller, to: host) }
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
        ProjectWindowRegistry.shared.dismissWelcomeForOpenedCanvas()
    }
    var openControllers: [FlowCanvasWindowController] { Array(windows.values) }
    var appearance: AppAppearanceState
    init(appearance: AppAppearanceState? = nil) { self.appearance = appearance ?? AppAppearanceState() }
    /// Returns whether the command belongs to a flow window, even when its save fails.
    @discardableResult func save(window: NSWindow?) -> Bool {
        guard let window, let controller = windows.values.first(where: { $0.window === window }) else { return false }
        _ = controller.documentOwner.save(); return true
    }
    func prepareToTerminate(decide: ((FlowCanvasWindowController) -> FlowCanvasWindowController.CloseDecision)? = nil) -> Bool {
        for key in windows.keys.sorted(by: { $0.path < $1.path }) {
            guard let controller = windows[key], controller.documentOwner.isDirty else { continue }
            if let decide {
                guard controller.prepareToClose(decide(controller)) else { return false }
            } else if let window = controller.window, !controller.windowShouldClose(window) { return false }
        }
        return true
    }
    @discardableResult func open(_ url: URL, present: Bool = true, callerPin: FlowDocumentRevision? = nil,
                                 accessLease: FlowFolderAccess.Lease? = nil, restorationTarget: CanvasTabSession.Target? = nil) throws -> FlowCanvasWindowController {
        let canonical = url.resolvingSymlinksInPath().standardizedFileURL
        if let accessLease {
            let folder = accessLease.folderURL.path
            guard canonical.path.hasPrefix(folder.hasSuffix("/") ? folder : folder + "/") else { throw FlowDocumentError.unsafeReference }
        }
        if let existing = windows[canonical] {
            let identity = Identity(locationID: accessLease?.locationID, documentID: existing.documentOwner.document.id)
            guard identities[identity] == nil || identities[identity] == canonical else { throw FlowDocumentError.invalidDocument }
            if let callerPin {
                guard existing.documentOwner.document.id == callerPin.documentID else { throw FlowDocumentError.revisionMismatch }
                existing.documentOwner.referenceNotice = "Caller uses pinned revision \(callerPin.revision). Saving this document does not update that caller."
            }
            identities[identity] = canonical
            existing.retainAccess(accessLease)
            if let restorationTarget { existing.restorationTarget = restorationTarget }
            if present { existing.showWindow(nil); existing.window?.makeKeyAndOrderFront(nil); ProjectWindowRegistry.shared.dismissWelcomeForOpenedCanvas() }; return existing }
        let owner = try FlowDocumentOwner(url: url)
        if let callerPin {
            guard owner.document.id == callerPin.documentID else { throw FlowDocumentError.revisionMismatch }
            owner.referenceNotice = "Caller uses pinned revision \(callerPin.revision). Saving this document does not update that caller."
        }
        let identity = Identity(locationID: accessLease?.locationID, documentID: owner.document.id)
        guard identities[identity] == nil else { throw FlowDocumentError.invalidDocument }
        let controller = try FlowCanvasWindowController(owner: owner, appearance: appearance)
        controller.window?.tabbingIdentifier = tabGroupIdentifier
        controller.window?.tabbingMode = .preferred
        controller.retainAccess(accessLease)
        controller.restorationTarget = restorationTarget
        controller.openDependency = { [weak self, weak controller] url, pin in
            guard let controller else { throw FlowDocumentError.invalidDocument }
            _ = try self?.open(url, callerPin: pin, accessLease: controller.accessLease, restorationTarget: try controller.targetForDependency(url, documentID: pin.documentID))
        }
        controller.didClose = { [weak self, weak controller] in
            guard let self, let controller, self.windows[canonical] === controller else { return }
            self.windows.removeValue(forKey: canonical)
            self.identities = self.identities.filter { $0.value != canonical }
        }
        windows[canonical] = controller; identities[identity] = canonical
        if present { presentNew(controller) }
        return controller
    }
}

@MainActor final class FlowCanvasWindowController: NSWindowController, NSWindowDelegate {
    enum CloseDecision { case save, revert, cancel }
    let documentOwner: FlowDocumentOwner
    let authority: FlowBlockCanvasAuthority
    let canvas: ModelCanvasAdapter
    var restorationTarget: CanvasTabSession.Target? { didSet { updateDocumentIdentity() } }
    private(set) var accessLease: FlowFolderAccess.Lease?
    fileprivate func retainAccess(_ lease: FlowFolderAccess.Lease?) {
        if accessLease == nil { accessLease = lease }
        updateDocumentIdentity()
    }
    var openDependency: ((URL, FlowDocumentRevision) throws -> Void)?
    var didClose: (() -> Void)?
    private var observation: AnyCancellable?
    private var documentIdentity: DocumentWindowIdentity?
    init(owner: FlowDocumentOwner, appearance: AppAppearanceState) throws {
        self.documentOwner = owner
        authority = try FlowBlockCanvasAuthority(owner: owner)
        canvas = try ModelCanvasAdapter(owner: authority, path: authority.path)
        let window = NSWindow(contentRect: .init(x: 0, y: 0, width: 1120, height: 760),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        super.init(window: window)
        window.isReleasedWhenClosed = false; window.delegate = self
        window.minSize = .init(width: 760, height: 520)
        updateDocumentIdentity()
        window.title = canvas.title; window.center()
        // Each peer keeps its own bottom readout; captured run binding is supplied separately from authoring.
        canvas.capturedOutput = { _, _ in nil }
        canvas.onOpenBlock = { [weak self] path in self?.openNested(path) }
        window.contentView = NSHostingView(rootView: VStack(spacing: 0) {
            FlowRecoveryBanner(owner: owner)
            ModelScopeEditorView(store: canvas,
            onApply: { [weak owner] in _ = owner?.save() }, onDiscard: { [weak owner] in owner?.revert() })
            }.buttonStyle(FlatButtonStyle()).scrollIndicators(.hidden).appAppearance(appearance))
        observation = owner.$document.receive(on: RunLoop.main).sink { [weak self] _ in
            guard let self else { return }; self.window?.isDocumentEdited = self.documentOwner.isDirty
            self.window?.title = self.canvas.title
            self.window?.tab.title = self.canvas.title
            self.updateDocumentIdentity()
        }
    }
    private func updateDocumentIdentity() {
        guard let window else { return }
        let identity = DocumentWindowIdentity.block(url: documentOwner.url, restorationTarget: restorationTarget,
            hasExternalAccess: accessLease != nil)
        if identity != documentIdentity { identity.apply(to: window); documentIdentity = identity }
    }
    required init?(coder: NSCoder) { fatalError("File-backed windows require a document owner") }
    @objc override func newWindowForTab(_ sender: Any?) { ProjectWindowRegistry.shared.createNewFromMenu() }
    private func openNested(_ path: [String]) {
        do {
            guard path.count == 2, case .modelBlock(let block) = documentOwner.document.payload,
                  let instance = block.body.instances.first(where: { $0.id == path[1] }),
                  let dependency = documentOwner.document.dependencies.first(where: { $0.definition == instance.definition }) else {
                throw FlowDocumentError.missingDependency
            }
            try openDependency?(documentOwner.url.deletingLastPathComponent().appendingPathComponent(dependency.relativePath), dependency.revision)
        } catch { canvas.statusMessage = error.localizedDescription }
    }
    func prepareToClose(_ decision: CloseDecision) -> Bool {
        guard documentOwner.isDirty else { return true }
        switch decision { case .save: return documentOwner.save(); case .revert: return documentOwner.revert(); case .cancel: return false }
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        handleCloseRequest {
            let alert = NSAlert(); alert.messageText = "Save changes to \(canvas.title)?"
            alert.informativeText = "This subflow has its own file. Saving does not change a caller's pinned revision."
            alert.addButton(withTitle: "Save"); alert.addButton(withTitle: "Cancel"); alert.addButton(withTitle: "Revert changes")
            let response = alert.runModal()
            return response == .alertFirstButtonReturn ? .save : response == .alertThirdButtonReturn ? .revert : .cancel
        }
    }
    /// Shares the native confirmation path with noninteractive lifecycle tests.
    func handleCloseRequest(decide: () -> CloseDecision) -> Bool {
        guard documentOwner.isDirty else { return true }
        // The footer observes documentOwner.error through the authority. Do not
        // duplicate it into canvas status: successful toolbar retries clear only
        // the authoritative error, and unrelated canvas notices must survive.
        return prepareToClose(decide())
    }
    func windowWillClose(_ notification: Notification) {
        canvas.invalidate(); observation = nil; openDependency = nil
        didClose?(); didClose = nil
        accessLease = nil
    }
}

private struct FlowRecoveryBanner: View {
    @ObservedObject var owner: FlowDocumentOwner
    var body: some View {
        if let notice = owner.referenceNotice {
            Text(notice).font(.caption).foregroundStyle(MLTheme.muted)
                .frame(maxWidth: .infinity, alignment: .leading).padding(8).background(MLTheme.panel)
        }
        if let notice = owner.recoveryNotice {
            HStack {
                Text(notice).font(.caption).textSelection(.enabled)
                Spacer()
                if !owner.recoveryCandidates.isEmpty {
                    Menu("Recovered sessions") {
                        ForEach(owner.recoveryCandidates, id: \.sessionID) { record in
                            Button("Restore session " + String(record.sessionID.uuidString.prefix(8))) { _ = owner.restoreRecovery(record) }
                                .disabled(owner.isDirty || (try? FlowDocumentCodec.reference(owner.document)) != record.base)
                        }
                    }
                }
            }.padding(10).background(MLTheme.panel)
        }
    }
}
