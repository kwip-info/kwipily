import Foundation
import AppKit
import Combine
import KwipMLCore

/// A parent-retained native child host. Authority and history stay with the workflow.
@MainActor final class WorkflowChildWorkspace:ObservableObject {
    private weak var workspace:WorkspaceStore?
    private weak var host:WorkflowWorkspaceSession?
    @Published private(set) var session:ModelChildEditorSession?
    @Published private var localError:String?
    var error:String? {localError ?? session?.error}
    /// Request presentation uses only this current open/flush failure, not an old child editor error.
    var openingError: String? { localError }
    /// Tests inject session.open here; production presents the existing native controller.
    var present:((ModelChildEditorSession,[String])->Bool)?
    var presentFlow: ((URL, FlowDocumentRevision) -> Bool)?
    private var controller:ModelChildWindowController?
    var restorationWindow: NSWindow? { controller?.restorationWindow }
    private var observation:AnyCancellable?
    private var active=true
    init(workspace:WorkspaceStore,host:WorkflowWorkspaceSession) {
        self.workspace=workspace;self.host=host
    }
    private var isCurrent:Bool {
        guard active,let workspace,let host else {return false}
        return workspace.workflowSession === host && workspace.activeProjectID == host.owner.projectID
            && !host.isRetired && host.owner.isCurrent(host.owner.lease())
    }
    @discardableResult func openPrivateCopy(address: WorkflowNodeAddress, lease: WorkflowEditorLease,
                                             nestedPath: [String] = [], presentWindow: Bool = true) -> Bool {
        open(address: address, lease: lease, nestedPath: nestedPath, presentWindow: presentWindow, privateCopy: true)
    }
    @discardableResult func open(address:WorkflowNodeAddress,lease:WorkflowEditorLease,nestedPath:[String]=[],presentWindow:Bool=true,privateCopy:Bool=false)->Bool {
        do {
            guard isCurrent,let workspace,let host,host.owner.isCurrent(lease),!host.owner.isBusy else {
                throw DataFlowContractError("This Block request belongs to a previous workflow state, or execution is active.")
            }
            let hasDraft = host.owner.document.drafts.first?.instancePath.first == address.localID
            if let rootURL = host.owner.flowRootURL, !privateCopy, !hasDraft {
                let target = try fileTarget(owner: host.owner, rootURL: rootURL, address: address, nestedPath: nestedPath)
                if let presentFlow {
                    guard presentFlow(target.0, target.1) else { throw FlowDocumentError.invalidDocument }
                } else { try FlowCanvasWindowRegistry.shared.open(target.0, present: presentWindow, callerPin: target.1, accessLease: workspace.externalFolderLease,
                    restorationTarget: .subflow(root: workspace.externalLocation.map { .external($0.id) } ?? .managed(UUID(uuidString: host.owner.projectID)!),
                        path: try CanvasRestorationPaths.relative(target.0, inside: rootURL.deletingLastPathComponent()), documentID: target.1.documentID)) }
                localError = nil; return true
            }
            let authority=try WorkflowBlockCanvasAuthority(owner:host.owner,root:address)
            var path=[address.localID]+nestedPath
            if let draft=host.owner.document.drafts.first {
                guard draft.instancePath.first == address.localID else {
                    throw DataFlowContractError("Resolve the current Block draft before opening another root.")
                }
                if nestedPath.isEmpty {path=draft.instancePath}
                guard draft.instancePath == path else {throw DataFlowContractError("Reopen the persisted Block draft before changing its scope.")}
            }
            if let current=session,let borrowed=current.owner as? WorkflowBlockCanvasAuthority,borrowed.root == address {
                guard show(current,path:path,presentWindow:presentWindow) else {localError=current.error ?? "The Block editor could not be opened.";return false}
                localError=nil;return true
            }
            guard flushPresentation() else {return false}
            let next=ModelChildEditorSession(owner:authority)
            next.presentation=workspace.appPreferences.canvasDefaults
            // Install before opening so configurePresentation can verify exact ownership.
            controller?.invalidate();controller=nil;session?.invalidate()
            session=next
            observation=next.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
            workspace.bindChildCanvasPersistence(next,isCurrent:{[weak self,weak next] in
                guard let self,let next else {return false}
                return self.isCurrent && self.session === next
            })
            guard show(next,path:path,presentWindow:presentWindow) else {localError=next.error ?? "The Block editor could not be opened.";return false}
            localError=nil;return true
        } catch {localError=error.localizedDescription;return false}
    }
    private func fileTarget(owner: WorkflowAuthoringStore, rootURL: URL, address: WorkflowNodeAddress,
                            nestedPath: [String]) throws -> (URL, FlowDocumentRevision) {
        guard let root = owner.pinnedFlowDocument,
              let node = owner.document.nodes.first(where: { $0.address == address }),
              case .modelBlock(let initial, _) = node.payload else { throw FlowDocumentError.missingDependency }
        let snapshot = try FlowDocumentStore.snapshot(at: rootURL)
        guard snapshot.root == (try FlowDocumentCodec.reference(root)) else { throw FlowDocumentError.revisionMismatch }
        var byPin: [FlowDocumentRevision: FlowDocument] = [:]
        for document in snapshot.documents { byPin[try FlowDocumentCodec.reference(document)] = document }
        var parent = root, location = rootURL, reference = initial, remaining = nestedPath
        while true {
            guard let dependency = parent.dependencies.first(where: { $0.definition == reference }),
                  let child = byPin[dependency.revision] else { throw FlowDocumentError.missingDependency }
            let target = location.deletingLastPathComponent().appendingPathComponent(dependency.relativePath)
            if remaining.isEmpty { return (target, dependency.revision) }
            let id = remaining.removeFirst()
            guard case .modelBlock(let definition) = child.payload,
                  let instance = definition.body.instances.first(where: { $0.id == id }) else { throw FlowDocumentError.missingDependency }
            parent = child; location = target; reference = instance.definition
        }
    }
    private func show(_ session:ModelChildEditorSession,path:[String],presentWindow:Bool)->Bool {
        if let present {return present(session,path)}
        if controller == nil, let workspace {
            let window = ModelChildWindowController(session:session, appearance:workspace.appearanceState)
            window.onDraftResolved = { [weak self, weak session] in
                guard let self, let session, self.isCurrent, self.session === session else { return }
                self.workspace?.focusOwningProject?()
            }
            window.capturedOutput = { [weak self, weak session] path, node in
                guard let self, let session, self.isCurrent, self.session === session,
                      let authority = session.owner as? WorkflowBlockCanvasAuthority,
                      path.first == authority.root.localID else { return nil }
                return self.host?.modelSnapshot(for:.init(domain:.model,localID:node),instancePath:path)
            }
            controller = window
        }
        return controller?.open(path:path,present:presentWindow) == true
    }
    @discardableResult func flushPresentation()->Bool {
        guard session != nil else {return true}
        guard isCurrent else {localError="This child editor belongs to a previous workflow session.";return false}
        guard session?.flushPresentation() != false else {localError=session?.error;return false}
        localError=nil;return true
    }
    func restorePresentation() {if isCurrent {session?.restorePresentation()}}
    @discardableResult func close()->Bool {
        guard session != nil else {return true}
        guard isCurrent,session?.close() == true else {localError=session?.error ?? "The child editor could not be closed.";return false}
        controller?.invalidate();controller=nil
        session?.invalidate();session=nil;observation=nil;localError=nil;return true
    }
    func invalidate() {
        active=false;controller?.invalidate();controller=nil
        session?.invalidate();session=nil;observation=nil;present=nil;presentFlow=nil
    }
}
