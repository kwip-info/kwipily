import AppKit
import KwipMLCore

@MainActor extension ProjectWindowRegistry {
    var standaloneCatalog: StandaloneFlowCatalog {
        StandaloneFlowCatalog(directory: restorationWorkspace.applicationDirectory.appendingPathComponent("standalone-flow-catalog"))
    }
    func makeRestorationCoordinator() -> CanvasTabSessionCoordinator {
        CanvasTabSessionCoordinator(store: .init(directory: restorationWorkspace.applicationDirectory.appendingPathComponent("canvas-session")), bindings: { [weak self] in
            guard let self else { return [] }
            var result: [CanvasTabSessionCoordinator.Binding] = []
            for controller in self.openControllers {
                guard let window = controller.window, let id = controller.session.workspace.activeProjectID.flatMap(UUID.init(uuidString:)) else { continue }
                let target = controller.session.workspace.externalLocation.map { CanvasTabSession.Target.external($0.id) } ?? .managed(id)
                result.append(.init(target: target, window: window))
                let root = controller.session.workspace.externalLocation.map { CanvasTabSession.Root.external($0.id) } ?? .managed(id)
                let workspace = controller.session.workspace
                if let child = workspace.workflowChildren, let childWindow = child.restorationWindow, let path = child.session?.editor?.path {
                    result.append(.init(target: .inlineScope(root: root, kind: .workflow, path: path), window: childWindow))
                }
                if let child = workspace.modelChildWindows, let childWindow = child.restorationWindow, let path = child.session.editor?.path {
                    result.append(.init(target: .inlineScope(root: root, kind: .model, path: path), window: childWindow))
                }
            }
            for controller in FlowCanvasWindowRegistry.shared.openControllers {
                guard let window = controller.window else { continue }
                guard let target = controller.restorationTarget else {
                    throw CanvasTabSession.Failure("The subflow \(controller.canvas.title) has no saved location permission. Reopen it with File → Open subflow document before saving this window session.")
                }
                result.append(.init(target: target, window: window))
            }
            return result
        }, resolve: { [weak self] target in
            guard let self else { throw CanvasTabSession.Failure("The project launcher is unavailable.") }
            return try self.restoreCanvas(target)
        }, presentsWindows: restorationPresentsWindows, includesLibrary: false,
           beginRestorationBatch: { RestorationCatalogSnapshot.shared.begin() },
           endRestorationBatch: { [weak self] in
               // Hydrate every owner once while discovery is still batch-scoped.
               // Catalog grants and actual source loads remain live operations.
               defer { RestorationCatalogSnapshot.shared.end() }
               guard let self else { return }
               for controller in self.openControllers { controller.session.workspace.refreshProjects() }
               self.restorationWorkspace.refreshProjects()
           })
    }
    func restoreCanvas(_ target: CanvasTabSession.Target) throws -> NSWindow {
        switch target {
        case .library:
            guard let window = restorationLauncherWindow else { throw CanvasTabSession.Failure("The project launcher is not ready.") }; return window
        case .managed(let id):
            guard let record = try restorationWorkspace.projectCatalogRecords().first(where: { UUID(uuidString: $0.id) == id }),
                  let window = try open(record, present: false).window else { throw CanvasTabSession.Failure("The saved managed project is unavailable. Its target was retained.") }
            return window
        case .external(let id):
            guard let record = try restorationCatalog.list().first(where: { $0.id == id }),
                  let window = try openExternal(record, present: false).window else { throw CanvasTabSession.Failure("The saved external project location is unavailable.") }
            return window
        case .subflow(let root, let path, let documentID):
            let folder: URL, lease: FlowFolderAccess.Lease?
            switch root {
            case .managed(let id):
                guard let record = try restorationWorkspace.projectCatalogRecords().first(where: { UUID(uuidString: $0.id) == id }),
                      (record.path as NSString).pathExtension.lowercased() == "kwipflow" else { throw CanvasTabSession.Failure("The subflow's managed project folder is unavailable.") }
                folder = URL(fileURLWithPath: record.path).deletingLastPathComponent(); lease = nil
            case .external(let id):
                guard let record = try restorationCatalog.list().first(where: { $0.id == id }) else { throw CanvasTabSession.Failure("The subflow's saved folder permission is unavailable.") }
                // Reopen the retained folder grant, without creating a root editor or
                // depending on the root file still being open (or present).
                let granted = try restorationCatalog.openFolder(record)
                folder = granted.folderURL; lease = granted
            }
            return try restoreSubflow(target, url: folder.appendingPathComponent(path), documentID: documentID, lease: lease)
        case .standalone(let id, let path, let documentID):
            let opened = try standaloneCatalog.open(id)
            return try restoreSubflow(target, url: opened.lease.folderURL.appendingPathComponent(path), documentID: documentID, lease: opened.lease)
        case .inlineScope(let root, let kind, let path):
            let rootTarget: CanvasTabSession.Target
            switch root { case .managed(let id): rootTarget = .managed(id); case .external(let id): rootTarget = .external(id) }
            let rootWindow = try restoreCanvas(rootTarget)
            guard let workspace = session(for: rootWindow)?.workspace, let first = path.first else { throw FlowDocumentError.invalidDocument }
            switch kind {
            case .model:
                guard let child = workspace.modelChildWindows, child.open(path: path, present: false), let window = child.restorationWindow else {
                    throw CanvasTabSession.Failure(workspace.modelChildWindows?.session.error ?? "The saved model Block scope is no longer available.")
                }
                return window
            case .workflow:
                guard let host = workspace.workflowSession,
                      host.owner.flowRootURL == nil || host.owner.document.drafts.first?.instancePath == path,
                      let child = workspace.workflowChildren,
                      child.open(address: .init(domain: .model, localID: first), lease: host.owner.lease(), nestedPath: Array(path.dropFirst()), presentWindow: false),
                      let window = child.restorationWindow else {
                    throw CanvasTabSession.Failure(workspace.workflowChildren?.error ?? "The saved private Block draft is no longer available. Open the reusable Block or explicitly start a new private copy.")
                }
                return window
            }
        }
    }
    private func restoreSubflow(_ target: CanvasTabSession.Target, url: URL, documentID: UUID, lease: FlowFolderAccess.Lease?) throws -> NSWindow {
        let document = try FlowDocumentStore.readHead(at: url)
        guard document.id == documentID else { throw CanvasTabSession.Failure("The saved subflow path now contains another document. Its target was retained.") }
        guard let window = try FlowCanvasWindowRegistry.shared.open(url, present: false, accessLease: lease, restorationTarget: target).window else { throw FlowDocumentError.invalidDocument }
        return window
    }
    func openStandaloneSubflow(_ url: URL) throws {
        if let existing = try standaloneCatalog.list().first(where: { $0.url.path == url.resolvingSymlinksInPath().path }),
           let opened = try? standaloneCatalog.open(existing.id) {
            _ = try FlowCanvasWindowRegistry.shared.open(opened.lease.rootURL, accessLease: opened.lease,
                restorationTarget: .standalone(locationID: existing.id, path: existing.filename, documentID: existing.documentID))
            return
        }
        guard let record = try chooseStandaloneFolder(for: url) else { return }
        let opened = try standaloneCatalog.open(record.id)
        _ = try FlowCanvasWindowRegistry.shared.open(opened.lease.rootURL, accessLease: opened.lease,
            restorationTarget: .standalone(locationID: record.id, path: record.filename, documentID: record.documentID))
    }
    func renewStandaloneAccess(_ target: CanvasTabSession.Target) {
        guard case .standalone(let id, _, _) = target else { return }
        do {
            guard let record = try standaloneCatalog.list().first(where: { $0.id == id }) else { throw CanvasTabSession.Failure("The saved subflow location is unavailable.") }
            if try chooseStandaloneFolder(for: record.url) != nil { restoration?.retry() }
        } catch { NSAlert(error: error).runModal() }
    }
    private func chooseStandaloneFolder(for url: URL) throws -> StandaloneFlowCatalog.Record? {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.allowsMultipleSelection = false
        panel.title = "Allow access to subflow folder"
        panel.message = "Choose the folder containing this subflow to reopen it and its nested documents on your next launch."
        panel.directoryURL = url.deletingLastPathComponent(); panel.prompt = "Open subflow folder"
        guard panel.runModal() == .OK, let folder = panel.url else { return nil }
        return try standaloneCatalog.attach(folder: folder, file: url)
    }

}

enum CanvasRestorationPaths {
    static func relative(_ url: URL, inside folder: URL) throws -> String {
        let base = folder.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        let path = url.resolvingSymlinksInPath().standardizedFileURL.path
        guard path.hasPrefix(base) else { throw FlowDocumentError.unsafeReference }
        return String(path.dropFirst(base.count))
    }
}
@MainActor extension FlowCanvasWindowController {
    func targetForDependency(_ url: URL, documentID: UUID) throws -> CanvasTabSession.Target? {
        guard let restorationTarget else { return nil }
        let currentPath: String
        switch restorationTarget {
        case .subflow(_, let path, _), .standalone(_, let path, _): currentPath = path
        default: throw FlowDocumentError.unsafeReference
        }
        var folder = documentOwner.url
        for _ in currentPath.split(separator: "/") { folder.deleteLastPathComponent() }
        let relative = try CanvasRestorationPaths.relative(url, inside: folder)
        switch restorationTarget {
        case .subflow(let root, _, _): return .subflow(root: root, path: relative, documentID: documentID)
        case .standalone(let id, _, _): return .standalone(locationID: id, path: relative, documentID: documentID)
        default: throw FlowDocumentError.unsafeReference
        }
    }
}
