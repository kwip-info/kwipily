import AppKit
import UniformTypeIdentifiers

@MainActor extension ProjectWindowRegistry {
    @discardableResult func relinkExternal(_ record: ExternalFlowCatalog.Record, folder: URL, rootURL: URL,
                                           children: FlowCanvasWindowRegistry = .shared) throws -> ExternalFlowCatalog.Record {
        let oldFolder = URL(fileURLWithPath: record.folderPath, isDirectory: true).resolvingSymlinksInPath().standardizedFileURL.path + "/"
        guard !openControllers.contains(where: { $0.session.workspace.externalLocation?.id == record.id }),
              !children.openControllers.contains(where: {
                  $0.accessLease?.locationID == record.id || $0.documentOwner.url.resolvingSymlinksInPath().standardizedFileURL.path.hasPrefix(oldFolder)
              }) else {
            throw FlowFolderAccess.Failure("Close this flow and subflow tabs in its current folder before changing its location. Unsaved edits remain in their current tabs.")
        }
        let destination = rootURL.resolvingSymlinksInPath().standardizedFileURL
        guard !openControllers.contains(where: { $0.window?.representedURL?.resolvingSymlinksInPath().standardizedFileURL == destination }) else {
            throw FlowFolderAccess.Failure("The selected file already has an open canvas. Close that canvas before changing this linked location.")
        }
        guard try !restorationWorkspace.projectLibrary.listSources().contains(where: {
            URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().standardizedFileURL == destination
        }) else {
            throw FlowFolderAccess.Failure("This file is already a managed project. Open it from Recent projects; its history cannot be merged with a linked copy.")
        }
        let updated = try restorationCatalog.relink(record, folder: folder, rootURL: rootURL)
        restorationWorkspace.refreshProjects()
        return updated
    }
    func locateExternalFlow(_ record: ExternalFlowCatalog.Record, workspace: WorkspaceStore) {
        let file = NSOpenPanel(); file.canChooseFiles = true; file.canChooseDirectories = false; file.allowsMultipleSelection = false
        file.allowedContentTypes = [UTType(filenameExtension: "kwipflow") ?? .data]
        file.title = "Locate moved flow"; file.prompt = "Choose flow"
        file.message = "Choose this project's workflow file. Its local run history will stay linked to this entry."
        guard file.runModal() == .OK, let root = file.url else { return }
        let folder = NSOpenPanel(); folder.canChooseFiles = false; folder.canChooseDirectories = true; folder.allowsMultipleSelection = false
        folder.title = "Allow access to flow folder"; folder.prompt = "Use flow folder"
        folder.directoryURL = root.deletingLastPathComponent()
        folder.message = "Choose the containing folder to access this flow and its subflow files."
        guard folder.runModal() == .OK, let directory = folder.url else { return }
        do {
            _ = try relinkExternal(record, folder: directory, rootURL: root)
            workspace.refreshProjects()
            workspace.statusMessage = "Flow location updated. Local history is retained; open the flow to continue."
        } catch {
            workspace.statusMessage = error.localizedDescription
            NSAlert(error: error).runModal()
        }
    }
}
