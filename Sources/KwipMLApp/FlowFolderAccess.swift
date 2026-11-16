import Foundation

/// Read/write document-folder grants are distinct from read-only dataset-source grants.
@MainActor final class FlowFolderAccess {
    struct Hooks {
        let createBookmark: (URL) throws -> Data
        let resolveBookmark: (Data) throws -> (URL, Bool)
        let start: (URL) -> Bool
        let stop: (URL) -> Void
        static let system = Hooks(createBookmark: {
            try $0.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
        }, resolveBookmark: {
            var stale = false
            let url = try URL(resolvingBookmarkData: $0, options: [.withSecurityScope, .withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale)
            return (url, stale)
        }, start: { $0.startAccessingSecurityScopedResource() }, stop: { $0.stopAccessingSecurityScopedResource() })
    }
    /// Share this reference with every root/child editor. The final holder releases the scope.
    final class Lease {
        let locationID: UUID
        let folderURL: URL
        let rootURL: URL
        private let grantedURL: URL
        private let started: Bool
        private let stop: (URL) -> Void
        fileprivate init(locationID: UUID, folderURL: URL, rootFilename: String, grantedURL: URL,
                         started: Bool, stop: @escaping (URL) -> Void) {
            self.locationID = locationID; self.folderURL = folderURL
            rootURL = folderURL.appendingPathComponent(rootFilename)
            self.grantedURL = grantedURL; self.started = started; self.stop = stop
        }
        deinit { if started { stop(grantedURL) } }
    }
    struct Grant { let lease: Lease; let bookmarkData: Data }
    private let hooks: Hooks
    init(hooks: Hooks = .system) { self.hooks = hooks }

    func grant(folder: URL, rootFilename: String, locationID: UUID) throws -> Grant {
        let lease = try acquire(folder: folder, rootFilename: rootFilename, locationID: locationID)
        let bookmark = try hooks.createBookmark(folder)
        guard !bookmark.isEmpty, bookmark.count <= 1_000_000 else { throw Failure("The folder permission could not be saved. Choose the flow folder again.") }
        return Grant(lease: lease, bookmarkData: bookmark)
    }
    func open(locationID: UUID, folderPath: String, rootFilename: String, bookmarkData: Data) throws -> Lease {
        guard !bookmarkData.isEmpty, bookmarkData.count <= 1_000_000 else { throw Failure("Saved folder permission is invalid. Choose the flow folder again.") }
        let (folder, stale) = try hooks.resolveBookmark(bookmarkData)
        guard !stale else { throw Failure("Saved folder permission needs renewal. Choose the flow folder again.") }
        let lease = try acquire(folder: folder, rootFilename: rootFilename, locationID: locationID)
        guard lease.folderURL.path == folderPath else { throw Failure("The flow folder moved. Choose its location explicitly before reopening it.") }
        return lease
    }
    private func acquire(folder: URL, rootFilename: String, locationID: UUID) throws -> Lease {
        guard folder.isFileURL, Self.validRootFilename(rootFilename) else { throw Failure("Choose a flow file directly inside its containing folder.") }
        let started = hooks.start(folder)
        do {
            let canonical = folder.resolvingSymlinksInPath().standardizedFileURL
            let values = try canonical.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory == true else { throw Failure("The saved flow folder is unavailable. Choose it again.") }
            // A false start is normal outside the sandbox; actual file I/O still enforces access.
            return Lease(locationID: locationID, folderURL: canonical, rootFilename: rootFilename,
                         grantedURL: folder, started: started, stop: hooks.stop)
        } catch {
            if started { hooks.stop(folder) }
            throw error
        }
    }
    static func validRootFilename(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 255 && !name.hasPrefix(".") && !name.contains("/") && !name.contains("\\") && !name.contains("\0") && (name as NSString).pathExtension.lowercased() == "kwipflow"
    }
    struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
