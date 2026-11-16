import Foundation
import KwipMLCore

/// Main-actor, synchronous restoration scope. This caches discovery metadata,
/// never document bytes, file access authority or an editor owner.
@MainActor final class RestorationCatalogSnapshot {
    static let shared = RestorationCatalogSnapshot()
    private var depth = 0
    private var listings: [URL: Result<[ProjectRecord], Error>] = [:]
    var isActive: Bool { depth > 0 }
    func begin() {
        if depth == 0 { listings.removeAll() }
        depth += 1
    }
    func end() {
        guard depth > 0 else { return }
        depth -= 1
        if depth == 0 { listings.removeAll() }
    }
    func withBatch<T>(_ body: () throws -> T) rethrows -> T {
        begin()
        defer { end() }
        return try body()
    }
    private func key(_ identity: URL) -> URL {
        identity.standardizedFileURL.resolvingSymlinksInPath()
    }
    func records(identity: URL, load: () throws -> [ProjectRecord]) throws -> [ProjectRecord] {
        guard isActive else { return try load() }
        let identity = key(identity)
        if let cached = listings[identity] { return try cached.get() }
        let result = Result(catching: load)
        listings[identity] = result
        return try result.get()
    }
    func invalidate(identity: URL) { listings.removeValue(forKey: key(identity)) }
}
