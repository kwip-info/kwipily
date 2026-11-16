import Foundation

/// Local UI restoration only: references existing authorities, never embeds graphs, grants or run data.
struct CanvasTabSession: Codable, Equatable {
    enum Root: Codable, Hashable {
        case managed(UUID)
        case external(UUID)
    }
    enum InlineKind: String, Codable, Hashable { case model, workflow }
    enum Target: Codable, Hashable {
        case library
        case managed(UUID)
        case external(UUID)
        case subflow(root: Root, path: String, documentID: UUID)
        case standalone(locationID: UUID, path: String, documentID: UUID)
        case inlineScope(root: Root, kind: InlineKind, path: [String])
    }
    struct Frame: Codable, Equatable {
        let x: Double, y: Double, width: Double, height: Double
        var isValid: Bool {
            [x, y, width, height].allSatisfy(\.isFinite) && abs(x) <= 100_000 && abs(y) <= 100_000 &&
                (100...32_000).contains(width) && (100...32_000).contains(height)
        }
    }
    struct Group: Codable, Equatable {
        let tabs: [Target]
        let selected: Int
        var frame: Frame? = nil
    }
    let schemaVersion: Int
    let groups: [Group]
    let activeGroup: Int?
    init(groups: [Group], activeGroup: Int?) {
        schemaVersion = 1; self.groups = groups; self.activeGroup = activeGroup
    }
    func validate() throws {
        guard schemaVersion == 1, groups.count <= 64,
              groups.reduce(0, { $0 + $1.tabs.count }) <= 128,
              groups.isEmpty ? activeGroup == nil : activeGroup.map({ groups.indices.contains($0) }) == true else {
            throw Failure("The saved canvas session uses an unsupported version or window layout. Its file has been preserved.")
        }
        var identities = Set<Target>()
        var paths = Set<String>()
        var inlineRoots = Set<Root>()
        for group in groups {
            guard !group.tabs.isEmpty, group.tabs.indices.contains(group.selected), group.frame?.isValid != false else {
                throw Failure("A saved canvas window has an invalid selection or frame.")
            }
            for target in group.tabs {
                guard identities.insert(target).inserted else { throw Failure("A canvas appears more than once in the saved session.") }
                if case .inlineScope(let root, _, let path) = target {
                    guard inlineRoots.insert(root).inserted, (1...16).contains(path.count),
                          path.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 && !$0.contains("\0") }) else {
                        throw Failure("A saved inline Block scope is invalid or duplicated for its parent project.")
                    }
                }
                let pathAndKey: (String, String)?
                switch target {
                case .subflow(let root, let path, _):
                    let key: String
                    switch root { case .managed(let id): key = "managed:" + id.uuidString
                    case .external(let id): key = "external:" + id.uuidString }
                    pathAndKey = (path, key)
                case .standalone(let id, let path, _): pathAndKey = (path, "standalone:" + id.uuidString)
                default: pathAndKey = nil
                }
                if let (path, rootKey) = pathAndKey {
                    let parts = path.split(separator: "/", omittingEmptySubsequences: false)
                    guard path.utf8.count <= 2_048, !path.contains("\\"), !path.contains("\0"),
                          !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                          (path as NSString).pathExtension.lowercased() == "kwipflow" else {
                        throw Failure("A saved subflow must name a relative flow file inside its project folder.")
                    }
                    guard paths.insert(rootKey + ":" + path).inserted else {
                        throw Failure("Conflicting document identities refer to the same saved subflow path.")
                    }
                }
            }
        }
    }
    struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}

struct CanvasTabSessionStore {
    let directory: URL
    private let parts = ["canvas-session.json"]
    private let maximumBytes = 256_000
    func load() throws -> CanvasTabSession? {
        guard let bytes = try DiagnosticFileIO.readIfPresent(root: directory, parts: parts, limit: maximumBytes) else { return nil }
        let session = try JSONDecoder().decode(CanvasTabSession.self, from: bytes)
        try session.validate()
        return session
    }
    func save(_ session: CanvasTabSession) throws {
        try session.validate()
        // Unknown/corrupt versions are not silently replaced by a fresh empty launch.
        _ = try load()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(session)
        guard bytes.count <= maximumBytes else { throw CanvasTabSession.Failure("The canvas session exceeds its saved-layout budget.") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try DiagnosticFileIO.writeAtomic(bytes, root: directory, parts: parts)
    }
}
