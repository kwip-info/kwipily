import Foundation
import Darwin

/// Only explicitly disposable capture stages participate. History, operations,
/// source snapshots, projects and authoring/publication records are never scanned.
public enum WorkflowScratchCleanup {
    public static let namespaces = ["preview-staging", "preparation-staging", "source-inspection-staging", "preparation-inspection-staging", "model-training-staging", "model-download-staging"]
    private static let receiptName = ".kwip-scratch.json"
    private static let lockName = "owner.lock"
    private static let maximumEntries = 10_000
    private struct Receipt: Codable {
        let version: Int
        let directory: WorkflowFileIdentity
        let lock: WorkflowFileIdentity
    }
    public final class Lifetime {
        private let lease: WorkflowFileLease
        fileprivate init(_ lease: WorkflowFileLease) { self.lease = lease }
        public func close() { lease.close() }
    }
    struct Entry: Equatable {
        let parts: [String]
        let identity: WorkflowFileIdentity
        let size: Int64
        let directory: Bool
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        let changedSeconds: Int
        let changedNanoseconds: Int
    }
    public struct Candidate: Identifiable {
        public var id: String { namespace + "/" + name }
        public let namespace: String
        public let name: String
        public let bytes: Int64
        fileprivate let identity: WorkflowFileIdentity
        fileprivate let entries: [Entry]
    }
    public struct Report {
        public let candidates: [Candidate]
        public let retainedStages: Int
        public var reclaimableBytes: Int64 { candidates.reduce(0) { $0 + $1.bytes } }
    }
    public enum Failure: Error, LocalizedError {
        case unsafe, changed, limit
        public var errorDescription: String? {
            switch self {
            case .unsafe: return "Temporary files could not be safely verified. They were retained."
            case .changed: return "Temporary files changed or are in use. Refresh the cleanup list."
            case .limit: return "Temporary files exceed the bounded cleanup scan. They were retained."
            }
        }
    }
    /// Register after exclusive stage creation, before writing any payload. Hold
    /// the returned existing file-lease primitive through capture consumption.
    public static func register(directory: URL) throws -> Lifetime {
        guard namespaces.contains(directory.deletingLastPathComponent().lastPathComponent),
              UUID(uuidString: directory.lastPathComponent)?.uuidString == directory.lastPathComponent else { throw Failure.unsafe }
        let fd = try openDirectory(directory); defer { Darwin.close(fd) }
        let lease = try WorkflowFileLease(directory: directory, name: lockName, mode: .exclusive, create: true, createExclusive: true, requireLocalAPFS: true)
        do {
            guard lease.parentIdentity == identity(try metadata(fd)) else { throw Failure.unsafe }
            let receipt = try JSONEncoder().encode(Receipt(version: 1, directory: lease.parentIdentity, lock: lease.identity))
            let file = openat(fd, receiptName, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard file >= 0 else { throw Failure.unsafe }
            defer { Darwin.close(file) }
            try receipt.withUnsafeBytes { buffer in
                var offset = 0
                while offset < buffer.count {
                    let count = Darwin.write(file, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
                    guard count > 0 else { throw Failure.unsafe }; offset += count
                }
            }
            guard fsync(file) == 0, fsync(fd) == 0 else { throw Failure.unsafe }
            try lease.checkIdentity()
            return Lifetime(lease)
        } catch { lease.close(); throw error }
    }
    public static func scan(workspace: URL, checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> Report {
        let root = try openDirectory(workspace); defer { Darwin.close(root) }
        var candidates: [Candidate] = [], retained = 0, count = 0
        for namespace in namespaces {
            try checkCancellation()
            let parent = openat(root, namespace, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            if parent < 0 { if errno == ENOENT { continue }; throw Failure.unsafe }
            defer { Darwin.close(parent) }
            for name in try names(parent) {
                count += 1; guard count <= maximumEntries else { throw Failure.limit }
                guard UUID(uuidString: name)?.uuidString == name else { retained += 1; continue }
                let directory = workspace.appendingPathComponent(namespace).appendingPathComponent(name)
                do {
                    let (fd, lease) = try acquire(directory)
                    defer { Darwin.close(fd); lease.close() }
                    let entries = try tree(fd, checkCancellation: checkCancellation)
                    count += entries.count
                    guard count <= maximumEntries else { throw Failure.limit }
                    candidates.append(.init(namespace: namespace, name: name, bytes: entries.filter { !$0.directory }.reduce(0) { $0 + $1.size }, identity: identity(try metadata(fd)), entries: entries))
                } catch is CancellationError { throw CancellationError() }
                catch Failure.limit { throw Failure.limit }
                catch { retained += 1 }
            }
        }
        return Report(candidates: candidates, retainedStages: retained)
    }
    /// Revalidates the reviewed inventory under an exclusive lifetime lease.
    /// Cancellation/crash can leave a smaller disposable stage; a fresh scan can
    /// reclaim it. No recovery/history namespace is eligible even by caller input.
    @discardableResult public static func remove(_ candidates: [Candidate], workspace: URL,
        checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> Int64 {
        let root = try openDirectory(workspace); defer { Darwin.close(root) }
        var removed: Int64 = 0
        for candidate in candidates {
            try checkCancellation()
            guard namespaces.contains(candidate.namespace), UUID(uuidString: candidate.name)?.uuidString == candidate.name else { throw Failure.unsafe }
            let parent = openat(root, candidate.namespace, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard parent >= 0 else { throw Failure.unsafe }; defer { Darwin.close(parent) }
            let directory = workspace.appendingPathComponent(candidate.id)
            let (fd, lease) = try acquire(directory)
            defer { Darwin.close(fd); lease.close() }
            guard identity(try metadata(fd)) == candidate.identity,
                  try tree(fd, checkCancellation: checkCancellation) == candidate.entries else { throw Failure.changed }
            let inventory = Dictionary(uniqueKeysWithValues: candidate.entries.map { ($0.parts.joined(separator: "/"), $0) })
            for entry in candidate.entries.sorted(by: { $0.parts.count > $1.parts.count }) {
                try checkCancellation(); try lease.checkIdentity()
                var member = dup(fd); guard member >= 0 else { throw Failure.unsafe }
                defer { Darwin.close(member) }
                var ancestorParts: [String] = []
                for part in entry.parts.dropLast() {
                    ancestorParts.append(part)
                    guard let expected = inventory[ancestorParts.joined(separator: "/")], expected.directory else { throw Failure.changed }
                    let next = openat(member, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                    guard next >= 0 else { throw Failure.unsafe }
                    do {
                        guard identity(try metadata(next)) == expected.identity else { throw Failure.changed }
                    } catch { Darwin.close(next); throw error }
                    Darwin.close(member); member = next
                }
                let name = entry.parts.last!
                var statValue = stat()
                guard fstatat(member, name, &statValue, AT_SYMLINK_NOFOLLOW) == 0,
                      identity(statValue) == entry.identity,
                      (statValue.st_mode & S_IFMT) == (entry.directory ? S_IFDIR : S_IFREG),
                      statValue.st_uid == geteuid(), statValue.st_mode & 0o022 == 0,
                      entry.directory || (statValue.st_nlink == 1 && statValue.st_size == entry.size &&
                        statValue.st_mtimespec.tv_sec == entry.modifiedSeconds && statValue.st_mtimespec.tv_nsec == entry.modifiedNanoseconds &&
                        statValue.st_ctimespec.tv_sec == entry.changedSeconds && statValue.st_ctimespec.tv_nsec == entry.changedNanoseconds),
                      unlinkat(member, name, entry.directory ? AT_REMOVEDIR : 0) == 0 else { throw Failure.changed }
            }
            // Keep the receipt and lock until all disposable payload is gone.
            // A crash between the final control unlinks and rmdir can leave an
            // empty/unregistered stage. Future scans retain it rather than infer
            // ownership from its name; no recovery evidence is sacrificed.
            try lease.checkIdentity()
            var current = stat()
            guard fstatat(parent, candidate.name, &current, AT_SYMLINK_NOFOLLOW) == 0,
                  identity(current) == candidate.identity,
                  Set(try names(fd)) == [receiptName, lockName] else { throw Failure.changed }
            guard unlinkat(fd, receiptName, 0) == 0, unlinkat(fd, lockName, 0) == 0,
                  unlinkat(parent, candidate.name, AT_REMOVEDIR) == 0 else { throw Failure.unsafe }
            removed += candidate.bytes
        }
        return removed
    }
    private static func acquire(_ directory: URL) throws -> (Int32, WorkflowFileLease) {
        let fd = try openDirectory(directory)
        do {
            let file = openat(fd, receiptName, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
            guard file >= 0 else { throw Failure.unsafe }; defer { Darwin.close(file) }
            let info = try metadata(file)
            guard info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1, (1...4096).contains(info.st_size) else { throw Failure.unsafe }
            var bytes = [UInt8](repeating: 0, count: Int(info.st_size))
            let size = bytes.count
            guard Darwin.read(file, &bytes, size) == size else { throw Failure.unsafe }
            let receipt = try JSONDecoder().decode(Receipt.self, from: Data(bytes))
            guard receipt.version == 1, receipt.directory == identity(try metadata(fd)) else { throw Failure.unsafe }
            let lease = try WorkflowFileLease(directory: directory, name: lockName, mode: .exclusive, expectedIdentity: receipt.lock, requireLocalAPFS: true)
            guard lease.parentIdentity == receipt.directory else { lease.close(); throw Failure.unsafe }
            return (fd, lease)
        } catch { Darwin.close(fd); throw error }
    }
    private static func tree(_ root: Int32, checkCancellation: () throws -> Void) throws -> [Entry] {
        var result: [Entry] = []
        func walk(_ fd: Int32, parts: [String]) throws {
            guard parts.count < 16 else { throw Failure.limit }
            for name in try names(fd).sorted() {
                try checkCancellation()
                if parts.isEmpty && [receiptName, lockName].contains(name) { continue }
                guard result.count < maximumEntries else { throw Failure.limit }
                var info = stat()
                guard fstatat(fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0, info.st_uid == geteuid(), info.st_mode & 0o022 == 0,
                      info.st_size >= 0, info.st_size <= 2_000_000_000 else { throw Failure.unsafe }
                let isDirectory = info.st_mode & S_IFMT == S_IFDIR
                guard isDirectory || (info.st_mode & S_IFMT == S_IFREG && info.st_nlink == 1) else { throw Failure.unsafe }
                let path = parts + [name]
                result.append(.init(parts: path, identity: identity(info), size: Int64(info.st_size), directory: isDirectory,
                    modifiedSeconds: info.st_mtimespec.tv_sec, modifiedNanoseconds: info.st_mtimespec.tv_nsec,
                    changedSeconds: info.st_ctimespec.tv_sec, changedNanoseconds: info.st_ctimespec.tv_nsec))
                if isDirectory {
                    let child = openat(fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                    guard child >= 0 else { throw Failure.unsafe }; defer { Darwin.close(child) }
                    guard identity(try metadata(child)) == identity(info) else { throw Failure.changed }
                    try walk(child, parts: path)
                }
            }
        }
        try walk(root, parts: [])
        return result
    }
    private static func identity(_ info: stat) -> WorkflowFileIdentity { .init(device: UInt64(UInt32(bitPattern: info.st_dev)), inode: UInt64(info.st_ino)) }
    private static func metadata(_ fd: Int32) throws -> stat {
        var value = stat()
        guard fstat(fd, &value) == 0, value.st_uid == geteuid(), value.st_mode & 0o022 == 0 else { throw Failure.unsafe }; return value
    }
    private static func openDirectory(_ url: URL) throws -> Int32 {
        guard url.isFileURL, !url.path.contains("\0"), !url.path.split(separator: "/").contains("..") else { throw Failure.unsafe }
        let fd = open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_CLOEXEC)
        guard fd >= 0 else { throw Failure.unsafe }
        do { _ = try metadata(fd); return fd } catch { Darwin.close(fd); throw error }
    }
    private static func names(_ fd: Int32) throws -> [String] {
        let copy = openat(fd, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard copy >= 0 else { throw Failure.unsafe }
        guard let stream = fdopendir(copy) else { Darwin.close(copy); throw Failure.unsafe }; defer { closedir(stream) }
        var values: [String] = []
        errno = 0
        while let item = readdir(stream) {
            let name = withUnsafePointer(to: &item.pointee.d_name) { $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) } }
            if name != "." && name != ".." { values.append(name) }
            guard values.count <= maximumEntries else { throw Failure.limit }; errno = 0
        }
        guard errno == 0 else { throw Failure.unsafe }; return values
    }
}
