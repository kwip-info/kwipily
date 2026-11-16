import Foundation
import CryptoKit
import Darwin
import KwipMLCore

/// One private broker destination. Its ledger reservation spans actual child exit,
/// streamed integrity verification, and the final source-binding authority check.
final class DatabaseSourceSnapshot: @unchecked Sendable {
    static let maximumBytes = 10_000_000
    let destination: URL
    let output: FileHandle
    private let root: URL
    private let name: String
    private let rootIdentity: WorkflowFileIdentity
    private let directoryIdentity: WorkflowFileIdentity
    private let fileIdentity: WorkflowFileIdentity
    private var namespaceFD: Int32
    private var directoryFD: Int32
    private var reservation: WorkflowStorageAdmission.Reservation?
    private var closed = false

    init(workspace: URL, accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil,
         identifier: UUID = UUID(), checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws {
        try checkCancellation()
        guard let pointer = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        root = URL(fileURLWithPath: String(cString: pointer), isDirectory: true); free(pointer)
        let captureName = identifier.uuidString
        name = captureName
        let admission = try WorkflowStorageAdmission(workspace: root, accountingRoot: accountingRoot)
        let scope = try WorkflowStorageAdmission.Scope(namespace: .databaseSourceSnapshots, components: [name])
        let token = try storageLimitBytes.map { try admission.reserve(scopes: [scope], peakBytes: Int64(Self.maximumBytes), limitBytes: $0) }
        var retained = false
        defer { if !retained { token?.close() } }
        let rootFD = try WorkflowEvaluationOwnership.openDirectory(root)
        defer { Darwin.close(rootFD) }
        rootIdentity = try WorkflowEvaluationOwnership.directoryIdentity(rootFD)
        let namespace = WorkflowStorageAdmission.Namespace.databaseSourceSnapshots.rawValue
        if mkdirat(rootFD, namespace, 0o700) != 0, errno != EEXIST { throw WorkflowFileLeaseError.system(errno) }
        let parent = openat(rootFD, namespace, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parent >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        var keepParent = false
        defer { if !keepParent { Darwin.close(parent) } }
        _ = try WorkflowEvaluationOwnership.directoryIdentity(parent)
        guard mkdirat(parent, name, 0o700) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        var keepDirectory = false
        defer { if !keepDirectory { _ = unlinkat(parent, captureName, AT_REMOVEDIR) } }
        let leaf = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard leaf >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { if !keepDirectory { Darwin.close(leaf) } }
        directoryIdentity = try WorkflowEvaluationOwnership.directoryIdentity(leaf)
        let fd = openat(leaf, "database.arrow", O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        var keepFile = false
        defer { if !keepFile { Darwin.close(fd); _ = unlinkat(leaf, "database.arrow", 0) } }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_nlink == 1, info.st_size == 0 else { throw WorkflowFileLeaseError.unsafePath }
        fileIdentity = WorkflowEvaluationOwnership.identity(info)
        destination = root.appendingPathComponent(namespace).appendingPathComponent(name).appendingPathComponent("database.arrow")
        try checkCancellation()
        output = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        namespaceFD = parent; directoryFD = leaf; reservation = token
        keepParent = true; keepDirectory = true; keepFile = true; retained = true
    }
    /// Run off the main actor. No complete 10 MB byte array is allocated.
    func verify(receipt: Data, checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws {
        try checkCancellation()
        guard !closed, receipt.count <= 65_536,
              case .object(let record) = try JSONDecoder().decode(JSONValue.self, from: receipt),
              record["schemaVersion"] == 1, record["kind"] == .string("databaseCapture"), record["format"] == .string("ipc"),
              record["sampleOnly"] == false, case .string(let expectedHash) = record["sha256"], WorkflowArchiveCodec.isHash(expectedHash),
              case .integer(let expectedBytes) = record["bytes"], (1...Int64(Self.maximumBytes)).contains(expectedBytes),
              case .integer(let rows) = record["rows"], case .integer(let columns) = record["columns"],
              rows >= 0, (1...128).contains(columns), rows <= 5_000_000 / columns else { throw WorkflowFileLeaseError.unsafePath }
        try output.synchronize()
        try checkLocation()
        let fd = openat(directoryFD, "database.arrow", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(fd) }
        var before = stat()
        guard fstat(fd, &before) == 0, before.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), before.st_nlink == 1,
              before.st_uid == geteuid(), before.st_mode & 0o077 == 0, before.st_size == expectedBytes,
              WorkflowEvaluationOwnership.identity(before) == fileIdentity else { throw WorkflowFileLeaseError.unsafePath }
        var hasher = SHA256(), count = 0, buffer = [UInt8](repeating: 0, count: 65_536)
        while true {
            try checkCancellation()
            let size = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress!, min($0.count, Self.maximumBytes + 1 - count)) }
            if size < 0, errno == EINTR { continue }
            guard size >= 0 else { throw WorkflowFileLeaseError.system(errno) }
            if size == 0 { break }
            count += size
            guard count <= Self.maximumBytes else { throw WorkflowFileLeaseError.unsafePath }
            hasher.update(data: Data(buffer.prefix(size)))
        }
        var after = stat(), named = stat()
        guard fstat(fd, &after) == 0, fstatat(directoryFD, "database.arrow", &named, AT_SYMLINK_NOFOLLOW) == 0,
              WorkflowEvaluationOwnership.identity(named) == fileIdentity, before.st_size == after.st_size, count == expectedBytes,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec, before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
              hasher.finalize().map({ String(format: "%02x", $0) }).joined() == expectedHash else { throw WorkflowFileLeaseError.unsafePath }
        try checkLocation(); try checkCancellation()
    }
    private func checkLocation() throws {
        let fd = try WorkflowEvaluationOwnership.openDirectory(root)
        defer { Darwin.close(fd) }
        guard try WorkflowEvaluationOwnership.directoryIdentity(fd) == rootIdentity else { throw WorkflowFileLeaseError.unsafePath }
        let named = try WorkflowEvaluationOwnership.openDirectory(destination.deletingLastPathComponent())
        defer { Darwin.close(named) }
        guard try WorkflowEvaluationOwnership.directoryIdentity(named) == directoryIdentity else { throw WorkflowFileLeaseError.unsafePath }
    }
    func finish(retain: Bool) {
        guard !closed else { return }; closed = true
        try? output.close()
        if !retain {
            var file = stat(), directory = stat()
            if fstatat(directoryFD, "database.arrow", &file, AT_SYMLINK_NOFOLLOW) == 0, WorkflowEvaluationOwnership.identity(file) == fileIdentity {
                _ = unlinkat(directoryFD, "database.arrow", 0)
            }
            if fstatat(namespaceFD, name, &directory, AT_SYMLINK_NOFOLLOW) == 0, WorkflowEvaluationOwnership.identity(directory) == directoryIdentity {
                _ = unlinkat(namespaceFD, name, AT_REMOVEDIR)
            }
        }
        Darwin.close(directoryFD); Darwin.close(namespaceFD); directoryFD = -1; namespaceFD = -1
        reservation?.close(); reservation = nil
    }
    deinit { finish(retain: false) }
}
