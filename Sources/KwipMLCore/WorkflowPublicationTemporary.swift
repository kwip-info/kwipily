import Foundation
import Darwin

public struct WorkflowPublicationCleanupReport: Equatable, Sendable {
    public var removedTemporaryFiles = 0
    public var removedReceipts = 0
    public var retainedReceipts: [String] = []
}

/// The same publication scratch object is used by normal metadata writes and by
/// crash fixtures. A receipt proves only this file, never arbitrary staging.
final class WorkflowPublicationTemporary {
    struct Receipt: Codable {
        let schemaVersion: Int
        let nonce: String
        let target: String
        let maximumBytes: Int
        let parent: WorkflowFileIdentity
        let owner: WorkflowFileIdentity
        let temporary: WorkflowFileIdentity
        let receipt: WorkflowFileIdentity
    }
    let name: String
    private let receiptName: String
    private let parent: Int32
    private let file: Int32
    private let proof: Receipt
    private let check: () throws -> Void

    init(parent: Int32, parentIdentity: WorkflowFileIdentity, ownerIdentity: WorkflowFileIdentity,
         target: String, maximumBytes: Int, check: @escaping () throws -> Void) throws {
        guard let limit = Self.targetLimit(target), (0...limit).contains(maximumBytes) else { throw WorkflowFileLeaseError.unsafePath }
        try check()
        let directory = dup(parent)
        guard directory >= 0 else { throw WorkflowFileLeaseError.system(errno) }
        guard fcntl(directory, F_SETFD, FD_CLOEXEC) == 0 else { Darwin.close(directory); throw WorkflowFileLeaseError.system(errno) }
        let nonce = UUID().uuidString
        let name = ".ownership-\(nonce).tmp", receiptName = ".ownership-\(nonce).receipt.json"
        var file: Int32 = -1, receipt: Int32 = -1
        do {
            file = openat(directory, name, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard file >= 0 else { throw WorkflowFileLeaseError.system(errno) }
            let tempInfo = try Self.regular(file)
            guard fsync(file) == 0 else { throw WorkflowFileLeaseError.system(errno) }
            receipt = openat(directory, receiptName, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard receipt >= 0 else { throw WorkflowFileLeaseError.system(errno) }
            let receiptInfo = try Self.regular(receipt)
            let proof = Receipt(schemaVersion: 1, nonce: nonce, target: target, maximumBytes: maximumBytes,
                parent: parentIdentity, owner: ownerIdentity, temporary: Self.identity(tempInfo), receipt: Self.identity(receiptInfo))
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            try Self.write(encoder.encode(proof), to: receipt)
            guard fsync(receipt) == 0, fsync(directory) == 0 else { throw WorkflowFileLeaseError.system(errno) }
            try check()
            Darwin.close(receipt)
            self.parent = directory; self.file = file; self.name = name; self.receiptName = receiptName
            self.proof = proof; self.check = check
        } catch {
            // An incomplete receipt proves nothing. Keep both entries for manual
            // inspection instead of guessing whether partial state is disposable.
            if receipt >= 0 { Darwin.close(receipt) }; if file >= 0 { Darwin.close(file) }; Darwin.close(directory)
            throw error
        }
    }
    func write(_ bytes: Data) throws {
        try check()
        guard bytes.count <= proof.maximumBytes, Self.identity(try Self.regular(file)) == proof.temporary else { throw WorkflowFileLeaseError.unsafePath }
        try Self.write(bytes, to: file)
        guard fsync(file) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        try check()
    }
    func discard() throws {
        _ = try Self.remove(receiptName: receiptName, parent: parent, parentIdentity: proof.parent, ownerIdentity: proof.owner, check: check)
    }
    deinit { try? discard(); Darwin.close(file); Darwin.close(parent) }

    static func cleanup(parent: Int32, parentIdentity: WorkflowFileIdentity, ownerIdentity: WorkflowFileIdentity,
                        check: () throws -> Void) throws -> WorkflowPublicationCleanupReport {
        try check()
        let duplicate = dup(parent)
        guard duplicate >= 0 else { throw WorkflowFileLeaseError.system(errno) }
        guard fcntl(duplicate, F_SETFD, FD_CLOEXEC) == 0 else { Darwin.close(duplicate); throw WorkflowFileLeaseError.system(errno) }
        guard let directory = fdopendir(duplicate) else { Darwin.close(duplicate); throw WorkflowFileLeaseError.system(errno) }
        defer { closedir(directory) }
        rewinddir(directory)
        var names: [String] = [], count = 0
        while let entry = readdir(directory) {
            count += 1; guard count <= 4096 else { throw WorkflowFileLeaseError.unsafePath }
            let name = withUnsafeBytes(of: entry.pointee.d_name) { String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self) }
            if nonce(name) != nil { names.append(name) }
        }
        guard names.count <= 256 else { throw WorkflowFileLeaseError.unsafePath }
        var report = WorkflowPublicationCleanupReport()
        for name in names.sorted() {
            try Task.checkCancellation(); try check()
            do {
                let removed = try remove(receiptName: name, parent: parent, parentIdentity: parentIdentity, ownerIdentity: ownerIdentity, check: check)
                report.removedTemporaryFiles += removed ? 1 : 0; report.removedReceipts += 1
            } catch is CancellationError { throw CancellationError() }
            catch { report.retainedReceipts.append(name) }
            try check()
        }
        return report
    }
    private static func remove(receiptName: String, parent: Int32, parentIdentity: WorkflowFileIdentity,
                               ownerIdentity: WorkflowFileIdentity, check: () throws -> Void) throws -> Bool {
        try check()
        guard let token = nonce(receiptName) else { throw WorkflowFileLeaseError.unsafePath }
        let receiptFD = openat(parent, receiptName, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard receiptFD >= 0 else { throw WorkflowFileLeaseError.system(errno) }
        defer { Darwin.close(receiptFD) }
        let receiptInfo = try regular(receiptFD)
        guard receiptInfo.st_size > 0, receiptInfo.st_size <= 4096 else { throw WorkflowFileLeaseError.unsafePath }
        var bytes = [UInt8](repeating: 0, count: 4097), offset = 0
        while offset < bytes.count {
            let count = bytes.withUnsafeMutableBytes { Darwin.read(receiptFD, $0.baseAddress!.advanced(by: offset), $0.count - offset) }
            if count < 0 { if errno == EINTR { continue }; throw WorkflowFileLeaseError.system(errno) }
            if count == 0 { break }; offset += count
        }
        guard offset == receiptInfo.st_size else { throw WorkflowFileLeaseError.unsafePath }
        let raw = Data(bytes.prefix(offset))
        guard let object = try JSONSerialization.jsonObject(with: raw) as? [String: Any], Set(object.keys) == ["schemaVersion", "nonce", "target", "maximumBytes", "parent", "owner", "temporary", "receipt"] else { throw WorkflowFileLeaseError.unsafePath }
        let proof = try JSONDecoder().decode(Receipt.self, from: raw)
        guard proof.schemaVersion == 1, proof.nonce == token, proof.parent == parentIdentity, proof.owner == ownerIdentity,
              proof.receipt == identity(receiptInfo), let limit = targetLimit(proof.target), (0...limit).contains(proof.maximumBytes) else { throw WorkflowFileLeaseError.unsafePath }
        try match(parent: parent, name: receiptName, identity: proof.receipt, maximumLinks: 1)
        let tempName = ".ownership-\(token).tmp"
        var temp = stat(), removed = false
        if fstatat(parent, tempName, &temp, AT_SYMLINK_NOFOLLOW) == 0 {
            guard validRegular(temp, maximumLinks: 2), identity(temp) == proof.temporary,
                  temp.st_size >= 0, temp.st_size <= proof.maximumBytes else { throw WorkflowFileLeaseError.unsafePath }
            if temp.st_nlink == 2 {
                // A crash between link publication and scratch unlink may leave
                // two names. Only the recorded canonical target may be the other.
                try match(parent: parent, name: proof.target, identity: proof.temporary, maximumLinks: 2)
            }
            try check(); try match(parent: parent, name: tempName, identity: proof.temporary, maximumLinks: 2)
            guard unlinkat(parent, tempName, 0) == 0 else { throw WorkflowFileLeaseError.system(errno) }
            removed = true
        } else if errno != ENOENT { throw WorkflowFileLeaseError.system(errno) }
        try check(); try match(parent: parent, name: receiptName, identity: proof.receipt, maximumLinks: 1)
        guard unlinkat(parent, receiptName, 0) == 0, fsync(parent) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        return removed
    }
    private static func nonce(_ name: String) -> String? {
        let prefix = ".ownership-", suffix = ".receipt.json"
        guard name.hasPrefix(prefix), name.hasSuffix(suffix) else { return nil }
        let token = String(name.dropFirst(prefix.count).dropLast(suffix.count))
        return UUID(uuidString: token)?.uuidString == token ? token : nil
    }
    private static func targetLimit(_ name: String) -> Int? {
        if name == "ownership.json" { return 16_384 }
        if name == "group.json" { return 256_000 }
        if name == "authoring.json" { return 2_000_000 }
        for prefix in ["launch-", "claim-"] where name.hasPrefix(prefix) && name.hasSuffix(".json") {
            let token = String(name.dropFirst(prefix.count).dropLast(5))
            if UUID(uuidString: token)?.uuidString == token { return prefix == "claim-" ? 4096 : 256_000 }
        }
        for phase in ["preparation", "training"] where name.hasPrefix("phase-") && name.hasSuffix("-\(phase).json") {
            let token = String(name.dropFirst(6).dropLast(phase.count + 6))
            if UUID(uuidString: token)?.uuidString == token { return 256_000 }
        }
        return nil
    }
    private static func identity(_ value: stat) -> WorkflowFileIdentity { .init(device: UInt64(UInt32(bitPattern: value.st_dev)), inode: UInt64(value.st_ino)) }
    private static func validRegular(_ value: stat, maximumLinks: Int) -> Bool {
        value.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) && value.st_uid == geteuid() && value.st_mode & 0o022 == 0 && value.st_nlink >= 1 && value.st_nlink <= maximumLinks
    }
    private static func regular(_ fd: Int32) throws -> stat {
        var value = stat()
        guard fstat(fd, &value) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        guard validRegular(value, maximumLinks: 1) else { throw WorkflowFileLeaseError.unsafePath }; return value
    }
    private static func match(parent: Int32, name: String, identity expected: WorkflowFileIdentity, maximumLinks: Int) throws {
        var value = stat()
        guard fstatat(parent, name, &value, AT_SYMLINK_NOFOLLOW) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        guard validRegular(value, maximumLinks: maximumLinks), identity(value) == expected else { throw WorkflowFileLeaseError.unsafePath }
    }
    private static func write(_ data: Data, to fd: Int32) throws {
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                if count < 0 { if errno == EINTR { continue }; throw WorkflowFileLeaseError.system(errno) }
                guard count > 0 else { throw WorkflowFileLeaseError.system(EIO) }; offset += count
            }
        }
    }
}
