import Foundation
import Darwin

/// Keeps every save write in the originally opened directory even if its pathname is replaced.
internal enum FlowDocumentSaveTransaction {
    static func save(_ document: FlowDocument, at url: URL, expected: FlowDocumentRevision,
                     beforeCommit: (Int32) throws -> Void) throws {
        guard let canonical = realpath(url.deletingLastPathComponent().path, nil) else { throw FlowDocumentError.unsafeReference }
        let parent = URL(fileURLWithPath: String(cString: canonical)); free(canonical)
        let directory = open(parent.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_CLOEXEC)
        guard directory >= 0 else { throw FlowDocumentError.unsafeReference }
        defer { close(directory) }
        var identity = stat()
        guard fstat(directory, &identity) == 0 else { throw posixError() }
        func verifyParent() throws {
            let current = open(parent.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_CLOEXEC)
            guard current >= 0 else { throw FlowDocumentError.unsafeReference }
            defer { close(current) }
            var value = stat()
            guard fstat(current, &value) == 0, value.st_dev == identity.st_dev,
                  value.st_ino == identity.st_ino else { throw FlowDocumentError.unsafeReference }
        }
        let name = url.lastPathComponent
        let lock = openat(directory, "." + name + ".lock", O_CREAT | O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, mode_t(0o600))
        guard lock >= 0 else { throw posixError() }
        defer { close(lock) }
        var lockInfo = stat()
        guard fstat(lock, &lockInfo) == 0, lockInfo.st_mode & S_IFMT == S_IFREG else { throw FlowDocumentError.unsafeReference }
        guard flock(lock, LOCK_EX | LOCK_NB) == 0 else { throw posixError() }
        defer { flock(lock, LOCK_UN) }
        try verifyParent()
        let current = try read(directory, name)
        guard try FlowDocumentCodec.reference(current) == expected,
              document.id == current.id, current.revision < UInt64.max,
              document.revision == current.revision + 1 else { throw FlowDocumentError.revisionMismatch }
        let bytes = try FlowDocumentCodec.encode(document)
        try beforeCommit(directory)
        try verifyParent()
        guard try FlowDocumentCodec.reference(read(directory, name)) == expected else { throw FlowDocumentError.revisionMismatch }
        let revisions = try folder(directory, ".revisions")
        defer { close(revisions) }
        let archiveDirectory = try folder(revisions, expected.documentID.uuidString)
        defer { close(archiveDirectory) }
        let archiveName = expected.sha256 + ".kwipflow"
        let archive = openat(archiveDirectory, archiveName, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        if archive >= 0 {
            defer { close(archive) }
            try write(FlowDocumentCodec.encode(current), to: archive)
            guard fsync(archiveDirectory) == 0 else { throw posixError() }
        } else {
            guard errno == EEXIST else { throw posixError() }
            guard try FlowDocumentCodec.reference(read(archiveDirectory, archiveName)) == expected else { throw FlowDocumentError.revisionMismatch }
        }
        try verifyParent()
        let temporary = ".kwipflow-save-" + UUID().uuidString
        let output = openat(directory, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard output >= 0 else { throw posixError() }
        defer { close(output); unlinkat(directory, temporary, 0) }
        try write(bytes, to: output)
        try verifyParent()
        guard renameat(directory, temporary, directory, name) == 0, fsync(directory) == 0 else { throw posixError() }
        try verifyParent()
    }

    private static func folder(_ parent: Int32, _ name: String) throws -> Int32 {
        if mkdirat(parent, name, mode_t(0o700)) != 0, errno != EEXIST { throw posixError() }
        let result = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard result >= 0 else { throw FlowDocumentError.unsafeReference }
        guard fsync(parent) == 0 else { close(result); throw posixError() }
        return result
    }
    private static func read(_ parent: Int32, _ name: String) throws -> FlowDocument {
        let input = openat(parent, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard input >= 0 else { throw FlowDocumentError.unsafeReference }
        defer { close(input) }
        var info = stat()
        guard fstat(input, &info) == 0, info.st_mode & S_IFMT == S_IFREG else { throw FlowDocumentError.unsafeReference }
        guard info.st_size >= 0, info.st_size <= FlowDocumentCodec.maximumBytes else { throw FlowDocumentError.resourceLimit }
        var bytes = Data(); var buffer = [UInt8](repeating: 0, count: 65536)
        while true {
            let count = Darwin.read(input, &buffer, buffer.count)
            if count < 0, errno == EINTR { continue }
            guard count >= 0 else { throw posixError() }
            if count == 0 { break }
            guard bytes.count + count <= FlowDocumentCodec.maximumBytes else { throw FlowDocumentError.resourceLimit }
            bytes.append(contentsOf: buffer.prefix(count))
        }
        return try FlowDocumentCodec.decode(bytes)
    }
    private static func write(_ bytes: Data, to descriptor: Int32) throws {
        try bytes.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let count = Darwin.write(descriptor, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                if count < 0, errno == EINTR { continue }
                guard count > 0 else { throw posixError() }
                offset += count
            }
        }
        guard fsync(descriptor) == 0 else { throw posixError() }
    }
    private static func posixError() -> NSError { NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
}
