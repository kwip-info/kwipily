import Foundation
import Darwin
import CryptoKit
import KwipMLCore

/// Descriptor-relative access beneath an explicitly configured workspace. No artifact-supplied paths.
enum DiagnosticFileIO {
    static let maximumBytes=8_000_000
    private static func failure()->DiagnosticContractError {DiagnosticContractError("Diagnostic storage requires bounded regular files and real directories within this workspace.")}
    private static func openingFailure(_ code: Int32) -> DiagnosticContractError {
        if code == ENOENT {
            return DiagnosticContractError("A required evidence file or folder is missing. Restore the original from a backup, then retry verification. The experiment record is preserved.")
        }
        return failure()
    }
    private static func directory(_ root:URL,_ parts:[String],create:Bool,allowMissing:Bool=false)throws->Int32 {
        var fd=Darwin.open(root.standardizedFileURL.resolvingSymlinksInPath().path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd>=0 else {if allowMissing && errno == ENOENT {return -1};throw openingFailure(errno)}
        do {
            for part in parts {
                guard !part.isEmpty,part != ".",part != "..",!part.contains("/") else {throw failure()}
                if create {if mkdirat(fd,part,0o700) != 0 && errno != EEXIST {throw failure()}}
                let next=openat(fd,part,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
                guard next>=0 else {if allowMissing && errno == ENOENT {Darwin.close(fd);return -1};throw openingFailure(errno)}
                Darwin.close(fd);fd=next
            }
            return fd
        } catch {Darwin.close(fd);throw error}
    }
    /// Caller owns the descriptor. Every archive component is opened relative to a held directory.
    static func openRegularFile(root: URL, parts: [String], limit: Int) throws -> Int32 {
        guard limit >= 0, let name = parts.last, !name.isEmpty, name != ".", name != "..",
              !name.contains("/"), !parts.contains(where: { $0.contains("\0") }) else { throw failure() }
        let parent = try directory(root, Array(parts.dropLast()), create: false)
        defer { Darwin.close(parent) }
        let fd = openat(parent, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw openingFailure(errno) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              info.st_size >= 0, info.st_size <= limit else { Darwin.close(fd); throw failure() }
        return fd
    }
    static func read(root:URL,parts:[String],limit:Int=maximumBytes)throws->Data {
        guard let data=try readOptional(root:root,parts:parts,limit:limit,allowMissing:false) else {throw failure()}
        return data
    }
    /// Only ENOENT means absence. Unsafe links, unreadable directories and corrupt bytes remain errors.
    static func readIfPresent(root:URL,parts:[String],limit:Int=maximumBytes)throws->Data? {
        try readOptional(root:root,parts:parts,limit:limit,allowMissing:true)
    }
    /// Only for an explicitly selected external source, while the caller holds
    /// its security scope. A file grant need not authorize opening its parent.
    /// Archive paths must continue to use the descriptor-relative reader above.
    static func readSelectedFile(_ url:URL,limit:Int)throws->Data {
        guard url.isFileURL,!url.path.contains("\0"),(0...10_000_000).contains(limit) else {throw failure()}
        let fd=Darwin.open(url.path,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)
        guard fd>=0 else {throw openingFailure(errno)}
        defer {Darwin.close(fd)}
        return try readDescriptor(fd,limit:limit)
    }
    /// A selected file grant is held by the caller throughout this bounded snapshot.
    /// Destination paths belong to a newly owned private stage, never to the source.
    static func copySelectedFile(_ source: URL, root: URL, parts: [String], limit: Int,
                                 checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> WorkflowArchiveFile {
        guard source.isFileURL, !source.path.contains("\0") else {throw failure()}
        try checkCancellation()
        let input = Darwin.open(source.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard input >= 0 else { throw openingFailure(errno) }
        defer { Darwin.close(input) }
        return try copyRegularDescriptor(input, root: root, parts: parts, limit: limit, checkCancellation: checkCancellation)
    }
    /// Borrows an already descriptor-anchored source. Pread preserves caller offset and ownership.
    static func copyRegularDescriptor(_ input: Int32, root: URL, parts: [String], limit: Int,
                                      checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> WorkflowArchiveFile {
        guard (0...9_007_199_254_740_991).contains(limit),
              let leaf = parts.last, !leaf.isEmpty, leaf != ".", leaf != "..", !leaf.contains("/"),
              !parts.contains(where: { $0.contains("\0") }) else { throw failure() }
        try checkCancellation()
        var before = stat()
        guard fstat(input, &before) == 0, before.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              before.st_size >= 0, before.st_size <= limit else { throw failure() }
        let parent = try directory(root, Array(parts.dropLast()), create: true)
        defer { Darwin.close(parent) }
        let output = openat(parent, leaf, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard output >= 0 else { throw openingFailure(errno) }
        var published = false
        defer { Darwin.close(output); if !published { unlinkat(parent, leaf, 0) } }
        var digest = SHA256(), total = 0, buffer = [UInt8](repeating: 0, count: 65_536)
        while true {
            try checkCancellation()
            let count = Darwin.pread(input, &buffer, buffer.count, off_t(total))
            if count < 0, errno == EINTR { continue }
            guard count >= 0 else { throw failure() }
            if count == 0 { break }
            guard count <= limit - total else { throw failure() }
            let bytes = Data(buffer.prefix(count))
            try bytes.withUnsafeBytes { raw in
                var written = 0
                while written < raw.count {
                    let amount = Darwin.write(output, raw.baseAddress!.advanced(by: written), raw.count - written)
                    if amount < 0, errno == EINTR { continue }
                    guard amount > 0 else { throw failure() }
                    written += amount
                }
            }
            digest.update(data: bytes); total += count
        }
        try checkCancellation()
        var after = stat()
        guard fstat(input, &after) == 0, total == before.st_size, before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec, before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec,
              fsync(output) == 0 else { throw failure() }
        published = true
        return .init(path: parts.joined(separator: "/"), bytes: total,
                     sha256: digest.finalize().map { String(format: "%02x", $0) }.joined())
    }
    private static func readOptional(root:URL,parts:[String],limit:Int,allowMissing:Bool)throws->Data? {
        guard let name=parts.last,!name.contains("/"),name != ".",name != "..",!name.isEmpty else {throw failure()}
        let parent=try directory(root,Array(parts.dropLast()),create:false,allowMissing:allowMissing)
        guard parent>=0 else {return nil};defer {Darwin.close(parent)}
        let fd=openat(parent,name,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)
        guard fd>=0 else {if allowMissing && errno == ENOENT {return nil};throw openingFailure(errno)};defer {Darwin.close(fd)}
        return try readDescriptor(fd,limit:limit)
    }
    private static func readDescriptor(_ fd:Int32,limit:Int)throws->Data {
        guard limit>=0,limit<Int.max else {throw failure()}
        var metadata=stat()
        guard fstat(fd,&metadata)==0,metadata.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),metadata.st_size>=0,metadata.st_size<=limit else {throw failure()}
        var result=Data(),buffer=[UInt8](repeating:0,count:65536)
        while result.count<=limit {
            let count=buffer.withUnsafeMutableBytes {Darwin.read(fd,$0.baseAddress!,min($0.count,limit+1-result.count))}
            if count<0 {if errno==EINTR {continue};throw failure()}
            if count==0 {break}
            result.append(contentsOf:buffer.prefix(count))
        }
        var after=stat()
        guard result.count<=limit,result.count==metadata.st_size,fstat(fd,&after)==0,
              after.st_size==metadata.st_size,
              after.st_mtimespec.tv_sec==metadata.st_mtimespec.tv_sec,after.st_mtimespec.tv_nsec==metadata.st_mtimespec.tv_nsec,
              after.st_ctimespec.tv_sec==metadata.st_ctimespec.tv_sec,after.st_ctimespec.tv_nsec==metadata.st_ctimespec.tv_nsec else {throw failure()}
        return result
    }
    static func createExclusiveDirectory(root:URL,name:String)throws {
        guard !name.isEmpty,name != ".",name != "..",!name.contains("/"),!name.contains("\0") else {throw failure()}
        let parent=try directory(root,[],create:false);defer {Darwin.close(parent)}
        guard mkdirat(parent,name,0o700)==0 else {throw openingFailure(errno)}
    }
    /// Serializes a metadata transaction across independent project windows/processes.
    static func withExclusiveLock<T>(root:URL,parts:[String],body:()throws->T)throws->T {
        guard let name=parts.last,!name.isEmpty,!name.contains("/"),name != ".",name != ".." else {throw failure()}
        let parent=try directory(root,Array(parts.dropLast()),create:true);defer {Darwin.close(parent)}
        let fd=openat(parent,name,O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK,0o600)
        guard fd>=0 else {throw failure()};defer {Darwin.close(fd)}
        var metadata=stat()
        guard fstat(fd,&metadata)==0,metadata.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),flock(fd,LOCK_EX|LOCK_NB)==0 else {throw failure()}
        defer {_ = flock(fd,LOCK_UN)}
        return try body()
    }
    static func writeImmutable(_ data:Data,root:URL,parts:[String],limit:Int=maximumBytes)throws {
        guard (0...10_000_000).contains(limit) else {throw failure()}
        try write(data,root:root,parts:parts,replace:false,limit:limit)
    }
    /// Only manifest-owned File Input members may use the separate 64 MiB
    /// payload allowance. The caller must own the collection root and inventory.
    static func writeCapturedFileInputMember(_ data: Data, root: URL, parts: [String]) throws {
        guard parts.count >= 2, parts[parts.count - 2] == "members",
              let name = parts.last, !name.isEmpty, name.utf8.count <= 255,
              !name.contains("\0"), !name.contains("/"), !name.contains("\\"), name != ".", name != ".." else { throw failure() }
        try write(data, root: root, parts: parts, replace: false, limit: FileInputCollectionManifest.maximumMemberBytes)
    }
    /// Lifecycle files have larger explicit budgets than ordinary small metadata.
    /// The caller owns a newly created continuation-parent directory.
    static func writeCheckpointArtifact(_ data:Data,root:URL,name:String)throws {
        var limits=WorkflowCheckpointSnapshot.artifactLimits
        limits[WorkflowCheckpointInventory.traceName]=WorkflowCheckpointInventory.traceLimit
        guard let limit=limits[name] else {throw failure()}
        try write(data,root:root,parts:["continuation-parent",name],replace:false,limit:limit)
    }
    /// Only the frozen evaluation snapshot's explicit file allowlist may use its larger bounds.
    static func writeFrozenEvaluationArtifact(_ data:Data,root:URL,path:String)throws {
        try write(data,root:root,parts:WorkflowArchiveCodec.pathParts(path),replace:false,limit:WorkflowFrozenModelSnapshot.fileLimit(path))
    }
    static func writeAtomic(_ data:Data,root:URL,parts:[String])throws {
        try write(data,root:root,parts:parts,replace:true)
    }
    private static func write(_ data:Data,root:URL,parts:[String],replace:Bool,limit:Int=maximumBytes)throws {
        guard data.count<=limit,let name=parts.last,!name.isEmpty,!name.contains("/"),name != ".",name != ".." else {throw failure()}
        let parent=try directory(root,Array(parts.dropLast()),create:true);defer {Darwin.close(parent)}
        func checkReplacement()throws {
            var info=stat()
            if fstatat(parent,name,&info,AT_SYMLINK_NOFOLLOW)==0 {
                guard info.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG) else {throw failure()}
            } else if errno != ENOENT {throw openingFailure(errno)}
        }
        if replace {try checkReplacement()}
        let temporary=".pending-"+UUID().uuidString
        let fd=openat(parent,temporary,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)
        guard fd>=0 else {throw openingFailure(errno)}
        defer {Darwin.close(fd);unlinkat(parent,temporary,0)}
        try data.withUnsafeBytes {bytes in
            var offset=0
            while offset<bytes.count {
                let written=Darwin.write(fd,bytes.baseAddress!.advanced(by:offset),bytes.count-offset)
                if written<0 {if errno==EINTR {continue};throw failure()}
                guard written>0 else {throw failure()};offset+=written
            }
        }
        guard fsync(fd)==0 else {throw failure()}
        if replace {
            try checkReplacement()
            guard renameat(parent,temporary,parent,name)==0 else {throw failure()}
        } else {
            guard linkat(parent,temporary,parent,name,0)==0 else {throw failure()}
        }
        // Commit is complete; a directory metadata flush must not report an uncommitted save.
        _ = fsync(parent)
    }
}
