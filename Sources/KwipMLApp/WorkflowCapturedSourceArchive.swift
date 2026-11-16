import Foundation
import Darwin
import KwipMLCore

/// A captured source is either one historical table file or one exact File Input
/// manifest/member closure. Members are never reopened from their original paths.
enum WorkflowCapturedSourceArchive {
    static let maximumAggregateBytes = FileInputCollectionManifest.maximumTotalBytes + FileInputCollectionManifest.maximumManifestBytes

    static func aggregateLimit(hasCollections: Bool) -> Int { hasCollections ? maximumAggregateBytes : 256_000_000 }

    static func declaredByteCount(_ source: WorkflowCapturedSource) throws -> Int {
        guard (0...(source.isFileCollection ? FileInputCollectionManifest.maximumManifestBytes : 10_000_000)).contains(source.byteCount),
              source.collectionMembers.count <= 1000, source.isFileCollection || source.collectionMembers.isEmpty else { throw failure() }
        var count = source.byteCount
        for member in source.collectionMembers {
            guard (0...FileInputCollectionManifest.maximumMemberBytes).contains(member.byteCount),
                  member.byteCount <= maximumAggregateBytes - count else { throw failure() }
            count += member.byteCount
        }
        return count
    }

    static func validate(_ source: WorkflowArchiveSource, prefix: String) throws -> Set<String> {
        let parts = try WorkflowArchiveCodec.pathParts(source.path)
        let relative = String(source.path.dropFirst(prefix.count))
        guard source.path.hasPrefix(prefix), WorkflowArchiveCodec.isHash(source.sha256), source.bytes >= 0 else { throw failure() }
        if let members = source.collectionMembers {
            let relativeParts = try WorkflowArchiveCodec.pathParts(relative)
            guard relativeParts.count == 3, relativeParts[2] == "manifest.json",
                  relativeParts[1] == source.sha256,
                  (0..<128).contains(where: { relativeParts[0] == "source-\($0).collection" }),
                  source.bytes <= FileInputCollectionManifest.maximumManifestBytes, members.count <= 1000 else { throw failure() }
            let folder = parts.dropLast().joined(separator: "/")
            var paths: Set<String> = [source.path], total = 0
            for member in members {
                let memberParts = try WorkflowArchiveCodec.pathParts(member.path)
                guard memberParts.count == parts.count + 1, memberParts.dropLast(2).joined(separator: "/") == folder,
                      memberParts[memberParts.count - 2] == "members", memberParts.last!.utf8.count <= 255,
                      (0...FileInputCollectionManifest.maximumMemberBytes).contains(member.bytes), WorkflowArchiveCodec.isHash(member.sha256),
                      paths.insert(member.path).inserted, member.bytes <= FileInputCollectionManifest.maximumTotalBytes - total else { throw failure() }
                total += member.bytes
            }
            return paths
        }
        guard parts.count == (prefix.isEmpty ? 1 : 2),
              (0..<128).contains(where: { WorkflowSourceFilename.validArchive(relative, index: $0) }),
              source.bytes <= 10_000_000 else { throw failure() }
        return [source.path]
    }

    static func publish(_ source: WorkflowCapturedSource, address: WorkflowNodeAddress, index: Int,
                        root: URL, prefix: String) throws -> WorkflowArchiveSource {
        try Task.checkCancellation()
        guard (0..<128).contains(index), (0...10_000_000).contains(source.byteCount),
              source.isFileCollection || source.collectionMembers.isEmpty else { throw failure() }
        let bytes = try DiagnosticFileIO.read(root: source.url.deletingLastPathComponent(), parts: [source.url.lastPathComponent], limit: source.isFileCollection ? FileInputCollectionManifest.maximumManifestBytes : 10_000_000)
        guard bytes.count == source.byteCount, WorkflowArchiveCodec.hash(bytes) == source.sha256 else { throw failure() }
        let path: String
        if source.isFileCollection { path = prefix + "source-\(index).collection/" + source.sha256 + "/manifest.json" }
        else { path = prefix + WorkflowSourceFilename.archive(index: index, source: source.url) }
        var members: [WorkflowArchiveFile]? = nil
        if source.isFileCollection {
            let manifest = try FileInputCollectionManifest.decode(bytes)
            guard manifest.complete else { throw failure() }
            let expected = manifest.members.filter { $0.status == "content" }
            guard source.collectionMembers.count == expected.count,
                  Set(source.collectionMembers.map(\.capturePath)).count == expected.count else { throw failure() }
            let captured = Dictionary(uniqueKeysWithValues: source.collectionMembers.map { ($0.capturePath, $0) })
            let folder = String(path.dropLast("manifest.json".count))
            var files: [WorkflowArchiveFile] = []
            for member in expected {
                try Task.checkCancellation()
                guard let name = member.capturePath, let hash = member.sha256, let capture = captured[name],
                      capture.byteCount == Int(member.bytes), capture.sha256 == hash,
                      capture.url.standardizedFileURL == source.url.deletingLastPathComponent().appendingPathComponent(name).standardizedFileURL else { throw failure() }
                let data = try DiagnosticFileIO.read(root: source.url.deletingLastPathComponent(), parts: try WorkflowArchiveCodec.pathParts(name), limit: FileInputCollectionManifest.maximumMemberBytes)
                guard data.count == capture.byteCount, WorkflowArchiveCodec.hash(data) == hash else { throw failure() }
                let destination = folder + name
                try DiagnosticFileIO.writeCapturedFileInputMember(data, root: root, parts: try WorkflowArchiveCodec.pathParts(destination))
                files.append(.init(path: destination, bytes: data.count, sha256: hash))
            }
            members = files
        }
        try DiagnosticFileIO.writeImmutable(bytes, root: root, parts: try WorkflowArchiveCodec.pathParts(path), limit: source.isFileCollection ? FileInputCollectionManifest.maximumManifestBytes : 10_000_000)
        let result = WorkflowArchiveSource(address: address, path: path, bytes: bytes.count, sha256: source.sha256, collectionMembers: members)
        _ = try validate(result, prefix: prefix)
        _ = try open(result, root: root, prefix: prefix)
        return result
    }

    static func open(_ source: WorkflowArchiveSource, root: URL, prefix: String, baseParts: [String] = []) throws -> WorkflowCapturedSource {
        let paths = try validate(source, prefix: prefix)
        func read(_ file: WorkflowArchiveFile, limit: Int) throws -> Data {
            try Task.checkCancellation()
            let data = try DiagnosticFileIO.read(root: root, parts: baseParts + WorkflowArchiveCodec.pathParts(file.path), limit: limit)
            guard data.count == file.bytes, WorkflowArchiveCodec.hash(data) == file.sha256 else { throw failure() }
            return data
        }
        let bytes = try read(.init(path: source.path, bytes: source.bytes, sha256: source.sha256), limit: source.collectionMembers == nil ? 10_000_000 : FileInputCollectionManifest.maximumManifestBytes)
        let sourceURL = (baseParts + (try WorkflowArchiveCodec.pathParts(source.path))).reduce(root) { $0.appendingPathComponent($1) }
        guard let files = source.collectionMembers else { return .init(url: sourceURL, sha256: source.sha256, byteCount: source.bytes) }
        let manifest = try FileInputCollectionManifest.decode(bytes)
        guard manifest.complete else { throw failure() }
        let folder = String(source.path.dropLast("manifest.json".count))
        let expected = manifest.members.filter { $0.status == "content" }
        guard expected.count == files.count else { throw failure() }
        let inventory = Dictionary(uniqueKeysWithValues: files.map { ($0.path, $0) })
        var captures: [FileInputCapturedMember] = []
        for member in expected {
            guard let path = member.capturePath, let hash = member.sha256,
                  let file = inventory[folder + path], file.sha256 == hash, file.bytes == Int(member.bytes) else { throw failure() }
            _ = try read(file, limit: FileInputCollectionManifest.maximumMemberBytes)
            captures.append(.init(capturePath: path, url: sourceURL.deletingLastPathComponent().appendingPathComponent(path), sha256: hash, byteCount: file.bytes))
        }
        let folderParts = baseParts + (try WorkflowArchiveCodec.pathParts(source.path)).dropLast()
        let localExpected = Set(paths.map { String($0.dropFirst(folder.count)) })
        try verifyExactCollectionDirectory(root: root, parts: Array(folderParts), expected: localExpected)
        return .init(url: sourceURL, sha256: source.sha256, byteCount: source.bytes, collectionMembers: captures, isFileCollection: true)
    }

    private static func verifyExactCollectionDirectory(root: URL, parts: [String], expected: Set<String>) throws {
        var fd = Darwin.open(root.standardizedFileURL.resolvingSymlinksInPath().path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw failure() }
        defer { Darwin.close(fd) }
        for part in parts {
            let next = openat(fd, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard next >= 0 else { throw failure() }; Darwin.close(fd); fd = next
        }
        func names(_ directory: Int32) throws -> [String] {
            let copy = dup(directory)
            guard copy >= 0, let stream = fdopendir(copy) else { if copy >= 0 { Darwin.close(copy) }; throw failure() }
            defer { closedir(stream) }
            var result: [String] = []
            while let entry = readdir(stream) {
                try Task.checkCancellation()
                let name = withUnsafePointer(to: &entry.pointee.d_name) { $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) } }
                if name == "." || name == ".." { continue }
                guard result.count < 1002 else { throw failure() }; result.append(name)
            }
            return result
        }
        var actual: Set<String> = []
        for name in try names(fd) {
            var info = stat(); guard fstatat(fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0 else { throw failure() }
            if name == "members", info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR) {
                let child = openat(fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard child >= 0 else { throw failure() }; defer { Darwin.close(child) }
                for member in try names(child) {
                    var metadata = stat()
                    guard fstatat(child, member, &metadata, AT_SYMLINK_NOFOLLOW) == 0, metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) else { throw failure() }
                    actual.insert("members/" + member)
                }
            } else {
                guard name == "manifest.json", info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) else { throw failure() }
                actual.insert(name)
            }
        }
        guard actual == expected else { throw failure() }
    }
    private static func failure() -> WorkflowArchiveError { .init("Captured File Input inventory, member bytes or collection identity changed.") }
}
