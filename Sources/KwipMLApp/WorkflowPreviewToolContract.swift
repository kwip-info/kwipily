import Foundation
import Darwin
import CryptoKit
import KwipMLCore

/// Preview inputs are an exact, bounded inventory beneath one owned operation.
enum WorkflowPreviewToolContract {
    static func fileLimit(path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") }), parts.first == "input" else { throw WorkflowFileLeaseError.unsafePath }
        switch path {
        case "input/graph.json", "input/weights/graph.json", "input/preparation/manifest.json": return 2_000_000
        case "input/weights/checkpoint.pt": return 100_000_000
        case "input/cache/activation-cache.pt": return 32_000_000
        case "input/cache/activation-cache.json": return 100_000
        default:
            guard parts.count == 3 else { throw WorkflowFileLeaseError.unsafePath }
            if parts[1] == "data" { return 10_000_000 }
            if parts[1] == "preparation" { return workload != nil && PreparationArtifactPolicy.isCapacityFileName(parts[2]) ? Int(workload!.limit(.retainedBytes)) : 50_000_000 }
            throw WorkflowFileLeaseError.unsafePath
        }
    }
    private static func options(_ arguments: [String]) throws -> [String: String] {
        let allowed: Set<String> = ["--graph", "--output", "--device", "--seed", "--preview", "--stop-at", "--resume-from", "--cache", "--checkpoint", "--dataset-csv", "--sample-limit", "--split-seed", "--validation-percent", "--prepared-dataflow", "--prepared-manifest-sha256", "--prepared-graph-digest", "--prepared-training-node", "--prepared-training-port", "--prepared-validation-node", "--prepared-validation-port", "--feature-columns-json", "--target-column"]
        guard arguments.count <= 64, arguments.allSatisfy({ !$0.isEmpty && !$0.contains("\0") && $0.utf8.count <= 4096 }) else { throw WorkflowFileLeaseError.unsafePath }
        var result: [String: String] = [:], index = 0
        while index < arguments.count {
            let key = arguments[index]
            guard allowed.contains(key), result[key] == nil else { throw WorkflowFileLeaseError.unsafePath }
            if key == "--preview" { result[key] = "true"; index += 1 }
            else {
                guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else { throw WorkflowFileLeaseError.unsafePath }
                result[key] = arguments[index + 1]; index += 2
            }
        }
        return result
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?, workload: ExecutionWorkloadPolicy? = nil) throws {
        let args = try options(arguments)
        guard let graph = args["--graph"], graph.hasPrefix("/"), graph.hasSuffix("/input/graph.json"),
              !graph.contains("//"), !graph.split(separator: "/").contains("."), !graph.split(separator: "/").contains("..") else { throw WorkflowFileLeaseError.unsafePath }
        let root = directory ?? URL(fileURLWithPath: graph).deletingLastPathComponent().deletingLastPathComponent()
        guard root.isFileURL, UUID(uuidString: root.lastPathComponent)?.uuidString == root.lastPathComponent,
              UUID(uuidString: root.deletingLastPathComponent().lastPathComponent)?.uuidString == root.deletingLastPathComponent().lastPathComponent,
              root.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations",
              graph == root.appendingPathComponent("input/graph.json").path,
              args["--output"] == root.appendingPathComponent("artifacts").path,
              ["auto", "cpu", "mps"].contains(args["--device"] ?? ""), let seed = args["--seed"].flatMap(UInt32.init), String(seed) == args["--seed"],
              (1...WorkflowPreviewCapture.Purpose.preview.captureCount(workload)).contains(files.count), Set(files.map(\.path)).count == files.count else { throw WorkflowFileLeaseError.unsafePath }
        var total = 0
        for file in files {
            guard file.bytes >= 0, file.bytes <= (try fileLimit(path: file.path, workload: workload)), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
            total += file.bytes
            guard total <= WorkflowPreviewCapture.Purpose.preview.captureBytes(workload) else { throw WorkflowFileLeaseError.unsafePath }
        }
        let actual = Set(files.map(\.path))
        var expected: Set<String> = ["input/graph.json"]
        if args["--preview"] != nil {
            guard args["--stop-at"] != nil, args["--resume-from"] == nil, args["--cache"] == nil else { throw WorkflowFileLeaseError.unsafePath }
        } else {
            guard args["--resume-from"] != nil, args["--stop-at"] == nil,
                  args["--cache"] == root.appendingPathComponent("input/cache/activation-cache.pt").path else { throw WorkflowFileLeaseError.unsafePath }
            expected.formUnion(["input/cache/activation-cache.pt", "input/cache/activation-cache.json"])
        }
        if let checkpoint = args["--checkpoint"] {
            guard checkpoint == root.appendingPathComponent("input/weights/checkpoint.pt").path else { throw WorkflowFileLeaseError.unsafePath }
            expected.formUnion(["input/weights/checkpoint.pt", "input/weights/graph.json"])
        }
        try WorkflowModelDataSelection.validate(args, files: files, root: root, expected: &expected, workload: workload)
        guard actual == expected else { throw WorkflowFileLeaseError.unsafePath }
    }
    /// Descriptor-based hashing keeps large checkpoints and prepared inventories out of RAM.
    static func verifyInputs(files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        for file in files { try verifyFile(file: file, directory: directory, limit: fileLimit(path: file.path, workload: workload)) }
    }
    static func verifyFile(file: WorkflowArchiveFile, directory: URL, limit: Int) throws {
        guard (0...9_007_199_254_740_991).contains(limit), (0...limit).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
        let components = try WorkflowArchiveCodec.pathParts(file.path)
        let parent = components.dropLast().reduce(directory) { $0.appendingPathComponent($1) }
        let parentFD = try WorkflowEvaluationOwnership.openDirectory(parent)
        defer { Darwin.close(parentFD) }
        _ = try WorkflowEvaluationOwnership.directoryIdentity(parentFD)
        let name = components.last!
        let fd = openat(parentFD, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard fd >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(fd) }
        var before = stat()
        guard fstat(fd, &before) == 0, before.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              before.st_nlink == 1, before.st_uid == geteuid(), before.st_mode & 0o022 == 0,
              before.st_size == file.bytes, (0...limit).contains(file.bytes) else { throw WorkflowFileLeaseError.unsafePath }
        var digest = SHA256(), total = 0, buffer = [UInt8](repeating: 0, count: 65_536)
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress!, min($0.count, limit + 1 - total)) }
            if count < 0 { if errno == EINTR { continue }; throw WorkflowFileLeaseError.system(errno) }
            if count == 0 { break }
            total += count
            guard total <= limit else { throw WorkflowFileLeaseError.unsafePath }
            digest.update(data: Data(buffer.prefix(count)))
        }
        var after = stat(), named = stat()
        guard total == file.bytes, fstat(fd, &after) == 0,
              fstatat(parentFD, name, &named, AT_SYMLINK_NOFOLLOW) == 0,
              WorkflowEvaluationOwnership.identity(before) == WorkflowEvaluationOwnership.identity(named),
              after.st_size == before.st_size, after.st_nlink == before.st_nlink,
              after.st_mtimespec.tv_sec == before.st_mtimespec.tv_sec, after.st_mtimespec.tv_nsec == before.st_mtimespec.tv_nsec,
              after.st_ctimespec.tv_sec == before.st_ctimespec.tv_sec, after.st_ctimespec.tv_nsec == before.st_ctimespec.tv_nsec,
              digest.finalize().map({ String(format: "%02x", $0) }).joined() == file.sha256 else { throw WorkflowFileLeaseError.unsafePath }
    }
    static func verifyPreparedInventory(arguments: [String], files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        let args = try options(arguments)
        guard args["--prepared-dataflow"] != nil else { return }
        let data = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input/preparation"), name: "manifest.json", limit: 2_000_000)
        guard WorkflowArchiveCodec.hash(data) == args["--prepared-manifest-sha256"] else { throw WorkflowFileLeaseError.unsafePath }
        try WorkflowCompletedPreparationInventory.verify(data: data, files: files, prefix: "input/preparation/",
            expectedGraphDigest: args["--prepared-graph-digest"], workload: workload)
        guard let training = args["--prepared-training-node"], let validation = args["--prepared-validation-node"] else { throw WorkflowFileLeaseError.unsafePath }
        _ = try WorkflowCompletedPreparationInventory.verifyCoverage(data: data, files: files, prefix: "input/preparation/", directory: directory, selectedNodes: [training, validation])
    }
}
