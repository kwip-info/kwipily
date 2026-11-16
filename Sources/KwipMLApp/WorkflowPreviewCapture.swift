import Foundation
import Darwin
import CryptoKit
import KwipMLCore

/// Incremental private capture. Invoke from a detached staging task, never from canvas rendering.
/// Ownership is transferred into an operation only after the complete inventory is validated.
struct WorkflowPreviewCapture {
    enum Purpose {
        case selfPlay, gameSession, backendFit, preview, preparation, sourceInspection, preparationInspection, modelTraining, tableViewQuery
        var fileCount: Int {
            switch self { case .selfPlay:return 600; case .gameSession:return 3; case .tableViewQuery: return 2; case .backendFit: return 302; case .preview: return 306; case .preparation: return 1536; case .sourceInspection: return 9; case .preparationInspection: return 301; case .modelTraining: return 307 }
        }
        var totalBytes: Int {
            switch self { case .selfPlay:return 1_000_000_000; case .gameSession:return 202_000_000; case .tableViewQuery: return 50_256_000; case .backendFit: return 202_100_000; case .preview: return 340_000_000; case .preparation: return 3_600_000_000; case .sourceInspection: return 270_435_456; case .preparationInspection: return 202_000_000; case .modelTraining: return 330_000_000 }
        }
        var allowsDecisionModels:Bool {if case .preparation=self{return true};return false}
        var namespace: String {
            switch self { case .selfPlay:return "model-training-staging"; case .gameSession:return "model-training-staging"; case .tableViewQuery: return "preparation-inspection-staging"; case .backendFit: return "model-training-staging"; case .preview: return "preview-staging"; case .preparation: return "preparation-staging"; case .sourceInspection: return "source-inspection-staging"; case .preparationInspection: return "preparation-inspection-staging"; case .modelTraining: return "model-training-staging" }
        }
        var usesPreparationWorkload: Bool {
            switch self {case .preparation,.preparationInspection,.tableViewQuery,.backendFit,.preview,.modelTraining:return true;default:return false}
        }
        func captureCount(_ workload: ExecutionWorkloadPolicy?) -> Int {
            guard let workload, usesPreparationWorkload else { return fileCount }
            let inventory = PreparationArtifactPolicy.inventoryCount(workload)
            switch self {
            case .preparation: return min(62_500, inventory + Int(min(62_500, workload.limit(.collectionMembers))) + 128)
            case .backendFit: return inventory + 2
            case .preview: return inventory + 6
            case .modelTraining: return inventory + 7
            case .preparationInspection: return inventory + 1
            default: return inventory + 2
            }
        }
        func captureBytes(_ workload: ExecutionWorkloadPolicy?) -> Int {
            guard let workload, usesPreparationWorkload else { return totalBytes }
            let retained = Int(workload.limit(.retainedBytes))
            switch self {
            case .tableViewQuery:return retained+256_000
            case .preparationInspection:return retained+2_000_000
            case .backendFit:return retained+2_100_000
            case .preview:return retained+140_000_000
            case .modelTraining:return retained+130_000_000
            case .preparation:return retained+3_400_000_000
            default:return totalBytes
            }
        }
        func fileLimit(_ path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
            if self == .tableViewQuery, let workload, path.hasPrefix("input/"),
               PreparationArtifactPolicy.isSegmentName(String(path.dropFirst(6))) { return Int(workload.limit(.retainedBytes)) }

            let parts = try WorkflowArchiveCodec.pathParts(path)
            if let workload, usesPreparationWorkload, parts.count == 3, parts[0] == "input",
               ["preparation","reuse"].contains(parts[1]), PreparationArtifactPolicy.isCapacityFileName(parts[2]) {
                // The mode's original path validator still owns namespace admission.
                let base: Int
                switch self {
                case .preparation:base = try WorkflowPreparationToolContract.fileLimit(path:path)
                case .preparationInspection:base = try WorkflowPreparationInspectionToolContract.fileLimit(path:path)
                case .backendFit:base = try BackendFitOperationToolContract.fileLimit(path:path)
                case .preview:base = try WorkflowPreviewToolContract.fileLimit(path:path)
                case .modelTraining:base = try WorkflowModelTrainingToolContract.fileLimit(path:path)
                default:throw WorkflowFileLeaseError.unsafePath
                }
                _ = base;return Int(workload.limit(.retainedBytes))
            }
            switch self {
            case .selfPlay:return try SelfPlayToolContract.fileLimit(path:path)
            case .gameSession:return try GameSessionToolContract.fileLimit(path:path)
            case .tableViewQuery: return try WorkflowTableViewQueryToolContract.fileLimit(path: path, workload: workload)
            case .backendFit: return try BackendFitOperationToolContract.fileLimit(path: path)
            case .preview: return try WorkflowPreviewToolContract.fileLimit(path: path)
            case .preparation: return try WorkflowPreparationToolContract.fileLimit(path: path)
            case .sourceInspection: return try WorkflowSourceInspectionToolContract.fileLimit(path: path)
            case .preparationInspection: return try WorkflowPreparationInspectionToolContract.fileLimit(path: path)
            case .modelTraining: return try WorkflowModelTrainingToolContract.fileLimit(path: path)
            }
        }
    }
    enum Input {
        case bytes(path: String, data: Data)
        case file(path: String, source: URL, expectedSHA256: String?)
        case anchoredFile(path: String, root: URL, parts: [String], expectedSHA256: String)
        var expectedSHA256: String? {
            switch self {case .bytes:return nil;case .file(_,_,let pin):return pin;case .anchoredFile(_,_,_,let pin):return pin}
        }
        var path: String {
            switch self { case .bytes(let path, _), .file(let path, _, _), .anchoredFile(let path, _, _, _): return path }
        }
    }
    let directory: URL
    let files: [WorkflowArchiveFile]
    private let device: dev_t
    private let inode: ino_t
    private let scratchLifetime: WorkflowScratchCleanup.Lifetime

    /// Only this unique stage is removable, even when a later operation fails.
    func remove() {
        var info = stat()
        guard lstat(directory.path, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR),
              info.st_dev == device, info.st_ino == inode else { return }
        try? FileManager.default.removeItem(at: directory)
        scratchLifetime.close()
    }
    /// Admission estimate from held regular-file metadata, without copying payloads or creating a stage.
    /// Capture must still enforce this bound because sources can change after admission.
    static func byteCountBound(inputs: [Input], purpose: Purpose = .preview, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        try validate(inputs: inputs, purpose: purpose, workload: workload)
        var total = 0,decisionBytes = 0
        for input in inputs {
            let limit = try purpose.fileLimit(input.path, workload: workload)
            let count: Int
            switch input {
            case .bytes(_, let data):
                guard data.count <= limit else { throw WorkflowArchiveError("Captured input exceeds its file allowance.") }
                count = data.count
            case .file, .anchoredFile:
                let (descriptor, metadata) = try openInput(input, limit: limit)
                Darwin.close(descriptor)
                count = Int(metadata.st_size)
            }
            guard count <= purpose.captureBytes(workload) - total else { throw WorkflowArchiveError("Captured inputs exceed their capture allowance.") }
            if purpose.allowsDecisionModels {
                if input.path.hasPrefix("input/decision-models/") {decisionBytes += count}
                guard decisionBytes<=2_100_000_000,total+count-decisionBytes<=Int(workload?.limit(.retainedBytes) ?? 200_000_000)+1_300_000_000 else {throw WorkflowArchiveError("Captured data or decision-model inputs exceed their separate allowances.")}
            }
            total += count
        }
        return total
    }

    static func stage(workspace: URL, inputs: [Input], purpose: Purpose = .preview, workload: ExecutionWorkloadPolicy? = nil, stageID: UUID = UUID(), maximumBytes: Int? = nil, checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> Self {
        guard maximumBytes == nil || maximumBytes! >= 0 else { throw WorkflowArchiveError("Capture reservation must not be negative.") }
        let totalLimit = min(maximumBytes ?? purpose.captureBytes(workload), purpose.captureBytes(workload))
        try validate(inputs: inputs, purpose: purpose, workload: workload)
        try checkCancellation()
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        guard let resolved = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let canonicalRoot = URL(fileURLWithPath: String(cString: resolved)); free(resolved)
        let root = try WorkflowEvaluationOwnership.openDirectory(canonicalRoot)
        defer { Darwin.close(root) }
        let namespace = try child(root, purpose.namespace, exclusive: false)
        defer { Darwin.close(namespace) }
        let name = stageID.uuidString
        let target = try child(namespace, name, exclusive: true)
        defer { Darwin.close(target) }
        var info = stat(); guard fstat(target, &info) == 0 else { throw WorkflowFileLeaseError.unsafePath }
        let destination = canonicalRoot.appendingPathComponent(purpose.namespace + "/" + name)
        var result = Self(directory: destination, files: [], device: info.st_dev, inode: info.st_ino, scratchLifetime: try WorkflowScratchCleanup.register(directory: destination))
        do {
            var inventory: [WorkflowArchiveFile] = [], total = 0, decisionBytes = 0
            for input in inputs.sorted(by: { $0.path < $1.path }) {
                try checkCancellation()
                let limit = try purpose.fileLimit(input.path, workload: workload)
                let parts = try WorkflowArchiveCodec.pathParts(input.path)
                var parent = dup(target); guard parent >= 0 else { throw WorkflowFileLeaseError.unsafePath }
                defer { Darwin.close(parent) }
                for part in parts.dropLast() {
                    let next = try child(parent, part, exclusive: false); Darwin.close(parent); parent = next
                }
                guard let leaf = parts.last else { throw WorkflowFileLeaseError.unsafePath }
                let output = openat(parent, leaf, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
                guard output >= 0 else { throw WorkflowFileLeaseError.unsafePath }
                defer { Darwin.close(output) }
                let decisionInput=purpose.allowsDecisionModels && input.path.hasPrefix("input/decision-models/")
                var hash = SHA256(), count = 0
                func append(_ bytes: Data) throws {
                    try checkCancellation()
                    guard bytes.count <= limit - count, bytes.count <= totalLimit - total else { throw WorkflowArchiveError("Captured inputs exceed their capture allowance.") }
                    if purpose.allowsDecisionModels {
                        let used=decisionInput ? decisionBytes:total-decisionBytes,cap=decisionInput ? 2_100_000_000:Int(workload?.limit(.retainedBytes) ?? 200_000_000)+1_300_000_000
                        guard bytes.count<=cap-used else {throw WorkflowArchiveError("Captured data or decision-model inputs exceed their separate allowances.")}
                    }
                    try bytes.withUnsafeBytes { buffer in
                        var written = 0
                        while written < buffer.count {
                            let amount = Darwin.write(output, buffer.baseAddress!.advanced(by: written), buffer.count - written)
                            if amount < 0, errno == EINTR { continue }
                            guard amount > 0 else { throw WorkflowFileLeaseError.unsafePath }
                            written += amount
                        }
                    }
                    hash.update(data: bytes); count += bytes.count; total += bytes.count;if decisionInput {decisionBytes += bytes.count}
                }
                switch input {
                case .bytes(_, let data):
                    guard data.count <= limit else { throw WorkflowArchiveError("Captured input exceeds its file allowance.") }
                    for offset in stride(from: 0, to: data.count, by: 65_536) {
                        try append(data.subdata(in: offset..<min(data.count, offset + 65_536)))
                    }
                case .file, .anchoredFile:
                    let (inputFD, before) = try openInput(input, limit: limit)
                    defer { Darwin.close(inputFD) }
                    var buffer = [UInt8](repeating: 0, count: 65_536)
                    while true {
                        try checkCancellation()
                        let size = Darwin.read(inputFD, &buffer, buffer.count)
                        if size < 0, errno == EINTR { continue }
                        guard size >= 0 else { throw WorkflowFileLeaseError.unsafePath }
                        if size == 0 { break }
                        try append(Data(buffer.prefix(size)))
                    }
                    var after = stat()
                    guard fstat(inputFD, &after) == 0, count == before.st_size, before.st_size == after.st_size,
                          before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
                          before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec, before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
                        throw WorkflowArchiveError("Source changed while being captured. Try again.")
                    }
                }
                let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
                if let expected = input.expectedSHA256, digest != expected {
                    throw WorkflowArchiveError("Input changed from its captured integrity pin. Capture it again.")
                }
                guard fsync(output) == 0 else { throw WorkflowFileLeaseError.unsafePath }
                inventory.append(.init(path: input.path, bytes: count, sha256: digest))
            }
            try checkCancellation()
            result = Self(directory: destination, files: inventory, device: info.st_dev, inode: info.st_ino, scratchLifetime: result.scratchLifetime)
            return result
        } catch { result.remove(); throw error }
    }
    private static func validate(inputs: [Input], purpose: Purpose, workload: ExecutionWorkloadPolicy? = nil) throws {
        guard !inputs.isEmpty, inputs.count <= purpose.captureCount(workload), Set(inputs.map(\.path)).count == inputs.count else { throw WorkflowFileLeaseError.unsafePath }
        if purpose.allowsDecisionModels {guard inputs.filter({$0.path.hasPrefix("input/decision-models/") && $0.path.hasSuffix("/model.safetensors")}).count<=1 else {throw WorkflowArchiveError("Prepare one distinct decision model per run.")}}
        for input in inputs {
            _ = try WorkflowArchiveCodec.pathParts(input.path)
            _ = try purpose.fileLimit(input.path, workload: workload)
        }
    }

    private static func openInput(_ input: Input, limit: Int) throws -> (Int32, stat) {
        switch input {
        case .file(_, let source, let expected): return try openSource(source, expected: expected, limit: limit)
        case .anchoredFile(_, let root, let parts, let pin):
            guard WorkflowArchiveCodec.isHash(pin) else { throw WorkflowFileLeaseError.unsafePath }
            let fd = try DiagnosticFileIO.openRegularFile(root: root, parts: parts, limit: limit)
            var info = stat()
            guard fstat(fd, &info) == 0 else { Darwin.close(fd); throw WorkflowFileLeaseError.unsafePath }
            return (fd, info)
        case .bytes: throw WorkflowFileLeaseError.unsafePath
        }
    }

    private static func openSource(_ source: URL, expected: String?, limit: Int) throws -> (Int32, stat) {
        guard source.isFileURL, !source.path.contains("\0"), expected == nil || WorkflowArchiveCodec.isHash(expected!) else { throw WorkflowFileLeaseError.unsafePath }
        let descriptor = Darwin.open(source.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0, metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              metadata.st_size >= 0, metadata.st_size <= limit else {
            Darwin.close(descriptor)
            throw WorkflowFileLeaseError.unsafePath
        }
        return (descriptor, metadata)
    }

    private static func child(_ parent: Int32, _ name: String, exclusive: Bool) throws -> Int32 {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\0") else { throw WorkflowFileLeaseError.unsafePath }
        if mkdirat(parent, name, 0o700) != 0, exclusive || errno != EEXIST { throw WorkflowFileLeaseError.unsafePath }
        let fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        return fd
    }
}
