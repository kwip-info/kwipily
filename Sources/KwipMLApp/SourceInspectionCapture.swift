import Foundation
import Darwin
import KwipMLCore

/// Capture on a background task; retain the source permission until the private copy is complete.
final class SourceInspectionCapture {
    static let maximumSampleBytes = 1_048_576
    private let stage: WorkflowPreviewCapture
    let sourceURL: URL
    let sha256: String
    let storageReservation: WorkflowCaptureStorageReservation?

    private init(stage: WorkflowPreviewCapture, sourceURL: URL, sha256: String,
                 storageReservation: WorkflowCaptureStorageReservation?) {
        self.stage = stage; self.sourceURL = sourceURL; self.sha256 = sha256
        self.storageReservation = storageReservation
    }
    func remove() { stage.remove(); storageReservation?.close() }
    deinit { remove() }

    static func capture(path: String, workspace: URL, accountingRoot: URL? = nil, sourcePermissions: URL,
                        window: ExecutionBudgetWindow, projectID: String? = nil,
                        storageLimitBytes: Int64? = nil, completeFile: Bool = false) throws -> SourceInspectionCapture {
        try check(window)
        return try SourceAccess(directory: sourcePermissions).withAccess(to: path) { url in
            try check(window)
            let relative = "input/data/" + url.lastPathComponent
            let bytes = try readPrefix(url, window: window, completeFile: completeFile)
            let inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: relative, data: bytes)]
            let sourceStageID = UUID()
            var reservation: WorkflowCaptureStorageReservation?
            if let storageLimitBytes {
                guard let projectID else { throw WorkflowArchiveError("Storage admission requires a project identity.") }
                let count = try WorkflowPreviewCapture.byteCountBound(inputs: inputs, purpose: .sourceInspection)
                reservation = try WorkflowCaptureStorageReservation(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
                    purpose: .sourceInspection, maximumCaptureBytes: count,
                    additionalScopes: [try WorkflowStorageAdmission.Scope(namespace: .sourceInspectionStaging, components: [sourceStageID.uuidString])],
                    additionalBytes: Int64(count), limitBytes: storageLimitBytes)
            }
            var retained = false
            defer { if !retained { reservation?.close() } }
            let stage = try WorkflowPreviewCapture.stage(workspace: workspace, inputs: inputs,
                purpose: .sourceInspection, stageID: sourceStageID, maximumBytes: reservation?.maximumCaptureBytes,
                checkCancellation: { try check(window) })
            do {
                try check(window)
                guard let file = stage.files.first, stage.files.count == 1 else { throw WorkflowFileLeaseError.unsafePath }
                let result = SourceInspectionCapture(stage: stage, sourceURL: stage.directory.appendingPathComponent(relative),
                                                     sha256: file.sha256, storageReservation: reservation)
                retained = true
                return result
            } catch { stage.remove(); throw error }
        }
    }

    /// Revalidate only the inspected prefix, never claiming complete-file identity.
    static func verifyCurrent(path: String, sourcePermissions: URL, window: ExecutionBudgetWindow,
                              expectedSHA256: String, completeFile: Bool = false) throws {
        try check(window)
        guard WorkflowArchiveCodec.isHash(expectedSHA256) else { throw WorkflowFileLeaseError.unsafePath }
        try SourceAccess(directory: sourcePermissions).withAccess(to: path) { url in
            try check(window)
            let bytes = try readPrefix(url, window: window, completeFile: completeFile)
            try check(window)
            guard WorkflowArchiveCodec.hash(bytes) == expectedSHA256 else {
                throw WorkflowArchiveError("Source changed during inspection. Refresh its columns.")
            }
            try check(window)
        }
    }

    /// A security-scoped, no-follow regular-file read with a fixed byte budget.
    /// File stability is checked during this read; bytes beyond the prefix are not evidence.
    private static func readPrefix(_ url: URL, window: ExecutionBudgetWindow, completeFile: Bool) throws -> Data {
        guard url.isFileURL, !url.path.contains("\0") else { throw WorkflowFileLeaseError.unsafePath }
        let fd = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(fd) }
        var before = stat()
        guard fstat(fd, &before) == 0, before.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), before.st_size >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        if completeFile && before.st_size > 10_000_000 { throw WorkflowArchiveError("Table Input source exceeds the 10 MB capture allowance.") }
        let count = completeFile ? Int(before.st_size) : Int(min(before.st_size, off_t(maximumSampleBytes)))
        var bytes = Data(), buffer = [UInt8](repeating: 0, count: 65_536)
        while bytes.count < count {
            try check(window)
            let size = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress!, min($0.count, count - bytes.count)) }
            if size < 0 && errno == EINTR { continue }
            guard size > 0 else { throw WorkflowFileLeaseError.unsafePath }
            bytes.append(contentsOf: buffer.prefix(size))
        }
        var after = stat()
        guard fstat(fd, &after) == 0, before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec, before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
            throw WorkflowArchiveError("Source changed during inspection. Refresh its preview.")
        }
        return bytes
    }

    private static func check(_ window: ExecutionBudgetWindow) throws {
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
    }
}
