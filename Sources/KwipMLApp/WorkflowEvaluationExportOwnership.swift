import Foundation
import Darwin
import KwipMLCore

struct WorkflowEvaluationExportOwnershipMetadata: Codable, Equatable {
    let schemaVersion: Int
    let kind: String
    let exportID: String
    let projectID: String
    let requestIdentity: String
    let requestSHA256: String
    let directoryIdentity: WorkflowFileIdentity
    let ownerLockIdentity: WorkflowFileIdentity
    let activityLockIdentity: WorkflowFileIdentity
}

/// Independent export authority. The source evaluation inspection is retained separately.
enum WorkflowEvaluationExportOwnership {
    static func create(workspace: URL, request: WorkflowEvaluationExportRequest) throws -> WorkflowEvaluationExportOwner {
        try request.validate()
        guard workspace.isFileURL, !workspace.path.contains("\0"),
              let canonical = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        let root = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        guard request.evaluationDirectory == root.appendingPathComponent("workflow-evaluations")
            .appendingPathComponent(request.projectID).appendingPathComponent(request.evaluationID).path else {
            throw WorkflowFileLeaseError.unsafePath
        }
        let fd = try WorkflowEvaluationOwnership.openDirectory(root)
        defer { Darwin.close(fd) }
        let rootIdentity = try WorkflowEvaluationOwnership.directoryIdentity(fd)
        func child(_ parent: Int32, _ name: String, exclusive: Bool = false) throws -> Int32 {
            if mkdirat(parent, name, 0o700) != 0 {
                guard !exclusive, errno == EEXIST else { throw WorkflowFileLeaseError.system(errno) }
            } else if fsync(parent) != 0 { throw WorkflowFileLeaseError.system(errno) }
            let result = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard result >= 0 else { throw WorkflowFileLeaseError.unsafePath }
            do { _ = try WorkflowEvaluationOwnership.directoryIdentity(result); return result }
            catch { Darwin.close(result); throw error }
        }
        let namespace = try child(fd, "workflow-evaluation-exports")
        defer { Darwin.close(namespace) }
        let project = try child(namespace, request.projectID)
        defer { Darwin.close(project) }
        let leaf = try child(project, request.exportID, exclusive: true)
        defer { Darwin.close(leaf) }
        let identity = try WorkflowEvaluationOwnership.directoryIdentity(leaf)
        let directory = root.appendingPathComponent("workflow-evaluation-exports")
            .appendingPathComponent(request.projectID).appendingPathComponent(request.exportID)
        let owner = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive,
            create: true, createExclusive: true, requireLocalAPFS: true)
        do {
            let activity = try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .exclusive,
                create: true, createExclusive: true, requireLocalAPFS: true)
            do {
                guard owner.parentIdentity == identity, activity.parentIdentity == identity else { throw WorkflowFileLeaseError.unsafePath }
                let metadata = WorkflowEvaluationExportOwnershipMetadata(schemaVersion: 1, kind: "portableEvaluationExport",
                    exportID: request.exportID, projectID: request.projectID, requestIdentity: request.requestIdentity,
                    requestSHA256: try request.requestSHA256(), directoryIdentity: identity,
                    ownerLockIdentity: owner.identity, activityLockIdentity: activity.identity)
                let bytes = try WorkflowArchiveCodec.encode(metadata)
                try owner.publishOwnershipMetadata(bytes)
                let result = WorkflowEvaluationExportOwner(directory: directory, metadata: metadata, bytes: bytes, owner: owner, activity: activity)
                let current = try WorkflowEvaluationOwnership.openDirectory(root)
                defer { Darwin.close(current) }
                guard try WorkflowEvaluationOwnership.directoryIdentity(current) == rootIdentity else { throw WorkflowFileLeaseError.unsafePath }
                try result.check()
                return result
            } catch { activity.close(); throw error }
        } catch { owner.close(); throw error }
    }

    fileprivate static func checkPaths(_ directory: URL, _ metadata: WorkflowEvaluationExportOwnershipMetadata) throws {
        let fd = try WorkflowEvaluationOwnership.openDirectory(directory)
        defer { Darwin.close(fd) }
        guard directory.lastPathComponent == metadata.exportID,
              directory.deletingLastPathComponent().lastPathComponent == metadata.projectID,
              directory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-evaluation-exports",
              try WorkflowEvaluationOwnership.directoryIdentity(fd) == metadata.directoryIdentity else { throw WorkflowFileLeaseError.unsafePath }
        for (name, expected) in [("owner.lock", metadata.ownerLockIdentity), ("activity.lock", metadata.activityLockIdentity)] {
            var info = stat()
            guard fstatat(fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0,
                  info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
                  info.st_uid == geteuid(), info.st_mode & 0o022 == 0,
                  WorkflowEvaluationOwnership.identity(info) == expected else { throw WorkflowFileLeaseError.unsafePath }
        }
    }
    fileprivate static func requireUnclaimed(_ directory: URL, _ metadata: WorkflowEvaluationExportOwnershipMetadata) throws {
        let fd = try WorkflowEvaluationOwnership.openDirectory(directory)
        defer { Darwin.close(fd) }
        guard try WorkflowEvaluationOwnership.directoryIdentity(fd) == metadata.directoryIdentity else { throw WorkflowFileLeaseError.unsafePath }
        var info = stat()
        guard fstatat(fd, "claim.json", &info, AT_SYMLINK_NOFOLLOW) != 0 else { throw WorkflowFileLeaseError.unsafePath }
        guard errno == ENOENT else { throw WorkflowFileLeaseError.system(errno) }
    }
}

final class WorkflowEvaluationExportOwner {
    let directory: URL
    let metadata: WorkflowEvaluationExportOwnershipMetadata
    let ownershipSHA256: String
    private let bytes: Data
    private let owner: WorkflowFileLease
    private var activity: WorkflowFileLease?
    private var closed = false
    private var launched = false
    private var launchBudgetSHA256: String?
    fileprivate init(directory: URL, metadata: WorkflowEvaluationExportOwnershipMetadata, bytes: Data,
                     owner: WorkflowFileLease, activity: WorkflowFileLease) {
        self.directory = directory; self.metadata = metadata; self.bytes = bytes
        self.owner = owner; self.activity = activity
        ownershipSHA256 = WorkflowArchiveCodec.hash(bytes)
    }
    func check() throws {
        guard !closed else { throw WorkflowFileLeaseError.closed }
        try owner.checkIdentity()
        try WorkflowEvaluationExportOwnership.checkPaths(directory, metadata)
        guard owner.parentIdentity == metadata.directoryIdentity, try owner.readOwnershipMetadata() == bytes else { throw WorkflowFileLeaseError.unsafePath }
        if let activity { try activity.checkIdentity() }
        else {
            let probe = try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .shared,
                expectedIdentity: metadata.activityLockIdentity, requireLocalAPFS: true)
            defer { probe.close() }
            try probe.checkIdentity()
        }
    }
    func validateDirectory(_ url: URL) throws {
        guard url.isFileURL, url.path == directory.path else { throw WorkflowFileLeaseError.unsafePath }
        try check()
    }
    func publishRequest(_ request: WorkflowEvaluationExportRequest) throws {
        try check()
        guard !launched, request.exportID == metadata.exportID, request.projectID == metadata.projectID,
              request.requestIdentity == metadata.requestIdentity, try request.requestSHA256() == metadata.requestSHA256 else {
            throw WorkflowFileLeaseError.unsafePath
        }
        try WorkflowEvaluationExportOwnership.requireUnclaimed(directory, metadata)
        try DiagnosticFileIO.writeImmutable(request.encoded(), root: directory, parts: ["request.json"], limit: 256_000)
        try check()
    }
    func publishLaunchBudget(remainingSeconds: Double) throws -> String {
        try check()
        guard !launched else { throw WorkflowFileLeaseError.unsafePath }
        try WorkflowEvaluationExportOwnership.requireUnclaimed(directory, metadata)
        let raw = try WorkflowEvaluationOwnership.readProtocol(directory, name: "request.json", limit: 256_000)
        let request = try WorkflowEvaluationExportRequest.decode(raw)
        guard WorkflowArchiveCodec.hash(raw) == metadata.requestSHA256,
              request.exportID == metadata.exportID, request.projectID == metadata.projectID,
              request.requestIdentity == metadata.requestIdentity, let budget = request.executionBudget else { throw WorkflowFileLeaseError.unsafePath }
        let pin = try WorkflowWorkerLaunchBudget.publish(directory: directory, requestSHA256: metadata.requestSHA256,
            budget: budget, remainingSeconds: remainingSeconds)
        launchBudgetSHA256 = pin
        try check()
        return pin
    }
    @discardableResult func prepareLaunch() throws -> String {
        try check()
        guard !launched else { throw WorkflowFileLeaseError.unsafePath }
        try WorkflowEvaluationExportOwnership.requireUnclaimed(directory, metadata)
        let raw = try WorkflowEvaluationOwnership.readProtocol(directory, name: "request.json", limit: 256_000)
        let request = try WorkflowEvaluationExportRequest.decode(raw)
        guard WorkflowArchiveCodec.hash(raw) == metadata.requestSHA256,
              request.exportID == metadata.exportID, request.projectID == metadata.projectID,
              request.requestIdentity == metadata.requestIdentity else { throw WorkflowFileLeaseError.unsafePath }
        try WorkflowWorkerLaunchBudget.verify(directory: directory, requestSHA256: metadata.requestSHA256,
            budget: request.executionBudget, expectedSHA256: launchBudgetSHA256)
        try check()
        launched = true
        activity?.close(); activity = nil
        try check()
        return ownershipSHA256
    }
    func close() {
        guard !closed else { return }
        closed = true; activity?.close(); activity = nil; owner.close()
    }
    deinit { close() }
}
