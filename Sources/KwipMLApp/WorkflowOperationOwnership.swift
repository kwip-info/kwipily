import Foundation
import Darwin
import KwipMLCore

struct WorkflowOperationOwnershipMetadata: Codable, Equatable {
    let schemaVersion: Int
    let kind: String
    let operationID: String
    let projectID: String
    let requestIdentity: String
    let requestSHA256: String
    let directoryIdentity: WorkflowFileIdentity
    let ownerLockIdentity: WorkflowFileIdentity
    let activityLockIdentity: WorkflowFileIdentity
}

/// Retained, one-shot authority shared by workflow phases and typed workspace tools.
enum WorkflowOperationOwnership {
    static func create(workspace: URL, request: WorkflowOperationRequest) throws -> WorkflowOperationOwner {
        try request.validate(workspace: workspace)
        return try create(workspace: workspace, operationID: request.operationID, projectID: request.projectID,
            requestIdentity: request.requestIdentity, requestSHA256: request.requestSHA256()) { raw, directory in
                let decoded = try WorkflowOperationRequest.decode(raw)
                try decoded.validate(workspace: directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent())
            }
    }
    static func create(workspace: URL, request: WorkflowToolRequest) throws -> WorkflowOperationOwner {
        let directory = workspace.appendingPathComponent("workflow-operations/\(request.projectID)/\(request.operationID)")
        try request.validate(directory: directory, verifyInputs: false)
        return try create(workspace: workspace, operationID: request.operationID, projectID: request.projectID,
            requestIdentity: request.requestIdentity, requestSHA256: request.requestSHA256()) { raw, directory in
                try WorkflowToolRequest.decode(raw).validate(directory: directory, verifyInputs: true)
            }
    }
    private static func create(workspace: URL, operationID: String, projectID: String,
                               requestIdentity: String, requestSHA256: String,
                               validateRequest: @escaping (Data, URL) throws -> Void) throws -> WorkflowOperationOwner {
        guard workspace.isFileURL, !workspace.path.contains("\0"),
              let canonical = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        let root = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        guard root.path == workspace.path else { throw WorkflowFileLeaseError.unsafePath }
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
        let namespace = try child(fd, "workflow-operations")
        defer { Darwin.close(namespace) }
        let project = try child(namespace, projectID)
        defer { Darwin.close(project) }
        let leaf = try child(project, operationID, exclusive: true)
        defer { Darwin.close(leaf) }
        let identity = try WorkflowEvaluationOwnership.directoryIdentity(leaf)
        let directory = root.appendingPathComponent("workflow-operations")
            .appendingPathComponent(projectID).appendingPathComponent(operationID)
        let owner = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive,
            create: true, createExclusive: true, requireLocalAPFS: true)
        do {
            let activity = try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .exclusive,
                create: true, createExclusive: true, requireLocalAPFS: true)
            do {
                guard owner.parentIdentity == identity, activity.parentIdentity == identity else { throw WorkflowFileLeaseError.unsafePath }
                let metadata = WorkflowOperationOwnershipMetadata(schemaVersion: 1, kind: "workflowOperation",
                    operationID: operationID, projectID: projectID, requestIdentity: requestIdentity,
                    requestSHA256: requestSHA256, directoryIdentity: identity,
                    ownerLockIdentity: owner.identity, activityLockIdentity: activity.identity)
                let bytes = try WorkflowArchiveCodec.encode(metadata)
                try owner.publishOwnershipMetadata(bytes)
                let result = WorkflowOperationOwner(directory: directory, metadata: metadata, bytes: bytes, owner: owner, activity: activity, validateRequest: validateRequest)
                let current = try WorkflowEvaluationOwnership.openDirectory(root)
                defer { Darwin.close(current) }
                guard try WorkflowEvaluationOwnership.directoryIdentity(current) == rootIdentity else { throw WorkflowFileLeaseError.unsafePath }
                try result.check()
                return result
            } catch { activity.close(); throw error }
        } catch { owner.close(); throw error }
    }

    fileprivate static func checkPaths(_ directory: URL, _ metadata: WorkflowOperationOwnershipMetadata) throws {
        let fd = try WorkflowEvaluationOwnership.openDirectory(directory)
        defer { Darwin.close(fd) }
        guard directory.lastPathComponent == metadata.operationID,
              directory.deletingLastPathComponent().lastPathComponent == metadata.projectID,
              directory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations",
              try WorkflowEvaluationOwnership.directoryIdentity(fd) == metadata.directoryIdentity else { throw WorkflowFileLeaseError.unsafePath }
        for (name, expected) in [("owner.lock", metadata.ownerLockIdentity), ("activity.lock", metadata.activityLockIdentity)] {
            var info = stat()
            guard fstatat(fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0,
                  info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
                  info.st_uid == geteuid(), info.st_mode & 0o022 == 0,
                  WorkflowEvaluationOwnership.identity(info) == expected else { throw WorkflowFileLeaseError.unsafePath }
        }
    }
    fileprivate static func requireUnclaimed(_ directory: URL, _ metadata: WorkflowOperationOwnershipMetadata) throws {
        let fd = try WorkflowEvaluationOwnership.openDirectory(directory)
        defer { Darwin.close(fd) }
        guard try WorkflowEvaluationOwnership.directoryIdentity(fd) == metadata.directoryIdentity else { throw WorkflowFileLeaseError.unsafePath }
        var info = stat()
        guard fstatat(fd, "claim.json", &info, AT_SYMLINK_NOFOLLOW) != 0 else { throw WorkflowFileLeaseError.unsafePath }
        guard errno == ENOENT else { throw WorkflowFileLeaseError.system(errno) }
    }
}

final class WorkflowOperationOwner {
    let directory: URL
    let metadata: WorkflowOperationOwnershipMetadata
    let ownershipSHA256: String
    private let bytes: Data
    private let owner: WorkflowFileLease
    private let validateRequest: (Data, URL) throws -> Void
    private var activity: WorkflowFileLease?
    private var closed = false
    private var launched = false
    fileprivate init(directory: URL, metadata: WorkflowOperationOwnershipMetadata, bytes: Data,
                     owner: WorkflowFileLease, activity: WorkflowFileLease,
                     validateRequest: @escaping (Data, URL) throws -> Void) {
        self.directory = directory; self.metadata = metadata; self.bytes = bytes
        self.owner = owner; self.activity = activity; self.validateRequest = validateRequest
        ownershipSHA256 = WorkflowArchiveCodec.hash(bytes)
    }
    func check() throws {
        guard !closed else { throw WorkflowFileLeaseError.closed }
        try owner.checkIdentity()
        try WorkflowOperationOwnership.checkPaths(directory, metadata)
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
    func publishRequest(_ request: WorkflowOperationRequest) throws { try publishRequestBytes(request.encoded()) }
    func publishRequest(_ request: WorkflowToolRequest) throws { try publishRequestBytes(request.encoded()) }
    private func publishRequestBytes(_ raw: Data) throws {
        try check()
        guard !launched, WorkflowArchiveCodec.hash(raw) == metadata.requestSHA256 else { throw WorkflowFileLeaseError.unsafePath }
        try validateRequest(raw, directory)
        try WorkflowOperationOwnership.requireUnclaimed(directory, metadata)
        try DiagnosticFileIO.writeImmutable(raw, root: directory, parts: ["request.json"], limit: 256_000)
        try check()
    }
    @discardableResult func prepareLaunch() throws -> String {
        try check()
        guard !launched else { throw WorkflowFileLeaseError.unsafePath }
        try WorkflowOperationOwnership.requireUnclaimed(directory, metadata)
        let raw = try WorkflowEvaluationOwnership.readProtocol(directory, name: "request.json", limit: 256_000)
        guard WorkflowArchiveCodec.hash(raw) == metadata.requestSHA256 else { throw WorkflowFileLeaseError.unsafePath }
        try validateRequest(raw, directory)
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
