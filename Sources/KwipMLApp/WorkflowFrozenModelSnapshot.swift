import Foundation
import Darwin
import KwipMLCore

/// Only native staging creates this shared cleanup authority. Decoded/history
/// snapshots remain ordinary nonowning values, even when their bytes are verified.
final class WorkflowFrozenSnapshotDirectory {
    let directory: URL
    private let identity: WorkflowFileIdentity
    private let lock = NSLock()
    private var removed = false
    private let removeOnDeinit: Bool
    init(directory: URL, removeOnDeinit: Bool = true) throws {
        self.removeOnDeinit = removeOnDeinit
        self.directory = try Self.canonicalDirectory(directory)
        let fd = try WorkflowEvaluationOwnership.openDirectory(self.directory)
        defer { Darwin.close(fd) }
        self.identity = try WorkflowEvaluationOwnership.directoryIdentity(fd)
    }
    fileprivate static func canonicalDirectory(_ directory: URL) throws -> URL {
        guard directory.isFileURL, !directory.path.contains("\0") else { throw WorkflowFileLeaseError.unsafePath }
        var entry = stat()
        guard lstat(directory.path, &entry) == 0, entry.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR),
              let pointer = realpath(directory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(pointer) }
        let canonical = URL(fileURLWithPath: String(cString: pointer), isDirectory: true)
        let fd = try WorkflowEvaluationOwnership.openDirectory(canonical)
        defer { Darwin.close(fd) }
        guard try WorkflowEvaluationOwnership.directoryIdentity(fd) == WorkflowEvaluationOwnership.identity(entry) else {
            throw WorkflowFileLeaseError.unsafePath
        }
        return canonical
    }
    func check() throws {
        lock.lock(); defer { lock.unlock() }
        guard !removed else { throw WorkflowFileLeaseError.closed }
        let fd = try WorkflowEvaluationOwnership.openDirectory(directory)
        defer { Darwin.close(fd) }
        guard try WorkflowEvaluationOwnership.directoryIdentity(fd) == identity else { throw WorkflowFileLeaseError.unsafePath }
    }
    func remove() {
        lock.lock(); defer { lock.unlock() }
        guard !removed else { return }
        removed = true
        guard let fd = try? WorkflowEvaluationOwnership.openDirectory(directory) else { return }
        defer { Darwin.close(fd) }
        guard (try? WorkflowEvaluationOwnership.directoryIdentity(fd)) == identity else { return }
        try? FileManager.default.removeItem(at: directory)
    }
    deinit { if removeOnDeinit { remove() } }
}

/// Private inference input, not a portable model export or a complete training archive.
/// Contains only independently verified parent evidence and the full preparation inventory.
struct WorkflowFrozenModelSnapshot {
    let reference:WorkflowFrozenModelReference
    let directory:URL
    let files:[WorkflowArchiveFile]
    let receiptSHA256:String
    private let directoryIdentity: WorkflowFileIdentity?
    private let cleanupDirectory: URL?
    private let ownership: WorkflowFrozenSnapshotDirectory?
    init(reference: WorkflowFrozenModelReference, directory: URL, files: [WorkflowArchiveFile], receiptSHA256: String, ownership: WorkflowFrozenSnapshotDirectory? = nil) {
        let directory = directory.standardizedFileURL
        self.reference = reference; self.directory = directory; self.files = files; self.receiptSHA256 = receiptSHA256
        self.ownership = ownership
        let cleanup = try? WorkflowFrozenSnapshotDirectory.canonicalDirectory(directory)
        cleanupDirectory = cleanup
        if let cleanup, let fd = try? WorkflowEvaluationOwnership.openDirectory(cleanup) {
            directoryIdentity = try? WorkflowEvaluationOwnership.directoryIdentity(fd)
            Darwin.close(fd)
        } else { directoryIdentity = nil }
    }
    /// Remove only the directory captured by this snapshot, never a replacement.
    func remove() {
        if let ownership { ownership.remove(); return }
        guard let directoryIdentity, let cleanupDirectory,
              let fd = try? WorkflowEvaluationOwnership.openDirectory(cleanupDirectory) else { return }
        defer { Darwin.close(fd) }
        guard (try? WorkflowEvaluationOwnership.directoryIdentity(fd)) == directoryIdentity else { return }
        try? FileManager.default.removeItem(at: cleanupDirectory)
    }
    private struct Receipt:Codable {
        let schemaVersion:Int
        let reference:WorkflowFrozenModelReference
        let testEndpoint:DataFlowEndpoint
        var partitionRole:String? = nil
        let files:[WorkflowArchiveFile]
    }
    private struct PreparationInventory:Decodable {
        struct Entry:Decodable {let path:String;let kind:String;let nodeID:String?;let port:String?}
        let artifacts:[Entry]
    }
    static func fileLimit(_ path:String)throws->Int {
        switch path {
        case "outcome.json":return 256_000
        case "capture.json":return 8_000_000
        case "workflow.json","model-registry.json","data-registry.json":return 2_000_000
        case "runtime-evidence.json":return 256_000
        case "training/artifacts/manifest.json":return 20_000_000
        case "training/artifacts/checkpoint.pt":return 100_000_000
        case "preparation/artifacts/manifest.json":return 2_000_000
        default:
            let parts=try WorkflowArchiveCodec.pathParts(path)
            guard parts.count == 3,parts[0] == "preparation",parts[1] == "artifacts" else {
                throw WorkflowArchiveError("Unsupported frozen evaluation snapshot file.")
            }
            return 50_000_000
        }
    }
    /// Selection authenticates a current completed archive; binding alone is not authority.
    static func select(_ history:WorkflowHistoryReference,root:URL)throws->WorkflowFrozenModelReference {
        let (value,outcomeBytes)=try verifiedParent(history,root:root)
        guard let model=value.outcome.modelManifestSHA256,let preparation=value.outcome.preparationManifestSHA256,
              let checkpoint=value.outcome.artifacts.first(where:{$0.path == "training/artifacts/checkpoint.pt"}) else {
            throw WorkflowArchiveError("Select a completed prepared model with a saved checkpoint.")
        }
        return .init(projectID:history.projectID,directoryID:history.directoryID.uuidString,runID:value.outcome.runID,
                     requestIdentity:value.outcome.requestIdentity,outcomeSHA256:WorkflowArchiveCodec.hash(outcomeBytes),
                     modelManifestSHA256:model,preparationManifestSHA256:preparation,checkpointSHA256:checkpoint.sha256)
    }
    static func stage(_ selected:WorkflowFrozenModelReference,document:WorkflowDocument,plan:WorkflowEvaluationPlan,root:URL)throws->Self {
        try stage(selected, document: document, plan: plan, root: root, stageID: UUID())
    }
    static func stage(_ selected:WorkflowFrozenModelReference,document:WorkflowDocument,plan:WorkflowEvaluationPlan,root:URL,stageID:UUID)throws->Self {
        try Task.checkCancellation()
        guard selected.hasValidPins,let directoryID=UUID(uuidString:selected.directoryID),
              UUID(uuidString:selected.projectID) == UUID(uuidString:document.projectID) else {
            throw WorkflowArchiveError("Select a verified completed model from this project.")
        }
        let history=WorkflowHistoryReference(projectID:selected.projectID,directoryID:directoryID)
        let (parent,outcomeBytes)=try verifiedParent(history,root:root)
        guard parent.outcome.runID == selected.runID,parent.outcome.requestIdentity == selected.requestIdentity,
              WorkflowArchiveCodec.hash(outcomeBytes) == selected.outcomeSHA256,
              parent.outcome.modelManifestSHA256 == selected.modelManifestSHA256,
              parent.outcome.preparationManifestSHA256 == selected.preparationManifestSHA256,
              parent.outcome.artifacts.first(where:{$0.path == "training/artifacts/checkpoint.pt"})?.sha256 == selected.checkpointSHA256 else {
            throw WorkflowArchiveError("The selected frozen model changed. Select its verified completed run again.")
        }
        try WorkflowEvaluationDeclaration.validate(document:document,plan:plan,archivedDataflow:parent.plan.dataflow)
        if plan.settings.partitionRole == "validation",plan.testEndpoint != parent.plan.validation {
            throw WorkflowArchiveError("Validation must use the exact archived Validation endpoint consumed by this frozen model.")
        }
        guard let capture=parent.outcome.capture else {throw WorkflowArchiveError("Frozen model capture is unavailable.")}
        let outcomeFile=WorkflowArchiveFile(path:"outcome.json",bytes:outcomeBytes.count,sha256:selected.outcomeSHA256)
        let files=([outcomeFile,capture,parent.capture.workflow,parent.capture.modelRegistry,parent.capture.dataRegistry,parent.capture.runtimeEvidence]
            + parent.outcome.artifacts.filter{$0.path.hasPrefix("preparation/artifacts/") || ["training/artifacts/manifest.json","training/artifacts/checkpoint.pt"].contains($0.path)})
            .sorted{$0.path<$1.path}
        guard (1...320).contains(files.count),Set(files.map(\.path)).count == files.count else {
            throw WorkflowArchiveError("Frozen evaluation inventory is incomplete or duplicated.")
        }
        var total=0
        for file in files {
            let limit=try fileLimit(file.path)
            guard file.bytes>=0,file.bytes<=limit,file.bytes<=340_000_000-total,WorkflowArchiveCodec.isHash(file.sha256) else {
                throw WorkflowArchiveError("Frozen evaluation snapshot exceeds its file or total byte budget.")
            }
            total += file.bytes
        }
        guard let preparation=files.first(where:{$0.path == "preparation/artifacts/manifest.json"}) else {
            throw WorkflowArchiveError("Archived protected test preparation is missing.")
        }
        let preparationBytes=try read(preparation,history:history,root:root)
        let inventory=try JSONDecoder().decode(PreparationInventory.self,from:preparationBytes)
        guard inventory.artifacts.contains(where:{$0.kind == "table" && $0.nodeID == plan.testEndpoint.node && $0.port == plan.testEndpoint.port}) else {
            throw WorkflowArchiveError("The selected test connection was not saved in this model's preparation. Connect an archived protected test output.")
        }
        guard let canonical = realpath(root.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let stageRoot = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        free(canonical)
        let fd = try WorkflowEvaluationOwnership.openDirectory(stageRoot)
        defer { Darwin.close(fd) }
        if mkdirat(fd, "evaluation-staging", 0o700) != 0, errno != EEXIST { throw WorkflowFileLeaseError.system(errno) }
        let namespace = stageRoot.appendingPathComponent("evaluation-staging")
        let stagingParent = openat(fd, "evaluation-staging", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard stagingParent >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(stagingParent) }
        let rootIdentity = try WorkflowEvaluationOwnership.directoryIdentity(fd)
        let namespaceIdentity = try WorkflowEvaluationOwnership.directoryIdentity(stagingParent)
        func checkParents() throws {
            let currentRoot = try WorkflowEvaluationOwnership.openDirectory(stageRoot)
            defer { Darwin.close(currentRoot) }
            let currentParent = try WorkflowEvaluationOwnership.openDirectory(namespace)
            defer { Darwin.close(currentParent) }
            guard try WorkflowEvaluationOwnership.directoryIdentity(currentRoot) == rootIdentity,
                  try WorkflowEvaluationOwnership.directoryIdentity(currentParent) == namespaceIdentity else { throw WorkflowFileLeaseError.unsafePath }
        }
        try checkParents()
        guard mkdirat(stagingParent, stageID.uuidString, 0o700) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        let directory = namespace.appendingPathComponent(stageID.uuidString)
        let stageOwnership = try WorkflowFrozenSnapshotDirectory(directory: directory)
        try checkParents()
        var retained=false
        defer {if !retained {stageOwnership.remove()}}
        for file in files {
            try Task.checkCancellation()
            let bytes=try read(file,history:history,root:root)
            try checkParents(); try stageOwnership.check()
            try DiagnosticFileIO.writeFrozenEvaluationArtifact(bytes,root:directory,path:file.path)
        }
        // Reject a changed parent identity even if individual earlier reads were valid.
        _ = try read(outcomeFile,history:history,root:root)
        _ = try read(capture,history:history,root:root)
        try Task.checkCancellation()
        let receipt=try WorkflowArchiveCodec.encode(Receipt(schemaVersion:plan.settings.partitionRole == "test" ? 1:2,reference:selected,testEndpoint:plan.testEndpoint,partitionRole:plan.settings.partitionRole == "test" ? nil:"validation",files:files))
        try checkParents(); try stageOwnership.check()
        try DiagnosticFileIO.writeImmutable(receipt,root:directory,parts:["snapshot.json"],limit:256_000)
        let result=Self(reference:selected,directory:directory,files:files,receiptSHA256:WorkflowArchiveCodec.hash(receipt),ownership:stageOwnership)
        try result.verify()
        retained=true
        return result
    }
    static func receiptRole(_ bytes:Data)throws->String {
        guard let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],
              case .integer(let version)=try JSONDecoder().decode([String:JSONValue].self,from:bytes)["schemaVersion"] else {throw WorkflowArchiveError("Invalid frozen receipt.")}
        if version == 1,Set(object.keys)==["schemaVersion","reference","testEndpoint","files"] {return "test"}
        if version == 2,Set(object.keys)==["schemaVersion","reference","testEndpoint","files","partitionRole"],object["partitionRole"] as? String == "validation" {return "validation"}
        throw WorkflowArchiveError("Frozen receipt role/version does not match its exact fields.")
    }
    /// Recheck staged bytes immediately before capture/launch, not mutable original paths.
    func verify()throws {
        try Task.checkCancellation()
        let receiptBytes=try DiagnosticFileIO.read(root:directory,parts:["snapshot.json"],limit:256_000)
        guard WorkflowArchiveCodec.hash(receiptBytes) == receiptSHA256 else {throw WorkflowArchiveError("Frozen snapshot receipt changed.")}
        _ = try Self.receiptRole(receiptBytes)
        let receipt=try JSONDecoder().decode(Receipt.self,from:receiptBytes)
        guard ((receipt.schemaVersion == 1 && receipt.partitionRole == nil) || (receipt.schemaVersion == 2 && receipt.partitionRole == "validation")),receipt.reference == reference,receipt.files == files else {throw WorkflowArchiveError("Frozen snapshot identity changed.")}
        for file in files {
            try Task.checkCancellation()
            let bytes=try DiagnosticFileIO.read(root:directory,parts:WorkflowArchiveCodec.pathParts(file.path),limit:Self.fileLimit(file.path))
            guard bytes.count == file.bytes,WorkflowArchiveCodec.hash(bytes) == file.sha256 else {throw WorkflowArchiveError("Frozen snapshot artifact changed: "+file.path)}
        }
    }
    private static func verifiedParent(_ history:WorkflowHistoryReference,root:URL)throws->(VerifiedWorkflowOutcome,Data) {
        try Task.checkCancellation()
        guard UUID(uuidString:history.projectID) != nil,case .verified(let value)=WorkflowHistoryReader.load(history,root:root),
              value.outcome.status == .completed,value.modelContent != nil else {
            throw WorkflowArchiveError("Select a completed run with independently verified model and preparation artifacts.")
        }
        let bytes=try DiagnosticFileIO.read(root:root,parts:parts(history)+["outcome.json"],limit:256_000)
        guard try WorkflowArchiveCodec.decodeOutcome(bytes) == value.outcome else {throw WorkflowArchiveError("Parent outcome changed during verification.")}
        try Task.checkCancellation()
        return(value,bytes)
    }
    private static func parts(_ history:WorkflowHistoryReference)->[String] { ["workflow-runs",history.projectID,history.directoryID.uuidString] }
    private static func read(_ file:WorkflowArchiveFile,history:WorkflowHistoryReference,root:URL)throws->Data {
        let bytes=try DiagnosticFileIO.read(root:root,parts:parts(history)+WorkflowArchiveCodec.pathParts(file.path),limit:fileLimit(file.path))
        guard bytes.count == file.bytes,WorkflowArchiveCodec.hash(bytes) == file.sha256 else {throw WorkflowArchiveError("Archived frozen model evidence changed: "+file.path)}
        return bytes
    }
}
