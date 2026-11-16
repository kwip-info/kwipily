import Foundation

/// Private, hash-verified lifecycle files from one completed universal History entry.
/// The manifest includes local training provenance. Never expose this as a portable export.
struct WorkflowCheckpointSnapshot {
    static let artifactLimits=["manifest.json":20_000_000,"checkpoint.pt":100_000_000,"metrics.json":4_000_000]
    let reference: WorkflowHistoryReference
    let runID: String
    let requestIdentity: String
    let manifestSHA256: String
    let directory: URL
    let files:[WorkflowArchiveFile]
    var storageLifetime: WorkflowExportStorageLifetime? = nil
    var dataLoadingIdentity:WorkflowDataLoadingIdentity? = nil

    /// Only snapshots staged by the native writer carry cleanup ownership.
    func remove() { storageLifetime?.remove() }

    /// Re-read every pinned byte before archiving. The owning capture is committed
    /// only after these immutable files; a partial write never means a runnable capture.
    func archive(to root:URL)throws->[WorkflowArchiveFile] {
        let contents=try verifiedContents()
        try Task.checkCancellation()
        try DiagnosticFileIO.createExclusiveDirectory(root:root,name:"continuation-parent")
        var result:[WorkflowArchiveFile]=[]
        for file in files.sorted(by:{$0.path<$1.path}) {
            try Task.checkCancellation()
            try DiagnosticFileIO.writeCheckpointArtifact(contents[file.path]!,root:root,name:file.path)
            result.append(.init(path:"continuation-parent/"+file.path,bytes:file.bytes,sha256:file.sha256))
        }
        return result
    }

    func parentRecord()throws->WorkflowLifecycleParentRecord {
        try WorkflowLifecycleParentRecord.decode(verifiedContents(),dataLoading:dataLoadingIdentity)
    }

    private func verifiedContents()throws->[String:Data] {
        try Task.checkCancellation()
        let limits=try WorkflowCheckpointInventory.structuralLimits(names:Set(files.map(\.path)))
        guard files.count == limits.count else {
            throw WorkflowArchiveError("Checkpoint snapshot inventory is incomplete.")
        }
        var contents:[String:Data]=[:]
        for file in files {
            try Task.checkCancellation()
            let bytes=try DiagnosticFileIO.read(root:directory,parts:[file.path],limit:limits[file.path]!)
            guard bytes.count == file.bytes,WorkflowArchiveCodec.hash(bytes) == file.sha256,
                  file.path != "manifest.json" || file.sha256 == manifestSHA256 else {
                throw WorkflowArchiveError("Checkpoint snapshot changed before archive capture.")
            }
            contents[file.path]=bytes
        }
        try Task.checkCancellation()
        try WorkflowCheckpointInventory.verify(contents)
        return contents
    }
}
