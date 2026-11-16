import Foundation

public struct TensorLearningDraftSnapshot:Equatable {
    public let draft:TensorLearningDraft
    public let revision:String
}

/// One owner per storage root. Revision checks reject stale sequential writes;
/// this is not a cross-process locking or hostile-filesystem guarantee.
public final class TensorLearningDraftRepository {
    public let directory:URL
    private let lock=NSLock()
    public init(directory:URL) {self.directory=directory}
    public func load(projectID:String)throws->TensorLearningDraftSnapshot {
        lock.lock();defer {lock.unlock()}
        let file=try location(projectID,create:false)
        return try read(file,projectID:projectID)
    }
    @discardableResult public func save(_ draft:TensorLearningDraft,expectedRevision:String?)throws->String {
        lock.lock();defer {lock.unlock()}
        let bytes=try draft.encoded()
        let file=try location(draft.projectID,create:true)
        let existing=(try? FileManager.default.attributesOfItem(atPath:file.path)) != nil
        if existing {
            let current=try read(file,projectID:draft.projectID)
            guard expectedRevision == current.revision else {throw LearningContractError("Lesson draft changed since it was opened. Reload before saving.")}
        } else if expectedRevision != nil {throw LearningContractError("The saved lesson draft is missing. Reopen before saving.")}
        try bytes.write(to:file,options:.atomic)
        return learningHash(bytes)
    }
    private func location(_ id:String,create:Bool)throws->URL {
        guard UUID(uuidString:id) != nil else {throw LearningContractError("Lesson storage requires a UUID project identity.")}
        let project=directory.appendingPathComponent(id,isDirectory:true)
        if create {try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)}
        if let info=try? FileManager.default.attributesOfItem(atPath:project.path) {
            guard info[.type] as? FileAttributeType == .typeDirectory else {throw LearningContractError("Lesson project storage must be a directory, not a symbolic link or file.")}
        } else if create {try FileManager.default.createDirectory(at:project,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])}
        return project.appendingPathComponent("tensor.json")
    }
    private func read(_ file:URL,projectID:String)throws->TensorLearningDraftSnapshot {
        let info=try FileManager.default.attributesOfItem(atPath:file.path)
        guard info[.type] as? FileAttributeType == .typeRegular,let size=info[.size] as? NSNumber,size.intValue<=2_000_000 else {
            throw LearningContractError("Lesson draft must be a bounded regular file.")
        }
        let handle=try FileHandle(forReadingFrom:file);defer {try? handle.close()}
        let bytes=try handle.read(upToCount:2_000_001) ?? Data()
        let draft=try TensorLearningDraft.decode(bytes)
        guard draft.projectID == projectID else {throw LearningContractError("Lesson draft belongs to a different project.")}
        return .init(draft:draft,revision:learningHash(bytes))
    }
}
