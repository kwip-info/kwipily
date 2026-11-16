import Foundation

/// Single-owner local receipt storage. Atomic publication follows immutable attempt creation;
/// interrupted publication may leave an unreferenced attempt, never a partial latest receipt.
/// This does not claim cross-process locking or hostile-filesystem race protection.
public struct TensorLearningResetPlan: Equatable {
    public let projectID: String
    public let objectiveID: String
    public let attemptID: String
    fileprivate let pointerDigest: String
    fileprivate let receiptDigest: String
}

public final class TensorLearningReceiptRepository {
    private struct Pointer:Codable {let schemaVersion:Int;let attemptID:String}
    public let directory:URL
    private let lock=NSLock()
    public init(directory:URL) {self.directory=directory}
    public func save(_ receipt:TensorLearningReceipt)throws {
        lock.lock();defer {lock.unlock()}
        let data=try receipt.encoded()
        let folder=try location(receipt.projectID,create:true)
        let file=try attemptURL(folder,receipt.attempt.id)
        if exists(file) {
            guard try read(file,maximum:2_000_000) == data else {throw LearningContractError("A different receipt already owns this attempt identity.")}
        } else {try data.write(to:file,options:.atomic)}
        let pointer=folder.appendingPathComponent("latest.json")
        if exists(pointer) {_ = try read(pointer,maximum:1024)}
        let bytes=try JSONEncoder().encode(Pointer(schemaVersion:1,attemptID:receipt.attempt.id))
        try bytes.write(to:pointer,options:.atomic)
    }
    public func load(projectID:String,attemptID:String)throws->TensorLearningReceipt {
        lock.lock();defer {lock.unlock()}
        return try loadUnlocked(projectID:projectID,attemptID:attemptID)
    }
    public func latest(projectID:String)throws->TensorLearningReceipt? {
        lock.lock();defer {lock.unlock()}
        let pointer=try location(projectID,create:false).appendingPathComponent("latest.json")
        guard exists(pointer) else {return nil}
        let bytes=try read(pointer,maximum:1024)
        guard case .object(let raw)=try JSONDecoder().decode(JSONValue.self,from:bytes),Set(raw.keys)==["schemaVersion","attemptID"] else {
            throw LearningContractError("Invalid latest tensor receipt pointer.")
        }
        let value=try JSONDecoder().decode(Pointer.self,from:bytes)
        guard value.schemaVersion == 1 else {throw LearningContractError("Unsupported tensor receipt pointer version.")}
        return try loadUnlocked(projectID:projectID,attemptID:value.attemptID)
    }
    /// Reset completion only. Immutable attempt evidence and editable drafts are retained.
    public func previewReset(projectID: String) throws -> TensorLearningResetPlan? {
        lock.lock(); defer { lock.unlock() }
        return try resetPlanUnlocked(projectID: projectID)
    }
    public func applyReset(_ plan: TensorLearningResetPlan) throws {
        lock.lock(); defer { lock.unlock() }
        guard try resetPlanUnlocked(projectID: plan.projectID) == plan else {
            throw LearningContractError("Saved check changed since reset preview. Refresh and review it again.")
        }
        try FileManager.default.removeItem(at: location(plan.projectID, create: false).appendingPathComponent("latest.json"))
    }
    private func resetPlanUnlocked(projectID: String) throws -> TensorLearningResetPlan? {
        let folder = try location(projectID, create: false)
        let pointer = folder.appendingPathComponent("latest.json")
        guard exists(pointer) else { return nil }
        let bytes = try read(pointer, maximum: 1024)
        guard case .object(let raw) = try JSONDecoder().decode(JSONValue.self, from: bytes),
              Set(raw.keys) == ["schemaVersion", "attemptID"] else {
            throw LearningContractError("Invalid latest tensor receipt pointer.")
        }
        let value = try JSONDecoder().decode(Pointer.self, from: bytes)
        guard value.schemaVersion == 1 else { throw LearningContractError("Unsupported tensor receipt pointer version.") }
        let receipt = try loadUnlocked(projectID: projectID, attemptID: value.attemptID)
        return TensorLearningResetPlan(projectID: projectID, objectiveID: receipt.attempt.objectiveID,
            attemptID: receipt.attempt.id, pointerDigest: learningHash(bytes),
            receiptDigest: learningHash(try read(attemptURL(folder, value.attemptID), maximum: 2_000_000)))
    }
    private func loadUnlocked(projectID:String,attemptID:String)throws->TensorLearningReceipt {
        let file=try attemptURL(location(projectID,create:false),attemptID)
        let receipt=try TensorLearningReceipt.decode(read(file,maximum:2_000_000))
        guard receipt.attempt.id == attemptID else {throw LearningContractError("Tensor receipt belongs to a different attempt.")}
        try receipt.verifiedReplay(expectedProjectID:projectID)
        return receipt
    }
    private func attemptURL(_ folder:URL,_ id:String)throws->URL {
        guard UUID(uuidString:id) != nil else {throw LearningContractError("Tensor attempt identity must be a UUID.")}
        return folder.appendingPathComponent(id+".json")
    }
    private func location(_ id:String,create:Bool)throws->URL {
        guard UUID(uuidString:id) != nil else {throw LearningContractError("Tensor project identity must be a UUID.")}
        let project=directory.appendingPathComponent(id),folder=project.appendingPathComponent("tensor-receipts")
        for path in [directory,project,folder] {
            if exists(path) {
                guard try FileManager.default.attributesOfItem(atPath:path.path)[.type] as? FileAttributeType == .typeDirectory else {
                    throw LearningContractError("Tensor receipt storage must use real directories.")
                }
            } else if create {try FileManager.default.createDirectory(at:path,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])}
        }
        return folder
    }
    private func exists(_ url:URL)->Bool {(try? FileManager.default.attributesOfItem(atPath:url.path)) != nil}
    private func read(_ url:URL,maximum:Int)throws->Data {
        let attributes=try FileManager.default.attributesOfItem(atPath:url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,let size=attributes[.size] as? NSNumber,size.intValue<=maximum else {
            throw LearningContractError("Tensor receipt storage requires bounded regular files.")
        }
        let handle=try FileHandle(forReadingFrom:url);defer {try? handle.close()}
        let data=try handle.read(upToCount:maximum+1) ?? Data()
        guard data.count<=maximum else {throw LearningContractError("Tensor receipt file exceeded its bound while reading.")}
        return data
    }
}
