import Foundation

public struct LearningAttempt:Codable,Equatable,Identifiable {
    public var id:String;public var objectiveID:String;public var contentVersion:ContractVersion;public var assessmentVersion:ContractVersion;public var mode:LearningMode;public var hintIDs:[String];public var grade:LearningGrade;public var timestamp:Date
    public init(id:String=UUID().uuidString,objectiveID:String,contentVersion:ContractVersion,assessmentVersion:ContractVersion,mode:LearningMode,hintIDs:[String]=[],grade:LearningGrade,timestamp:Date=Date()){self.id=id;self.objectiveID=objectiveID;self.contentVersion=contentVersion;self.assessmentVersion=assessmentVersion;self.mode=mode;self.hintIDs=hintIDs;self.grade=grade;self.timestamp=timestamp}
}
public struct LearningProgressResetPlan:Equatable {
    public let objectiveID:String;public let attemptIDs:Set<String>;public let revision:String
}
public final class LearningProgressStore {
    private struct Archive:Codable {var schemaVersion=1;var attempts:[LearningAttempt]}
    public let directory:URL
    private var url:URL {directory.appendingPathComponent("progress.json")}
    public init(directory:URL){self.directory=directory}
    public func attempts()throws->[LearningAttempt] {try archive().attempts}
    public func append(_ attempt:LearningAttempt)throws {
        var value=try archive();guard value.attempts.count<5000,!value.attempts.contains(where:{$0.id==attempt.id}) else {throw LearningContractError("Progress is full or attempt identity already exists.")}
        try validate(attempt);value.attempts.append(attempt);try write(value)
    }
    public func hasCurrentCompletion(for plan:LearningAssessmentPlan)throws->Bool {
        try attempts().contains{$0.objectiveID==plan.objectiveID && $0.contentVersion==plan.contentVersion && $0.assessmentVersion==plan.assessmentVersion && $0.mode == .assessment && $0.grade.status == .passed && $0.grade.assessmentID==plan.id}
    }
    public func previewReset(objectiveID:String)throws->LearningProgressResetPlan {
        let value=try archive();return LearningProgressResetPlan(objectiveID:objectiveID,attemptIDs:Set(value.attempts.filter{$0.objectiveID==objectiveID}.map(\.id)),revision:try revision(value))
    }
    public func applyReset(_ plan:LearningProgressResetPlan)throws {
        var value=try archive();guard try revision(value)==plan.revision,Set(value.attempts.filter{$0.objectiveID==plan.objectiveID}.map(\.id))==plan.attemptIDs else {throw LearningContractError("Progress changed since reset preview. Review the updated affected attempts.")}
        value.attempts.removeAll{plan.attemptIDs.contains($0.id)};try write(value)
    }
    private func validate(_ attempt:LearningAttempt)throws {
        if attempt.mode == .assessment && attempt.grade.status == .passed {
            guard !attempt.grade.outcomes.isEmpty,attempt.grade.outcomes.allSatisfy(\.passed),attempt.grade.objectiveID==attempt.objectiveID,attempt.grade.contentVersion==attempt.contentVersion,attempt.grade.assessmentVersion==attempt.assessmentVersion,let id=attempt.grade.assessmentID,!id.isEmpty,let digest=attempt.grade.fixtureDigest,digest.count==64 else {throw LearningContractError("Completion requires criterion outcomes bound to this objective and assessment version.")}
        }
        guard UUID(uuidString:attempt.id) != nil,!attempt.objectiveID.isEmpty,attempt.objectiveID.count<=128,attempt.hintIDs.count<=100,attempt.hintIDs.allSatisfy({!$0.isEmpty && $0.count<=128}),attempt.timestamp.timeIntervalSince1970.isFinite,attempt.grade.outcomes.count<=32,attempt.grade.summary.count<=2000,attempt.grade.outcomes.allSatisfy({$0.criterionID.count<=128 && $0.evidence.count<=2000 && $0.nextStep.count<=2000}) else {throw LearningContractError("Invalid or unbounded attempt record.")}
    }
    private func checkPath()throws {for path in [directory,url] {if (try? path.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink)==true {throw LearningContractError("Learning progress must not use symbolic links.")}}}
    private func archive()throws->Archive {
        try checkPath();guard FileManager.default.fileExists(atPath:url.path) else{return Archive(attempts:[])}
        let info=try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey]);guard info.isRegularFile==true,(info.fileSize ?? Int.max)<=5_000_000 else {throw LearningContractError("Progress file exceeds bounds.")}
        let handle=try FileHandle(forReadingFrom:url);defer{try? handle.close()};let data=try handle.read(upToCount:5_000_001) ?? Data();guard data.count<=5_000_000 else {throw LearningContractError("Progress file exceeded bounds while reading.")}
        let value=try JSONDecoder().decode(Archive.self,from:data);guard value.schemaVersion==1,value.attempts.count<=5000,Set(value.attempts.map(\.id)).count==value.attempts.count else {throw LearningContractError("Unsupported or duplicate progress records.")};for attempt in value.attempts {try validate(attempt)};return value
    }
    private func revision(_ value:Archive)throws->String {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];return learningHash(try encoder.encode(value))}
    private func write(_ value:Archive)throws {try checkPath();let data=try JSONEncoder().encode(value);guard data.count<=5_000_000 else {throw LearningContractError("Progress file exceeds bounds.")};try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true);try data.write(to:url,options:.atomic)}
}
