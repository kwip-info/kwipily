import Foundation

public struct CheckpointFormativeFixture:Identifiable,Equatable {
    public let id:String
    public let inputs:[String:LearningTensor]
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public struct CheckpointFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[CheckpointFormativeFixture]
}
public extension TensorLearningReceipt {
    func checkpointFormativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->CheckpointFormativeProjection? {
        guard CheckpointLearningTask(rawValue:operation) != nil,currentMode != .assessment,attempt.mode != .assessment,currentSubmission == submissionData,let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        var result:[CheckpointFormativeFixture]=[]
        for fixture in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID==fixture.id}),Set(fixture.expected.keys).isSubset(of:Set(measured.tensors.keys)) else {return nil}
            result.append(.init(id:fixture.id,inputs:fixture.inputs,measured:measured.tensors,expected:fixture.expected))
        }
        return .init(submissionDigest:submissionSHA256,fixtures:result)
    }
}
