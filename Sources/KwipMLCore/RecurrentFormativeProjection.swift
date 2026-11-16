import Foundation
public struct RecurrentFormativeFixture:Equatable,Identifiable {
    public let id:String
    public let inputs:[String:LearningTensor]
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public struct RecurrentFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[RecurrentFormativeFixture]
}
public extension TensorLearningReceipt {
    func recurrentFormativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->RecurrentFormativeProjection? {
        guard RecurrentLearningTask(rawValue:operation) != nil,currentMode != .assessment,attempt.mode != .assessment,currentSubmission==submissionData,let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        var fixtures:[RecurrentFormativeFixture]=[]
        for f in plan.fixtures {guard let measured=evidence.fixtures.first(where:{$0.fixtureID==f.id}) else {return nil};fixtures.append(.init(id:f.id,inputs:f.inputs,measured:measured.tensors,expected:f.expected))}
        return .init(submissionDigest:submissionSHA256,fixtures:fixtures)
    }
}
