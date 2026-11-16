import Foundation

public struct SchedulerFormativeFixture:Equatable,Identifiable {
    public let id:String
    public let initial:LearningTensor
    public let gradients:LearningTensor
    public let active:LearningTensor
    public let epochOfBatch:LearningTensor
    public let metrics:LearningTensor
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public struct SchedulerFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[SchedulerFormativeFixture]
}
public extension TensorLearningReceipt {
    func schedulerFormativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->SchedulerFormativeProjection? {
        guard SchedulerLearningTask(rawValue:operation) != nil,currentMode != .assessment,attempt.mode != .assessment,currentSubmission == submissionData,let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        var fixtures:[SchedulerFormativeFixture]=[]
        for f in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID == f.id}),let initial=f.inputs["initial"],let gradients=f.inputs["gradients"],let active=f.inputs["active"],let epoch=f.inputs["epochOfBatch"],let metrics=f.inputs["metrics"] else {return nil}
            fixtures.append(.init(id:f.id,initial:initial,gradients:gradients,active:active,epochOfBatch:epoch,metrics:metrics,measured:measured.tensors,expected:f.expected))
        }
        return .init(submissionDigest:submissionSHA256,fixtures:fixtures)
    }
}
