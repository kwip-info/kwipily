import Foundation

public struct OptimizerFormativeFixture:Equatable,Identifiable {
    public let id:String
    public let initial:LearningTensor
    public let gradients:LearningTensor
    public let active:LearningTensor
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public struct OptimizerFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[OptimizerFormativeFixture]
}
public extension TensorLearningReceipt {
    func optimizerFormativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->OptimizerFormativeProjection? {
        guard OptimizerLearningTask(rawValue:operation) != nil,currentMode != .assessment,attempt.mode != .assessment,currentSubmission == submissionData,let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        var fixtures:[OptimizerFormativeFixture]=[]
        for fixture in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID == fixture.id}),let initial=fixture.inputs["initial"],let gradients=fixture.inputs["gradients"],let active=fixture.inputs["active"] else {return nil}
            fixtures.append(.init(id:fixture.id,initial:initial,gradients:gradients,active:active,measured:measured.tensors,expected:fixture.expected))
        }
        return .init(submissionDigest:submissionSHA256,fixtures:fixtures)
    }
}
