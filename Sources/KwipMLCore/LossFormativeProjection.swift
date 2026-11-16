import Foundation
public struct LossFormativeFixture:Equatable,Identifiable {
    public let id:String
    public let inputs:[String:LearningTensor]
    public let measured:[String:LearningTensor]
    public let expected:[String:LearningTensor]
}
public enum LossLearningAssessment {
    /// Reviewed coverage modes; these names alone never establish accepted execution evidence.
    public static let modes=["training-objective","evaluation-objective","target-contract","reduction","input-gradients"]
    public static func plan(task:LossLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        let fixtures=try task.fixtures(seed:Int(truncatingIfNeeded:seed)).map {fixture->LearningFixture in
            let result=try expected(task:task,fixture:fixture)
            var format=fixture.inputs["prediction"]!.shape.map(Double.init)
            if task == .triplet {format += [0,1,2]}
            for role in task.sourceRoles where role != "prediction" {
                let tensor=fixture.inputs[role]!
                let integerRole=(task == .crossEntropy || task == .nll || task == .ctc) && role != "prediction"
                format += [integerRole ? 1:0,Double(tensor.shape.count)] + tensor.shape.map(Double.init) + tensor.values
            }
            return .init(id:fixture.id,inputs:fixture.inputs,expected:["trainingObjective":result["objective"]!,"evaluationObjective":result["objective"]!,"targetContract":.init(shape:[format.count],values:format),"reduction":result["raw"]!,"inputInfluence":result["inputInfluence"]!])
        }
        let descriptions=["Use the configured objective to produce the combined training score.","Apply the same fixed objective to evaluate the supplied held-out roles.","Preserve each observation, desired format and explicitly ordered role.","Keep individual scores before the declared combination.","Measure how a small input change affects the combined score."]
        let keys=["trainingObjective","evaluationObjective","targetContract","reduction","inputInfluence"]
        let criteria=zip(modes.indices,modes).map {index,mode in LearningCriterion(id:mode,description:descriptions[index],nextStep:task.instructions,rule:.tensor(key:keys[index],absoluteTolerance:index == 2 ? 0:1e-6,relativeTolerance:index == 2 ? 0:1e-5))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"loss."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
    public static func expected(task:LossLearningTask,fixture:LossLessonFixture)throws->[String:LearningTensor] {
        let reference=try LossLessonReference(task:task,fixture:fixture)
        return ["raw":reference.raw,"objective":reference.objective,"inputInfluence":reference.inputInfluence]
    }
}

public struct LossFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[LossFormativeFixture]
}
public extension TensorLearningReceipt {
    func lossFormativeProjection(expectedProjectID:String,currentSubmission:Data,currentMode:LearningMode)->LossFormativeProjection? {
        guard LossLearningTask(rawValue:operation) != nil,currentMode != .assessment,attempt.mode != .assessment,currentSubmission == submissionData,let plan=try? verifiedReplay(expectedProjectID:expectedProjectID) else {return nil}
        var fixtures:[LossFormativeFixture]=[]
        for f in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID==f.id}) else {return nil}
            fixtures.append(.init(id:f.id,inputs:f.inputs,measured:measured.tensors,expected:f.expected))
        }
        return .init(submissionDigest:submissionSHA256,fixtures:fixtures)
    }
}
