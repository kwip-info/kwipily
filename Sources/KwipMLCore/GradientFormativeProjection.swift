import Foundation

/// Admission for current measured difference tables; this is not a runtime receipt verifier.
public struct GradientFormativeProjection:Equatable {
    public let submissionDigest:String
    public let fixtures:[LearningFixtureEvidence]
    public static func project(plan:LearningAssessmentPlan,evidence:LearningEvidence,currentSubmission:Data,mode:LearningMode,attemptMode:LearningMode)->Self? {
        guard mode != .assessment,attemptMode != .assessment,mode==attemptMode,let task=GradientLearningTask.allCases.first(where:{plan.id=="assessment."+$0.rawValue}),plan.objectiveID=="gradient."+task.rawValue,(try? GradientLearningAssessment.validate(plan)) != nil,(try? GradientLearningSubmission.decode(currentSubmission,task:task)) != nil,evidence.assessmentID==plan.id,evidence.assessmentVersion==plan.assessmentVersion,evidence.fixtureDigest==plan.fixtureDigest,evidence.runtime==plan.runtime,evidence.submissionDigest==learningHash(currentSubmission),!evidence.cancelled,evidence.backendError==nil,evidence.elapsedMilliseconds.isFinite,(0...plan.budget.milliseconds).contains(evidence.elapsedMilliseconds),evidence.maximumTensorElements<=plan.budget.maximumTensorElements,evidence.maximumTensorElements>=0,evidence.fixtures.count==plan.fixtures.count,Set(evidence.fixtures.map(\.fixtureID)).count==evidence.fixtures.count else {return nil}
        for fixture in plan.fixtures {
            guard let measured=evidence.fixtures.first(where:{$0.fixtureID==fixture.id}),Set(measured.tensors.keys)==Set(fixture.expected.keys),measured.partitions.isEmpty,measured.tensors.allSatisfy({$0.value.wellFormed && $0.value.shape==fixture.expected[$0.key]?.shape}) else {return nil}
        }
        return .init(submissionDigest:evidence.submissionDigest,fixtures:plan.fixtures.compactMap {fixture in evidence.fixtures.first {$0.fixtureID==fixture.id}})
    }
}
