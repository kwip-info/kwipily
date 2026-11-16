import Foundation

public enum LearningAssessor {
    public static func evaluate(_ plan:LearningAssessmentPlan,evidence:LearningEvidence)throws->LearningGrade {
        try plan.validate()
        func grade(status:LearningGradeStatus,summary:String,outcomes:[LearningCriterionOutcome]=[])->LearningGrade {
            LearningGrade(status:status,summary:summary,outcomes:outcomes,assessmentID:plan.id,objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest)
        }
        if evidence.cancelled {return grade(status:.cancelled,summary:"Assessment cancelled; no correctness judgment recorded.")}
        if let failure=evidence.backendError {return grade(status:.executionFailed,summary:"Execution failed: "+String(failure.prefix(500)))}
        guard evidence.runtime==plan.runtime else {return grade(status:.unavailable,summary:"Required backend/capability versions are unavailable.")}
        let largestMeasured=evidence.fixtures.flatMap{Array($0.tensors.values)}.map{$0.values.count}.max() ?? 0
        guard evidence.elapsedMilliseconds.isFinite,evidence.elapsedMilliseconds>=0,evidence.maximumTensorElements>=largestMeasured else {return grade(status:.incomplete,summary:"Resource evidence is invalid.")}
        if evidence.elapsedMilliseconds>plan.budget.milliseconds || evidence.maximumTensorElements>plan.budget.maximumTensorElements {return grade(status:.budgetExceeded,summary:"Declared assessment compute budget exceeded; retry within the stated limits.")}
        let ids=evidence.fixtures.map(\.fixtureID)
        guard evidence.assessmentID==plan.id,evidence.assessmentVersion==plan.assessmentVersion,evidence.fixtureDigest==plan.fixtureDigest,evidence.submissionDigest.count==64,evidence.submissionDigest.utf8.allSatisfy({(48...57).contains($0)||(97...102).contains($0)}),ids.count==plan.fixtures.count,Set(ids).count==ids.count,Set(ids)==Set(plan.fixtures.map(\.id)) else {return grade(status:.incomplete,summary:"Evidence must match the assessment version and every independent fixture.")}
        let observed=Dictionary(uniqueKeysWithValues:evidence.fixtures.map{($0.fixtureID,$0)})
        var outcomes:[LearningCriterionOutcome]=[]
        for criterion in plan.criteria {
            var passed=true,maximumError=0.0,checked=0,detail=""
            for fixture in plan.fixtures {
                let actual=observed[fixture.id]!
                switch criterion.rule {
                case .tensor(let key,let absolute,let relative):
                    guard let tensor=actual.tensors[key] else{return grade(status:.incomplete,summary:"Missing measured output for criterion \(criterion.id).")}
                    let expected=fixture.expected[key]!
                    if !tensor.wellFormed || tensor.shape != expected.shape {passed=false;detail="A measured tensor had the wrong shape or nonfinite values.";continue}
                    for (a,b) in zip(tensor.values,expected.values) {let difference=abs(a-b);maximumError=max(maximumError,difference);checked+=1;if difference>absolute+relative*abs(b){passed=false}}
                case .trainingOnly(let training,let heldOut,let fit):
                    guard let train=actual.partitions[training],let validation=actual.partitions[heldOut],let fitted=actual.partitions[fit] else{return grade(status:.incomplete,summary:"Missing partition lineage for criterion \(criterion.id).")}
                    let lists=[train,validation,fitted]
                    if lists.contains(where:{$0.isEmpty || $0.count>100_000 || Set($0).count != $0.count || $0.contains(where:{$0.isEmpty || $0.count>200})}) || !Set(train).isDisjoint(with:Set(validation)) || !Set(fitted).isSubset(of:Set(train)) {passed=false}
                    checked+=1;detail="Checked \(checked) partition fixtures for disjoint holdout rows and training-only fit membership."
                }
            }
            if detail.isEmpty {detail="Compared \(checked) measured values; maximum absolute difference \(String(format:"%.6g",maximumError))."}
            outcomes.append(LearningCriterionOutcome(criterionID:criterion.id,passed:passed,evidence:detail,nextStep:passed ? "Criterion met on these fixtures.":criterion.nextStep))
        }
        let passed=outcomes.allSatisfy(\.passed)
        return grade(status:passed ? .passed:.failed,summary:passed ? "Criteria met on the independent fixtures; broader competence is not inferred.":"Some criteria need another attempt. Review the specific evidence and next step.",outcomes:outcomes)
    }
}
