import XCTest
@testable import KwipMLCore
final class LearningAssessmentTests:XCTestCase {
    private func receipt(_ plan:LearningAssessmentPlan)->LearningEvidence {
        LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:plan.fixtures.map{LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:10,maximumTensorElements:16)
    }
    func testEquivalentAlternativeOutputsPassWithoutNodeIdentityRequirements()throws {
        let plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:42)
        var evidence=receipt(plan)
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.passed)
        evidence.submissionDigest=String(repeating:"b",count:64)
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.passed)
        XCTAssertGreaterThan(Set(plan.fixtures.flatMap{$0.expected["output"]!.values}).count,4)
    }
    func testConstantOutputCheatWrongShapeAndNonfiniteFail()throws {
        let plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:99)
        var evidence=receipt(plan)
        for i in evidence.fixtures.indices {evidence.fixtures[i].tensors["output"]=LearningTensor(shape:[2,2],values:[0,0,0,0])}
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.failed)
        evidence=receipt(plan);evidence.fixtures[0].tensors["output"]!.shape=[4]
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.failed)
        evidence=receipt(plan);evidence.fixtures[0].tensors["output"]!.values[0] = .nan
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.failed)
    }
    func testMissingWrongFixtureUnsupportedRuntimeAndCancellationAreNotConceptualFailures()throws {
        let plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:42)
        var evidence=receipt(plan);evidence.fixtures.removeLast()
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.incomplete)
        evidence=receipt(plan);evidence.fixtureDigest="wrong"
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.incomplete)
        evidence=receipt(plan);evidence.runtime.version="other"
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.unavailable)
        evidence=receipt(plan);evidence.cancelled=true
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.cancelled)
        evidence=receipt(plan);evidence.backendError="Adapter failed"
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.executionFailed)
        evidence=receipt(plan);evidence.elapsedMilliseconds=plan.budget.milliseconds+1
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.budgetExceeded)
    }
    func testResourceEvidenceCannotUnderstateMeasuredTensorAndInvalidVersionRejected()throws {
        var plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:42)
        var evidence=receipt(plan);evidence.maximumTensorElements=0
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.incomplete)
        plan.assessmentVersion=ContractVersion(-1)
        XCTAssertThrowsError(try LearningAssessor.evaluate(plan,evidence:receipt(plan)))
    }
    func testToleranceBoundaryAndMalformedPlanRejected()throws {
        var plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:42)
        var evidence=receipt(plan);evidence.fixtures[0].tensors["output"]!.values[0]+=1e-7
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.passed)
        plan.criteria.append(plan.criteria[0]);XCTAssertThrowsError(try LearningAssessor.evaluate(plan,evidence:evidence))
    }
    func testTrainingOnlyNormalizationAcceptsStateAndRejectsLeakage()throws {
        let plan=LearningExamples.trainingOnlyStandardization(runtimeVersion:"2.3.4",seed:7)
        var evidence=receipt(plan)
        for i in evidence.fixtures.indices {
            evidence.fixtures[i].partitions=["training":["t1","t2","t3"],"validation":["v1"],"fit":["t1","t2","t3"]]
        }
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.passed)
        evidence.fixtures[0].partitions["fit"]!.append("v1")
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.failed)
    }
    func testPublicTaskOmitsPrivateAnswersAndReferenceCompositionIsIndependent()throws {
        let plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:123)
        var evidence=receipt(plan)
        for (index,fixture) in plan.fixtures.enumerated() {
            let input=fixture.inputs["x"]!.values
            var result:[Double]=[]
            for row in 0..<2 {
                let x=input[row*2],y=input[row*2+1],h0=2*x-y+1,h1=0.5*x+3*y-2
                result += [h0+2*h1+0.25,-2*h0+h1+1]
            }
            evidence.fixtures[index].tensors["output"]=LearningTensor(shape:[2,2],values:result)
        }
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.passed)
        let task=plan.publicTask(mode:.assessment)
        let encoded=String(decoding:try JSONEncoder().encode(task),as:UTF8.self)
        XCTAssertFalse(encoded.contains("fixtureSeed"));XCTAssertFalse(encoded.contains("expected"))
        XCTAssertEqual(task.mode,.assessment);XCTAssertEqual(task.criteria.count,1)
    }
    func testEmptyPassCannotAwardProgressAndAssessmentIdentityMustMatch()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let store=LearningProgressStore(directory:dir),plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:42)
        let empty=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:.assessment,grade:LearningGrade(status:.passed,summary:"Next clicked"))
        XCTAssertThrowsError(try store.append(empty))
        let grade=try LearningAssessor.evaluate(plan,evidence:receipt(plan))
        try store.append(LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:.assessment,grade:grade))
        var other=plan;other.id="another-assessment-for-same-objective"
        XCTAssertFalse(try store.hasCurrentCompletion(for:other))
    }
    func testPrerequisiteDAGAndVersionedProgressResetScope()throws {
        let objectives=LearningExamples.objectives
        XCTAssertNoThrow(try LearningCurriculum.validate(objectives))
        var broken=objectives;broken[0].prerequisiteIDs=[broken.last!.id]
        XCTAssertThrowsError(try LearningCurriculum.validate(broken))
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let store=LearningProgressStore(directory:dir),plan=LearningExamples.affineComposition(runtimeVersion:"2.14.1",seed:42)
        let grade=try LearningAssessor.evaluate(plan,evidence:receipt(plan))
        let first=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:.assessment,hintIDs:["shape-hint"],grade:grade)
        try store.append(first)
        var historical=plan;historical.contentVersion=ContractVersion(2)
        XCTAssertFalse(try store.hasCurrentCompletion(for:historical))
        XCTAssertTrue(try store.hasCurrentCompletion(for:plan))
        let reset=try store.previewReset(objectiveID:plan.objectiveID)
        let second=LearningAttempt(objectiveID:"other",contentVersion:ContractVersion(),assessmentVersion:ContractVersion(),mode:.guided,grade:grade)
        try store.append(second);XCTAssertThrowsError(try store.applyReset(reset))
        let refreshed=try store.previewReset(objectiveID:plan.objectiveID);try store.applyReset(refreshed)
        XCTAssertEqual(try store.attempts().map(\.id),[second.id])
    }
}
