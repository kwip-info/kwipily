import XCTest
@testable import KwipMLCore

final class AffineBackwardChainTests:XCTestCase {
    private let identity=AffineLessonIdentity.backwardChain
    func testReferenceUsesRowConventionReverseChainAndIgnoresForwardBias()throws {
        let reference=LearningExamples.affineBackwardReference
        XCTAssertEqual(reference.forward,LearningExamples.affineReference)
        XCTAssertEqual(try reference.apply([1,0]),[3,5])
        XCTAssertEqual(try reference.apply([0,1]),[-3.5,5])
        XCTAssertEqual(try reference.apply([0,0]),[0,0])
        XCTAssertThrowsError(try reference.apply([1]))
        XCTAssertThrowsError(try reference.apply([.infinity,0]))
        let g=[0.75,-1.25],x=[1.1,-0.8],epsilon=1e-5,gradient=try reference.apply(g)
        // Independent central differences of scalar g dot forward(x), including nonzero biases.
        for column in 0..<2 {
            var plus=x,minus=x;plus[column]+=epsilon;minus[column]-=epsilon
            let yp=try reference.forward.apply(plus),ym=try reference.forward.apply(minus)
            let numerical=zip(g,zip(yp,ym)).reduce(0){$0+$1.0*($1.1.0-$1.1.1)/(2*epsilon)}
            XCTAssertEqual(gradient[column],numerical,accuracy:1e-8)
        }
        let text=reference.taskText
        for token in ["W1","W2","b1","b2","[batch,2]","g W2 W1"] {XCTAssertTrue(text.contains(token),token)}
        XCTAssertTrue(reference.instructions.contains("bias"))
    }
    func testPublicReferenceDescribesRectangularForwardShapes()throws {
        let first=try AffineReferenceMap(name:"First",weight:.init(shape:[3,2],values:[1,0,0,1,1,1]),bias:.init(shape:[3],values:[1,2,3]))
        let second=try AffineReferenceMap(name:"Second",weight:.init(shape:[4,3],values:[1,0,0,0,1,0,0,0,1,1,1,1]),bias:.init(shape:[4],values:[1,2,3,4]))
        let reference=AffineBackwardChainReference(forward:try AffineCompositionReference(first:first,second:second))
        XCTAssertEqual(try reference.apply([0,0,0,1]),[2,2])
        for token in ["g [batch,4]","input gradient [batch,2]","W1 [3,2]","W2 [4,3]","b1 [3]","b2 [4]"] {XCTAssertTrue(reference.taskText.contains(token),token)}
        XCTAssertThrowsError(try reference.apply([1,0]))
    }
    func testVersionedVariedPlanAndWrongConstructionsFail()throws {
        let task=try AffineLessonTasks.publicTask(identity:identity)
        XCTAssertEqual(task.allowedNodeTypes,["input","linear","add","concat","output"])
        XCTAssertTrue(task.probes.isEmpty)
        let plan=try AffineLessonTasks.plan(identity:identity,runtimeVersion:"test",fixtureSeed:42)
        XCTAssertEqual(plan.id,"assessment.affine-backward-chain");XCTAssertEqual(plan.objectiveID,"matrices.affine-backward-chain")
        XCTAssertEqual(plan.assessmentVersion,ContractVersion());XCTAssertEqual(plan.runtime.capabilities["linear"],ContractVersion(1,1,0))
        XCTAssertNotEqual(plan.fixtureDigest,try AffineLessonTasks.plan(identity:identity,runtimeVersion:"test",fixtureSeed:43).fixtureDigest)
        for fixture in plan.fixtures {
            let x=try XCTUnwrap(fixture.inputs["x"]),expected=try XCTUnwrap(fixture.expected["output"])
            XCTAssertEqual(x.shape,[2,2]);XCTAssertEqual(expected.shape,[2,2])
            for row in 0..<2 {let a=x.values[row*2],b=x.values[row*2+1];XCTAssertEqual(expected.values[row*2],3*a-3.5*b,accuracy:1e-12);XCTAssertEqual(expected.values[row*2+1],5*a+5*b,accuracy:1e-12)}
        }
        func grade(_ transform:([Double])->[Double])throws->LearningGrade {
            let fixtures=plan.fixtures.map{fixture in LearningFixtureEvidence(fixtureID:fixture.id,tensors:["output":LearningTensor(shape:[2,2],values:transform(fixture.inputs["x"]!.values))])}
            return try LearningAssessor.evaluate(plan,evidence:LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:fixtures,elapsedMilliseconds:1,maximumTensorElements:16))
        }
        func rowMap(_ values:[Double],_ matrix:[Double],bias:Double=0)->[Double] {
            var output:[Double]=[]
            for row in 0..<2 {
                let a=values[row*2],b=values[row*2+1]
                output.append(a*matrix[0]+b*matrix[2]+bias)
                output.append(a*matrix[1]+b*matrix[3])
            }
            return output
        }
        let correct=try grade{rowMap($0,[3,5,-3.5,5])}
        XCTAssertEqual(correct.status,.passed)
        // Transposed combined Jacobian (forward map), wrong multiplication order, and a spurious bias.
        let transposed=try grade{rowMap($0,[3,-3.5,5,5])}
        let wrongOrder=try grade{rowMap($0,[4,3,-5.5,4])}
        let biased=try grade{rowMap($0,[3,5,-3.5,5],bias:1)}
        XCTAssertEqual(transposed.status,.failed)
        XCTAssertEqual(wrongOrder.status,.failed)
        XCTAssertEqual(biased.status,.failed)
    }
    func testIdentityDraftReplayAndFormativePrivacy()throws {
        XCTAssertThrowsError(try AffineLessonIdentity(assessmentID:identity.assessmentID,taskSeed:42).validate())
        XCTAssertThrowsError(try AffineLessonTasks.plan(identity:identity,runtimeVersion:"test",fixtureSeed:1,assessmentVersion:ContractVersion(1,1)))
        let draft=AffineLearningDraft(projectID:UUID().uuidString,graph:ExampleFlows.parallelClassifier,lessonIdentity:identity)
        XCTAssertEqual(try AffineLearningCodec.decodeDraft(AffineLearningCodec.encodeDraft(draft)).resolvedLessonIdentity(),identity)
        XCTAssertThrowsError(try draft.submission())
        let plan=try AffineLessonTasks.plan(identity:identity,runtimeVersion:"test",fixtureSeed:7)
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:plan.fixtures.map{LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:16)
        let attempt=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:.practice,grade:try LearningAssessor.evaluate(plan,evidence:evidence))
        var receipt=AffineLearningReceipt(attempt:attempt,projectID:draft.projectID,submission:AffineLearningSubmission(graph:draft.graph,parameters:[:]),submissionSHA256:evidence.submissionDigest,runtime:plan.runtime,fixtureSeed:plan.fixtureSeed,fixtureDigest:plan.fixtureDigest,notes:"",evidence:evidence,lessonIdentity:identity)
        XCTAssertEqual(try receipt.replayPlan(),plan)
        XCTAssertNotNil(AffineFormativeProjection.make(plan:plan,evidence:evidence,recordedMode:.practice,lessonIdentity:identity))
        XCTAssertNil(AffineFormativeProjection.make(plan:plan,evidence:evidence,recordedMode:.assessment,lessonIdentity:identity))
        XCTAssertNil(AffineFormativeProjection.make(plan:plan,evidence:evidence,recordedMode:.practice,lessonIdentity:.composition))
        receipt.lessonIdentity = .composition;XCTAssertThrowsError(try receipt.replayPlan())
        XCTAssertEqual(try AffineLessonTasks.plan(identity:.composition,runtimeVersion:"test",fixtureSeed:7).assessmentVersion,ContractVersion(1,1,0))
    }
}
