import XCTest
@testable import KwipMLCore

final class GELULessonTests:XCTestCase {
    func testNeutralIdentityPreservesHistoricalCanonicalBytes()throws {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        for (identity,id,seed) in [(ModelConstructionLessonIdentity.composition,"assessment.affine-composition",nil as UInt64?),(.backwardChain,"assessment.affine-backward-chain",nil),(.recovery(taskSeed:42),"assessment.affine-from-probes",42)] {
            var expected:[String:Any]=["assessmentID":id,"contentVersion":["major":1,"minor":0,"patch":0]]
            if let seed {expected["taskSeed"]=seed}
            let bytes=try JSONSerialization.data(withJSONObject:expected,options:[.sortedKeys,.withoutEscapingSlashes])
            XCTAssertEqual(try encoder.encode(identity),bytes)
            XCTAssertEqual(try JSONDecoder().decode(AffineLessonIdentity.self,from:bytes),identity)
            let draft=AffineLearningDraft(projectID:"11111111-1111-1111-1111-111111111111",graph:ExampleFlows.parallelClassifier,lessonIdentity:identity)
            let old=try AffineLearningCodec.encodeDraft(draft)
            XCTAssertEqual(try AffineLearningCodec.encodeDraft(AffineLearningCodec.decodeDraft(old)),old)
            XCTAssertNil(identity.geluApproximateTanh)
        }
        XCTAssertEqual(ModelConstructionLessonIdentity.geluExact.geluApproximateTanh,false)
        XCTAssertEqual(ModelConstructionLessonIdentity.geluTanh.geluApproximateTanh,true)
    }
    func testIndependentReferenceAndBalancedFixturesSeparateModesAndMisconceptions()throws {
        let exact=LearningExamples.geluReference(approximateTanh:false),approximate=LearningExamples.geluReference(approximateTanh:true)
        XCTAssertEqual(try exact.apply(0),0);XCTAssertEqual(try exact.apply(1),0.8413447460685429,accuracy:1e-14)
        XCTAssertEqual(try exact.apply(-1),-0.15865525393145707,accuracy:1e-14)
        XCTAssertEqual(try approximate.apply(1),0.8411919906082768,accuracy:1e-14)
        XCTAssertThrowsError(try exact.apply(.infinity));XCTAssertThrowsError(try approximate.apply(.nan))
        XCTAssertTrue(exact.formula.contains("erf"));XCTAssertTrue(approximate.formula.contains("tanh"))
        for identity in [ModelConstructionLessonIdentity.geluExact,.geluTanh] {
            let task=try ModelConstructionLessonTasks.publicTask(identity:identity)
            XCTAssertEqual(task.allowedNodeTypes,["input","linear","gelu","add","concat","output"])
            let plan=try ModelConstructionLessonTasks.plan(identity:identity,runtimeVersion:"2.14.1",fixtureSeed:42)
            XCTAssertEqual(plan.runtime.capabilities,["linear":ContractVersion(1,1,0),"gelu":ContractVersion()])
            XCTAssertNotEqual(plan.fixtureDigest,try ModelConstructionLessonTasks.plan(identity:identity,runtimeVersion:"2.14.1",fixtureSeed:43).fixtureDigest)
            XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertTrue(plan.fixtures.flatMap{$0.inputs["x"]!.values}.contains(0))
            for fixture in plan.fixtures {
                let values=fixture.inputs["x"]!.values
                XCTAssertEqual(fixture.inputs["x"]!.shape,[2,2]);XCTAssertLessThan(values[0],-1);XCTAssertGreaterThan(values[3],1)
                XCTAssertLessThanOrEqual(abs(values[1]),0.1);XCTAssertGreaterThan(values[2],0);XCTAssertLessThanOrEqual(values[2],0.1)
            }
            func grade(_ transform:(Double)throws->Double)throws->LearningGrade {
                let fixtures=try plan.fixtures.map{fixture in LearningFixtureEvidence(fixtureID:fixture.id,tensors:["output":LearningTensor(shape:[2,2],values:try fixture.inputs["x"]!.values.map(transform))])}
                return try LearningAssessor.evaluate(plan,evidence:LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:fixtures,elapsedMilliseconds:1,maximumTensorElements:16))
            }
            let correct=LearningExamples.geluReference(approximateTanh:identity.geluApproximateTanh!)
            let wrong=LearningExamples.geluReference(approximateTanh:!identity.geluApproximateTanh!)
            XCTAssertEqual(try grade(correct.apply).status,.passed)
            XCTAssertEqual(try grade(wrong.apply).status,.failed)
            XCTAssertEqual(try grade{max(0,$0)}.status,.failed)
            XCTAssertEqual(try grade{tanh($0)}.status,.failed)
            XCTAssertEqual(try grade{try correct.apply(correct.apply($0))}.status,.failed)
        }
    }
    private func draft(_ mode:Bool)->AffineLearningDraft {
        let graph=GraphDocument(nodes:[GraphNode(id:"i",type:"input",config:["features":2]),GraphNode(id:"m",type:"linear",config:["in_features":2,"out_features":2,"bias":.bool(false)]),GraphNode(id:"g",type:"gelu",config:["approximate_tanh":.bool(mode)]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"m"),GraphEdge(source:"m",target:"g"),GraphEdge(source:"g",target:"o")])
        var draft=AffineLearningDraft(projectID:UUID().uuidString,graph:graph,lessonIdentity:mode ? .geluTanh:.geluExact)
        draft.parameters["m"]=LinearParameterValues(weight:.init(shape:[2,2],values:[1,0,0,1]),bias:.init(shape:[2],values:[0,0]));return draft
    }
    func testConstructionPolicyAndReplayPrivacyAreTaskSpecific()throws {
        for mode in [false,true] {
            var draft=draft(mode);XCTAssertNoThrow(try draft.submission())
            let identity=try draft.resolvedLessonIdentity(),submission=try draft.submission()
            let plan=try ModelConstructionLessonTasks.plan(identity:identity,runtimeVersion:"2.14.1",fixtureSeed:7)
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(try AffineLearningCodec.encodeSubmission(submission)),fixtures:plan.fixtures.map{LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:16)
            let attempt=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:.practice,grade:try LearningAssessor.evaluate(plan,evidence:evidence))
            var receipt=AffineLearningReceipt(attempt:attempt,projectID:draft.projectID,submission:submission,submissionSHA256:evidence.submissionDigest,runtime:plan.runtime,fixtureSeed:plan.fixtureSeed,fixtureDigest:plan.fixtureDigest,notes:"",evidence:evidence,lessonIdentity:identity)
            XCTAssertEqual(try receipt.replayPlan(),plan)
            XCTAssertNotNil(AffineFormativeProjection.make(plan:plan,evidence:evidence,recordedMode:.practice,lessonIdentity:identity))
            XCTAssertNil(AffineFormativeProjection.make(plan:plan,evidence:evidence,recordedMode:.assessment,lessonIdentity:identity))
            receipt.lessonIdentity=mode ? .geluExact:.geluTanh;XCTAssertThrowsError(try receipt.replayPlan())
            draft.graph.nodes[2].config["approximate_tanh"] = .bool(!mode);XCTAssertThrowsError(try draft.submission())
            draft.graph.nodes[2].type="relu";draft.graph.nodes[2].config=[:];XCTAssertThrowsError(try draft.submission())
            draft.graph.nodes.remove(at:2);draft.graph.edges=[GraphEdge(source:"i",target:"m"),GraphEdge(source:"m",target:"o")];XCTAssertThrowsError(try draft.submission())
            XCTAssertThrowsError(try ModelConstructionLessonTasks.plan(identity:identity,runtimeVersion:"2.14.1",fixtureSeed:7,assessmentVersion:ContractVersion(1,1)))
        }
        for identity in [ModelConstructionLessonIdentity.composition,.backwardChain,.recovery(taskSeed:7)] {XCTAssertFalse(try ModelConstructionLessonTasks.publicTask(identity:identity).allowedNodeTypes.contains("gelu"))}
    }
    func testEveryFixtureSeedHasGuaranteedModeSeparatingCoordinate()throws {
        for seed:UInt64 in [0,1,7,42,999,UInt64.max] {
            for identity in [ModelConstructionLessonIdentity.geluExact,.geluTanh] {
                let plan=try ModelConstructionLessonTasks.plan(identity:identity,runtimeVersion:"2.14.1",fixtureSeed:seed)
                XCTAssertEqual(plan.fixtures.first?.inputs["x"]?.values.last,2.5)
                let wrong=LearningExamples.geluReference(approximateTanh:!identity.geluApproximateTanh!)
                let fixtures=try plan.fixtures.map { fixture in
                    LearningFixtureEvidence(fixtureID:fixture.id,tensors:["output":LearningTensor(shape:[2,2],values:try fixture.inputs["x"]!.values.map(wrong.apply))])
                }
                let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:fixtures,elapsedMilliseconds:1,maximumTensorElements:16)
                XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.failed)
            }
        }
    }

}
