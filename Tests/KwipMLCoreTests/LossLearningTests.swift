import XCTest
@testable import KwipMLCore

final class LossLearningTests:XCTestCase {
    func testEightDistinctTasksHaveFourChangedFixturesAndPlainVisualInstructions()throws {
        XCTAssertEqual(Set(LossLearningTask.allCases.map(\.componentID)),LossConfiguration.componentIDs)
        for task in LossLearningTask.allCases {
            XCTAssertFalse(task.instructions.isEmpty,task.rawValue)
            XCTAssertTrue(task.instructions.contains("Objective check"),task.rawValue)
            XCTAssertTrue(task.instructions.contains("Input influence"),task.rawValue)
            if task.usesLogSoftmax {XCTAssertTrue(task.instructions.contains("LogSoftmax"))}
            let fixtures=task.fixtures(seed:17)
            XCTAssertEqual(fixtures.count,4)
            XCTAssertEqual(Set(fixtures.map(\.id)).count,4)
            XCTAssertEqual(fixtures,task.fixtures(seed:17))
            XCTAssertNotEqual(fixtures,task.fixtures(seed:18))
            for fixture in fixtures {
                XCTAssertEqual(Set(fixture.inputs.keys),Set(task.sourceRoles))
                XCTAssertTrue(fixture.inputs.values.allSatisfy(\.wellFormed))
                XCTAssertEqual(fixture.inputs["prediction"]?.shape,[2]+task.predictionSampleShape)
                let expected=try LossLearningAssessment.expected(task:task,fixture:fixture)
                XCTAssertEqual(Set(expected.keys),["raw","objective","inputInfluence"])
                XCTAssertTrue(expected.values.allSatisfy(\.wellFormed))
                XCTAssertEqual(expected["inputInfluence"]?.shape,fixture.inputs["prediction"]?.shape)
            }
            XCTAssertNoThrow(try LossConfiguration(componentID:task.componentID,values:task.configuration))
        }
        XCTAssertEqual(LossLearningAssessment.modes,["training-objective","evaluation-objective","target-contract","reduction","input-gradients"])
    }
    func testDenseLossArithmeticAndInfluenceHaveIndependentHandCalculatedAnswers()throws {
        let source=LearningTensor(shape:[1,2],values:[1,3]),target=LearningTensor(shape:[1,2],values:[0,1])
        let fixture=LossLessonFixture(id:"hand",inputs:["prediction":source,"target":target])
        let mse=try LossLessonReference(task:.mse,fixture:fixture,configuration:["reduction":.string("sum"),"target_broadcast":false])
        XCTAssertEqual(mse.raw.values,[1,4]);XCTAssertEqual(mse.objective.values,[5])
        for (actual,expected) in zip(mse.inputInfluence.values,[2.0,4.0]) {XCTAssertEqual(actual,expected,accuracy:1e-7)}
        let l1=try LossLessonReference(task:.l1,fixture:fixture,configuration:["reduction":.string("mean"),"target_broadcast":false])
        XCTAssertEqual(l1.raw.values,[1,2]);XCTAssertEqual(l1.objective.values,[1.5])
        for actual in l1.inputInfluence.values {XCTAssertEqual(actual,0.5,accuracy:1e-7)}
        let binary=LossLessonFixture(id:"binary",inputs:["prediction":.init(shape:[1,2],values:[0,0]),"target":target])
        let bce=try LossLessonReference(task:.bce,fixture:binary,configuration:["reduction":.string("mean")])
        XCTAssertEqual(bce.objective.values.first!,log(2),accuracy:1e-12)
        for (actual,expected) in zip(bce.inputInfluence.values,[0.25,-0.25]) {XCTAssertEqual(actual,expected,accuracy:1e-7)}
        let classes=LossLessonFixture(id:"classes",inputs:["prediction":.init(shape:[1,3],values:[0,0,0]),"target":.init(shape:[1],values:[1])])
        for task in [LossLearningTask.crossEntropy,.nll] {
            let ref=try LossLessonReference(task:task,fixture:classes,configuration:["reduction":.string("mean")])
            XCTAssertEqual(ref.objective.values.first!,log(3),accuracy:1e-12)
            for (actual,expected) in zip(ref.inputInfluence.values,[1.0/3,-2.0/3,1.0/3]) {XCTAssertEqual(actual,expected,accuracy:1e-7)}
        }
    }
    func testCTCEnumeratedPathsAndNormalizedScoreGradients()throws {
        let f=LossLessonFixture(id:"ctc-hand",inputs:["prediction":.init(shape:[1,2,2],values:[0,0,0,0]),"target":.init(shape:[1,1],values:[1]),"input_lengths":.init(shape:[1],values:[2]),"target_lengths":.init(shape:[1],values:[1])])
        let ref=try LossLessonReference(task:.ctc,fixture:f,configuration:["reduction":.string("mean")])
        XCTAssertEqual(ref.objective.values.first!,-log(0.75),accuracy:1e-12)
        for (actual,expected) in zip(ref.inputInfluence.values,[1.0/6,-1.0/6,1.0/6,-1.0/6]) {XCTAssertEqual(actual,expected,accuracy:1e-7)}
        var inputs=f.inputs;inputs["target"] = .init(shape:[1,2],values:[1,1]);inputs["target_lengths"] = .init(shape:[1],values:[2])
        XCTAssertThrowsError(try LossLessonReference(task:.ctc,fixture:.init(id:"impossible",inputs:inputs),configuration:["zero_infinity":false]))
        let zero=try LossLessonReference(task:.ctc,fixture:.init(id:"masked",inputs:inputs),configuration:["zero_infinity":true])
        XCTAssertEqual(zero.objective.values,[0]);XCTAssertEqual(zero.inputInfluence.values,[0,0,0,0])
    }
    func testKLBatchDenominatorTripletSwapAndExplicitNoneScalarization()throws {
        let k=LossLessonFixture(id:"kl",inputs:["prediction":.init(shape:[2,2],values:[0,0,0,0]),"target":.init(shape:[2,2],values:[1,0,1,0])])
        let mean=try LossLessonReference(task:.kl,fixture:k,configuration:["reduction":.string("mean"),"log_target":false])
        let batch=try LossLessonReference(task:.kl,fixture:k,configuration:["reduction":.string("batchmean"),"log_target":false])
        XCTAssertEqual(mean.objective.values.first!,log(2)/2,accuracy:1e-12)
        XCTAssertEqual(batch.objective.values.first!,log(2),accuracy:1e-12)
        let f=LossLessonFixture(id:"triplet",inputs:["prediction":.init(shape:[1,3,2],values:[0,0,1,0,1.1,0])])
        let a=try LossLessonReference(task:.triplet,fixture:f,configuration:["eps":0,"swap":false])
        let b=try LossLessonReference(task:.triplet,fixture:f,configuration:["eps":0,"swap":true])
        XCTAssertEqual(a.objective.values.first!,0.9,accuracy:1e-12);XCTAssertEqual(b.objective.values.first!,1.9,accuracy:1e-12)
        let m=LossLessonFixture(id:"reduce",inputs:["prediction":.init(shape:[1,2],values:[1,2]),"target":.init(shape:[1,2],values:[0,0])])
        XCTAssertEqual(try LossLessonReference(task:.mse,fixture:m,configuration:["reduction":.string("none")],scalarization:"sum").objective.values,[5])
        XCTAssertEqual(try LossLessonReference(task:.mse,fixture:m,configuration:["reduction":.string("none")],scalarization:"mean").objective.values,[2.5])
        XCTAssertThrowsError(try LossLessonReference(task:.mse,fixture:m,scalarization:"hidden"))
    }
    func testStrictSourceQualifiedRecipeTopologyAndVisibleNormalization()throws {
        for task in LossLearningTask.allCases {
            let graph=LossLearningCanvas.preparedGraph(task:task)
            XCTAssertNoThrow(try LossLearningSubmission(graph:graph,task:task))
            XCTAssertEqual(graph.nodes.filter {$0.type==task.componentID}.count,1)
            XCTAssertEqual(graph.nodes.filter {$0.type=="objective_check"}.count,1)
            XCTAssertEqual(graph.nodes.filter {$0.type=="log_softmax"}.count,task.usesLogSoftmax ? 1:0)
            let data=try LossLearningSubmission(graph:graph,task:task).encoded()
            let json=try JSONDecoder().decode(JSONValue.self,from:data)
            guard case .object(let envelope)=json else {return XCTFail("Envelope")}
            XCTAssertEqual(envelope["kind"],.string("lossObjectiveGraph"))
            guard !graph.edges.isEmpty,!graph.nodes.isEmpty else {XCTFail("Prepared graph must contain real lesson connections");continue}
            var broken=graph;broken.edges.removeLast()
            XCTAssertThrowsError(try LossLearningSubmission(graph:broken,task:task))
            XCTAssertNoThrow(try LossLearningSubmission.validatedGraph(broken,task:task,complete:false))
            var forged=graph;forged.nodes.append(.init(id:"foreign",type:"linear"))
            XCTAssertThrowsError(try LossLearningSubmission(graph:forged,task:task))
            var collision=graph;collision.nodes.append(graph.nodes[0])
            XCTAssertThrowsError(try LossLearningSubmission(graph:collision,task:task))
            guard let recipe=graph.nodes.first(where:{$0.type==task.componentID}) else {XCTFail("Real recipe node required");continue}
            XCTAssertEqual(graph.edges.filter {$0.source==recipe.id}.map(\.targetPort),["objective"])
        }
    }
    func testIndependentFiveModePlansAndStrictQualifiedSourceReplay()throws {
        for task in LossLearningTask.allCases {
            let plan=try LossLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:17)
            XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),Set(LossLearningAssessment.modes))
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            XCTAssertNoThrow(try plan.validate())
            XCTAssertNotEqual(plan.fixtureDigest,try LossLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:18).fixtureDigest)
            for fixture in plan.fixtures {XCTAssertEqual(Set(fixture.expected.keys),["trainingObjective","evaluationObjective","targetContract","reduction","inputInfluence"])}
            let submission=try LossLearningSubmission(graph:LossLearningCanvas.preparedGraph(task:task),task:task)
            let data=try submission.encoded()
            XCTAssertEqual(try LossLearningSubmission.decode(data,task:task),submission)
            guard case .object(var envelope)=try JSONDecoder().decode(JSONValue.self,from:data) else {return XCTFail("Typed envelope")}
            envelope["objectiveSource"] = .object(["domain":.string("model"),"localID":.string("forged")])
            XCTAssertThrowsError(try LossLearningSubmission.decode(JSONEncoder().encode(JSONValue.object(envelope)),task:task))
            envelope["unexpected"] = true
            XCTAssertThrowsError(try LossLearningSubmission.decode(JSONEncoder().encode(JSONValue.object(envelope)),task:task))
        }
    }
    func testPrivateSeedChangesMeasuredAnswersRatherThanOnlyFixtureIdentity()throws {
        for task in LossLearningTask.allCases {
            let first=try LossLessonReference(task:task,fixture:task.fixtures(seed:17)[0])
            let changed=try LossLessonReference(task:task,fixture:task.fixtures(seed:18)[0])
            let difference=zip(first.raw.values,changed.raw.values).reduce(0) {$0+abs($1.0-$1.1)}
            XCTAssertGreaterThan(difference,1e-8,task.rawValue+" must not cancel its private seed inside normalization or relative distances")
        }
    }
    func testMalformedRolesAndLengthsRefuseWithoutAnswerLeakOrSilentRepair()throws {
        let malformed=LossLessonFixture(id:"wrong",inputs:["prediction":.init(shape:[1,2],values:[1,2]),"target":.init(shape:[1],values:[0.5])])
        XCTAssertThrowsError(try LossLessonReference(task:.crossEntropy,fixture:malformed,configuration:[:]))
        XCTAssertThrowsError(try LossLessonReference(task:.mse,fixture:malformed,configuration:["target_broadcast":false]))
        for task in LossLearningTask.allCases {
            if let f=task.fixtures(seed:11).first {
                var missing=f.inputs;missing.removeValue(forKey:task.sourceRoles.last!)
                XCTAssertThrowsError(try LossLessonReference(task:task,fixture:.init(id:"missing",inputs:missing)))
            }
        }
    }
}
