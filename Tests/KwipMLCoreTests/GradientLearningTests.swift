import XCTest
@testable import KwipMLCore

final class GradientLearningTests:XCTestCase {
    func testCorrectKeepSettingsRequestNeverAsksForFrozenParameter() throws {
        let task=GradientLearningTask.keepSettings
        guard case .array(let requests)=task.gradientValues["requests"],case .object(let request)=try XCTUnwrap(requests.first),case .array(let targets)=request["targets"] else {return XCTFail("Keep-settings needs one real input-only request.")}
        XCTAssertEqual(targets,[.object(["kind":.string("input"),"node":.string("input"),"port":.string("out")])])
        XCTAssertEqual(request["unavailable"],.string("error"))
        let configuration=try GradientLearningConfiguration(gradientValues:task.gradientValues,trainabilityValues:task.trainabilityValues,task:task)
        XCTAssertEqual(configuration.trainability.owners.count,2)
        XCTAssertEqual(try GradientLearningConfiguration(gradientValues:GradientLearningTask.combineSmallGroups.gradientValues,trainabilityValues:[:],task:.combineSmallGroups).gradient.accumulationSteps,2)
    }
    func testIndependentAffineMeanInfluenceMatchesHandCalculatedValues() throws {
        let result=try GradientLessonReference.meanDerivatives(widths:[2,2,1],parameters:[1,2,-1,0.5,0.25,-0.5,2,-3,0.75],rows:[1,-2,0.5,1])
        XCTAssertEqual(result.output,[2.75,7.75])
        XCTAssertEqual(result.input,[2.5,1.25,2.5,1.25])
        XCTAssertEqual(result.firstWeight,[1.5,-1,-2.25,1.5])
        XCTAssertThrowsError(try GradientLessonReference.meanDerivatives(widths:[2,2,1],parameters:[0],rows:[1,2]))
    }
    func testFourVariedFixturesFiveCriteriaAndFloat32PartitionUniqueness() throws {
        for task in GradientLearningTask.allCases {
            let a=try GradientLearningAssessment.plan(task:task,runtimeVersion:"fixture-runtime",seed:7)
            let b=try GradientLearningAssessment.plan(task:task,runtimeVersion:"fixture-runtime",seed:41)
            XCTAssertEqual(a.fixtures.count,4);XCTAssertEqual(a.criteria.count,5);XCTAssertNotEqual(a.fixtureDigest,b.fixtureDigest)
            try a.validate()
            for (index,f) in a.fixtures.enumerated() {
                XCTAssertEqual(f.expected.mapValues(\.shape),try task.measuredShapes(widths:GradientLearningTask.widths[index]))
                let i=GradientLearningTask.widths[index][0],training=try XCTUnwrap(f.inputs["trainingInputs"]),heldout=try XCTUnwrap(f.inputs["heldoutInputs"])
                let values=training.values+heldout.values
                XCTAssertEqual(Set(stride(from:0,to:values.count,by:i).map {Float(values[$0]).bitPattern}).count,8)
                XCTAssertEqual(f.inputs.count,5)
                var boundary=f.inputs;boundary["seed"] = .init(shape:[1],values:[4_294_967_295])
                XCTAssertNoThrow(try GradientLessonReference(task:task,inputs:boundary))
                boundary["seed"] = .init(shape:[1],values:[4_294_967_296])
                XCTAssertThrowsError(try GradientLessonReference(task:task,inputs:boundary))
                boundary=f.inputs;boundary["trainingInputs"]?.values[0]=1_000_001
                XCTAssertThrowsError(try GradientLessonReference(task:task,inputs:boundary))
            }
        }
    }
    func testCompleteTypedConstructionPartialEditingAndStrictDecode() throws {
        for task in GradientLearningTask.allCases {
            let graph=try GradientLearningExamples.preparedGraph(task:task)
            XCTAssertEqual(graph.schemaVersion,10);XCTAssertEqual(graph.nodes.count,4);XCTAssertEqual(graph.edges.count,3)
            let submission=try GradientLearningSubmission(graph:graph,task:task)
            XCTAssertEqual(try GradientLearningSubmission.decode(submission.encoded(),task:task),submission)
            let partial=try GradientLearningExamples.defaultGraph(task:task)
            XCTAssertTrue(partial.edges.isEmpty)
            XCTAssertNoThrow(try GradientLearningSubmission.validatedGraph(partial,task:task,complete:false))
            XCTAssertThrowsError(try GradientLearningSubmission(graph:partial,task:task))
            var wrong=graph;wrong.nodes[2].config["lr"] = .number(1)
            XCTAssertThrowsError(try GradientLearningSubmission(graph:wrong,task:task))
            var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:submission.encoded()) as? [String:Any]);raw["expected"]=[0]
            XCTAssertThrowsError(try GradientLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw),task:task))
        }
    }
    func testMeasuredProjectionRefusesPrivateStaleAndMissingCriteria() throws {
        let task=GradientLearningTask.selectedInfluence,plan=try GradientLearningAssessment.plan(task:task,runtimeVersion:"fixture-runtime",seed:7)
        let submission=try GradientLearningSubmission(graph:GradientLearningExamples.preparedGraph(task:task),task:task).encoded()
        // Explicit synthetic transport fixture tests projection admission only; no runtime claim.
        var evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:4096)
        XCTAssertNotNil(GradientFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:submission,mode:.guided,attemptMode:.guided))
        XCTAssertNil(GradientFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:submission,mode:.assessment,attemptMode:.assessment))
        XCTAssertNil(GradientFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:Data("stale".utf8),mode:.guided,attemptMode:.guided))
        evidence.fixtures[0].tensors.removeValue(forKey:"inputInfluenceDifference")
        XCTAssertNil(GradientFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:submission,mode:.guided,attemptMode:.guided))
    }
    func testIndependentMeanUpdateAndFreshMomentumKeepFrozenSettingsSeparate() throws {
        // y=2x for two scalar affine layers; mean square over x=1,3 is 20.
        let p=[1.0,0,2,0],x=[1.0,3],target=[0.0,0]
        let a=try GradientLessonReference.adjustment(widths:[1,1,1],parameters:p,rows:x,targets:target,learningRate:0.1,momentum:0.5)
        XCTAssertEqual(a.objective,20);XCTAssertEqual(a.gradients,[40,16,20,8])
        XCTAssertEqual(a.parameters,[-3,-1.6,0,-0.8]);XCTAssertEqual(a.memory,[40,16,20,8])
        let frozen=try GradientLessonReference.adjustment(widths:[1,1,1],parameters:p,rows:x,targets:target,learningRate:0.1,freezeFirst:true)
        XCTAssertEqual(Array(frozen.parameters.prefix(2)),[1,0]);XCTAssertNil(frozen.memory)
        // A short group must be weighted by its actual row count.
        let one=try GradientLessonReference.adjustment(widths:[1,1,1],parameters:p,rows:[1],targets:[0])
        XCTAssertEqual(one.objective,4);XCTAssertNotEqual(a.objective,one.objective)
        let uneven=try GradientLessonReference.adjustment(widths:[1,1,1],parameters:p,rows:[1,1,3],targets:[0,0,0])
        XCTAssertEqual(uneven.objective,44.0/3,accuracy:1e-12);XCTAssertNotEqual(uneven.objective,(4+36)/2)
        XCTAssertThrowsError(try GradientLessonReference.adjustment(widths:[1,1,1],parameters:p,rows:x,targets:[0]))
    }
    func testUnavailableOwnersForeignRequestsAndNoTrainableCompleteConstructionRefuse() throws {
        let task=GradientLearningTask.keepSettings
        var gradient=task.gradientValues
        guard case .array(var requests)=gradient["requests"],case .object(var request)=requests[0] else {return XCTFail("Missing genuine input request")}
        request["targets"] = .array([.object(["kind":.string("parameter"),"node":.string("first"),"parameter":.string("weight")])]);requests[0] = .object(request);gradient["requests"] = .array(requests)
        XCTAssertThrowsError(try GradientLearningConfiguration(gradientValues:gradient,trainabilityValues:task.trainabilityValues,task:task))
        let allFrozen:JSONValue = .array(["first","second"].flatMap {node in ["weight","bias"].map {parameter in .object(["node":.string(node),"parameter":.string(parameter),"mode":.string("frozen")])}})
        var graph=try GradientLearningExamples.preparedGraph(task:task);graph.nodes[1].config=["owners":allFrozen]
        XCTAssertNoThrow(try GradientLearningSubmission.validatedGraph(graph,task:task,complete:false))
        XCTAssertThrowsError(try GradientLearningSubmission(graph:graph,task:task))
        graph=try GradientLearningExamples.preparedGraph(task:.selectedInfluence)
        graph.edges[0].targetPort="optimizer"
        XCTAssertThrowsError(try GradientLearningSubmission(graph:graph,task:.selectedInfluence))
        XCTAssertThrowsError(try GradientLearningConfiguration(gradientValues:[:],trainabilityValues:["owners":.array([.object(["node":.string("not-supplied"),"parameter":.string("weight"),"mode":.string("frozen")])])],task:.keepSettings))
    }
    func testPlanAndProjectionRejectStaleIncompleteAndForeignEvidence() throws {
        let task=GradientLearningTask.selectedInfluence,plan=try GradientLearningAssessment.plan(task:task,runtimeVersion:"fixture-runtime",seed:41)
        let bytes=try GradientLearningSubmission(graph:GradientLearningExamples.preparedGraph(task:task),task:task).encoded()
        var evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:4096)
        XCTAssertNil(GradientFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:bytes,mode:.guided,attemptMode:.assessment))
        evidence.fixtures[0].tensors["foreign"] = .init(shape:[1],values:[0])
        XCTAssertNil(GradientFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:bytes,mode:.guided,attemptMode:.guided))
        var incomplete=plan;incomplete.criteria.removeLast()
        XCTAssertThrowsError(try GradientLearningAssessment.validate(incomplete))
        var wrong=plan;wrong.fixtures[0].expected.removeValue(forKey:"analysisStateDifference")
        XCTAssertThrowsError(try GradientLearningAssessment.validate(wrong))
    }

}
