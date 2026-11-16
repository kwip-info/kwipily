import XCTest
@testable import KwipMLCore

final class RemainingActivationLearningTests:XCTestCase {
    private func graph(_ task:RemainingActivationLearningTask)->GraphDocument {
        .init(name:"My measurements",nodes:[.init(id:"x",type:"input",x:0,y:0),.init(id:"a",type:task.componentID,x:200,y:0,config:task.configuration),.init(id:"out",type:"output",x:400,y:0)],edges:[.init(id:"first",source:"x",target:"a"),.init(id:"second",source:"a",target:"out")])
    }
    func testAllFiveSeededPlansAndStrictParameterAwareSubmissionRoundtrip()throws {
        for task in RemainingActivationLearningTask.allCases {
            let first=try RemainingActivationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:17),next=try RemainingActivationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:18)
            XCTAssertEqual(first.fixtures.count,4);XCTAssertNotEqual(first.fixtureDigest,next.fixtureDigest)
            XCTAssertTrue(first.fixtures.allSatisfy {$0.expected.values.allSatisfy(\.wellFormed)})
            XCTAssertEqual(Set(first.criteria.map(\.id)),task.isPReLU ? ["output","inputGradient","slopeGradient"]:task == .dropoutEvaluation ? ["output","inputGradient","trainingOutput","trainingInputGradient"]:["output","inputGradient"])
            let state=task.slopes.map {["a":["weight":$0]]} ?? [:]
            let submission=try RemainingActivationLearningSubmission(graph:graph(task),parameters:state,gradientNodeID:task.isPReLU ? "a":nil,task:task)
            let bytes=try submission.encoded();XCTAssertEqual(try RemainingActivationLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any]);raw["mask"]=[1]
            XCTAssertThrowsError(try RemainingActivationLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw),task:task))
            if task.isDropout {XCTAssertThrowsError(try RemainingActivationAssessment.plan(task:task,runtimeVersion:"2.99.0",seed:17))}
        }
    }
    func testScalarChannelSlopesZeroConventionAndWeightedParameterInfluence()throws {
        let input=LearningTensor(shape:[2,2,3],values:Array(repeating:[-2.0,0,2,-4,0,4],count:2).flatMap {$0})
        let reference=try RemainingActivationLessonReference(input:input,task:.channelSlopes)
        XCTAssertEqual(reference.output.values,Array(repeating:[-0.2,0,2,-2,0,4],count:2).flatMap {$0})
        let gradient=try reference.gradients(for:.init(shape:input.shape,values:Array(repeating:1,count:12)))
        XCTAssertEqual(gradient["inputGradient"]?.values,Array(repeating:[0.1,0.1,1,0.5,0.5,1],count:2).flatMap {$0})
        XCTAssertEqual(gradient["slopeGradient"]?.values,[-4,-8])
        let shared=try RemainingActivationLessonReference(input:input,task:.sharedSlope)
        XCTAssertEqual(try shared.gradients(for:.init(shape:input.shape,values:Array(repeating:2,count:12)))["slopeGradient"]?.values,[-24])
    }
    func testLogarithmicReferenceIsStableAndUsesFirstSampleDirection()throws {
        let input=LearningTensor(shape:[2,2,3],values:Array(repeating:[1000.0,1001,1002,1000,1001,1002],count:2).flatMap {$0})
        let reference=try RemainingActivationLessonReference(input:input,task:.logScores)
        for value in reference.output.values {XCTAssertEqual(value,-log(2),accuracy:1e-12)}
        let influence=LearningTensor(shape:input.shape,values:Array(repeating:[2.0,2,2,1,1,1],count:2).flatMap {$0})
        XCTAssertEqual(try reference.inputGradient(for:influence).values,Array(repeating:[0.5,0.5,0.5,-0.5,-0.5,-0.5],count:2).flatMap {$0})
    }
    func testFrozenMasksDriveScalarTrainingAndEvaluationChecksWithoutMaskOverrides()throws {
        for index in 0..<12 {
            let entry=try RemainingActivationMaskBank.entry(index:index,runtimeVersion:"2.14.1")
            let input=LearningTensor(shape:entry.shape,values:Array(repeating:3,count:entry.mask.count)),upstream=LearningTensor(shape:entry.shape,values:Array(repeating:-2,count:entry.mask.count))
            let train=try RemainingActivationLessonReference(input:input,task:.dropoutTraining,canonicalMask:entry.mask)
            XCTAssertEqual(train.output.values,entry.mask.map {Double($0)*6})
            XCTAssertEqual(try train.inputGradient(for:upstream).values,entry.mask.map {Double($0)*(-4)})
            let evaluation=try RemainingActivationLessonReference(input:input,task:.dropoutEvaluation,canonicalMask:entry.mask)
            XCTAssertEqual(evaluation.output,input);XCTAssertEqual(evaluation.trainingOutput,train.output)
            let gradients=try evaluation.gradients(for:upstream)
            XCTAssertEqual(gradients["inputGradient"],upstream);XCTAssertEqual(gradients["trainingInputGradient"]?.values,entry.mask.map {Double($0)*(-4)})
        }
        XCTAssertThrowsError(try RemainingActivationMaskBank.entry(index:12,runtimeVersion:"2.14.1"))
        XCTAssertThrowsError(try RemainingActivationMaskBank.entry(index:0,runtimeVersion:"2.99.0"))
    }
    func testIncompleteDraftSchemaStateAndReferenceBoundariesStayStrict()throws {
        for task in RemainingActivationLearningTask.allCases {
            XCTAssertNoThrow(try RemainingActivationLearningSubmission.validatedGraph(GraphDocument(),task:task,complete:false))
            XCTAssertThrowsError(try RemainingActivationLearningSubmission(graph:GraphDocument(),task:task))
            var wrong=graph(task);wrong.schemaVersion=1
            XCTAssertThrowsError(try RemainingActivationLearningSubmission.validatedGraph(wrong,task:task,complete:false))
            if task.isPReLU {
                XCTAssertThrowsError(try RemainingActivationLearningSubmission(graph:graph(task),parameters:[:],gradientNodeID:"a",task:task))
                XCTAssertThrowsError(try RemainingActivationLearningSubmission(graph:graph(task),parameters:["a":["weight":.init(shape:[3],values:[1,2,3])]],gradientNodeID:"a",task:task))
            } else {XCTAssertThrowsError(try RemainingActivationLearningSubmission(graph:graph(task),parameters:["a":["weight":.init(shape:[1],values:[1])]],task:task))}
        }
        XCTAssertThrowsError(try RemainingActivationLessonReference(input:.init(shape:[1,2,3],values:Array(repeating:1,count:6)),task:.sharedSlope))
        XCTAssertThrowsError(try RemainingActivationLessonReference(input:.init(shape:[2,2,3],values:Array(repeating:1,count:12)),task:.dropoutTraining,canonicalMask:Array(repeating:2,count:12)))
    }
}
