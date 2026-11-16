import XCTest
@testable import KwipMLCore
final class StatefulNormalizationLearningTests:XCTestCase {
    func graph(_ t:StatefulNormalizationLearningTask)->GraphDocument {.init(name:"Channel patterns",nodes:[.init(id:"x",type:"input"),.init(id:"op",type:t.componentID,config:t.configuration),.init(id:"out",type:"output")],edges:[.init(id:"a",source:"x",target:"op"),.init(id:"b",source:"op",target:"out")])}
    func parameters(_ t:StatefulNormalizationLearningTask)->[String:[String:LearningTensor]] {guard let scale=t.scale,let bias=t.bias else {return [:]};return ["op":["weight":scale,"bias":bias]]}
    func testTenSeededPlansExactStateOwnershipAndCanonicalSubmissions()throws {
        XCTAssertEqual(StatefulNormalizationLearningTask.allCases.count,10)
        for t in StatefulNormalizationLearningTask.allCases {
            let a=try StatefulNormalizationAssessment.plan(task:t,runtimeVersion:"2.14.1",seed:41),b=try StatefulNormalizationAssessment.plan(task:t,runtimeVersion:"2.14.1",seed:173)
            XCTAssertNotEqual(a.fixtures,b.fixtures);XCTAssertEqual(a.fixtures.map {$0.inputs["x"]!.shape[0]},[1,2,3,1])
            XCTAssertEqual(Set(a.fixtures[0].inputs.keys),t.remembered ? ["x","upstream","calibrationA","calibrationB"]:["x","upstream"])
            let s=try StatefulNormalizationLearningSubmission(graph:graph(t),parameters:parameters(t),gradientNodeID:t.affine ? "op":nil,task:t)
            XCTAssertEqual(try StatefulNormalizationLearningSubmission.decode(s.encoded(),task:t),s)
            XCTAssertEqual(a.runtime.capabilities,[t.componentID:ContractVersion()])
            XCTAssertEqual(a.fixtures[0].expected["batchesTracked"]?.values,t.remembered ? [t.componentID.hasPrefix("batch") ? 2:0]:nil)
        }
    }
    func testKnownRememberedSummariesAndFixedEvaluationInfluence()throws {
        for t in [StatefulNormalizationLearningTask.bn1Remembered,.in1Remembered] {
            let a=LearningTensor(shape:[2,3,4],values:[2,2,2,2,5,5,5,5,-1,-1,-1,-1,2,2,2,2,5,5,5,5,-1,-1,-1,-1])
            let b=LearningTensor(shape:[3,3,4],values:Array(repeating:[6.0,6,6,6,-1,-1,-1,-1,3,3,3,3],count:3).flatMap {$0})
            let x=LearningTensor(shape:[1,3,4],values:Array(repeating:1,count:12)),r=try StatefulNormalizationLessonReference(input:x,task:t,calibrationA:a,calibrationB:b)
            XCTAssertEqual(r.runningMean,[1.875,0.6875,0.5625]);XCTAssertEqual(r.runningVariance,[0.5625,0.5625,0.5625])
            let expected=try r.expected(for:.init(shape:x.shape,values:Array(repeating:1,count:12)))
            XCTAssertEqual(expected["runningMean"],expected["postEvalRunningMean"]);XCTAssertEqual(expected["runningVariance"],expected["postEvalRunningVariance"])
            for ch in 0..<3 {for pos in 0..<4 {XCTAssertEqual(expected["inputGradient"]!.values[ch*4+pos],t.scale!.values[ch]/sqrt(0.6875),accuracy:1e-12)}}
        }
    }
    func testCurrentStatisticsAnalyticInfluenceMatchesIndependentFiniteDifference()throws {
        for t in [StatefulNormalizationLearningTask.bn1Current,.in1Current,.groupAdjustments] {
            let shape=[2]+t.sampleShape,count=shape.reduce(1,*),x=LearningTensor(shape:shape,values:(0..<count).map {Double(($0*7)%23)-9}),g=LearningTensor(shape:shape,values:(0..<count).map {Double(($0*3)%7)-2})
            let r=try StatefulNormalizationLessonReference(input:x,task:t),dx=try r.expected(for:g)["inputGradient"]!.values
            for i in [0,count/2,count-1] {var plus=x,minus=x;plus.values[i]+=1e-5;minus.values[i]-=1e-5
                let p=try StatefulNormalizationLessonReference(input:plus,task:t).output.values,m=try StatefulNormalizationLessonReference(input:minus,task:t).output.values
                let numerical=zip(zip(p,m),g.values).reduce(0.0) {$0+($1.0.0-$1.0.1)*$1.1}/2e-5
                XCTAssertEqual(dx[i],numerical,accuracy:1e-7)
            }
        }
    }
    func testNoAuthoredBuffersModesAnswersOrInactiveParameters()throws {
        let t=StatefulNormalizationLearningTask.bn1Remembered,s=try StatefulNormalizationLearningSubmission(graph:graph(t),parameters:parameters(t),gradientNodeID:"op",task:t)
        for key in ["buffers","training","seed","expected"] {var raw=try JSONSerialization.jsonObject(with:s.encoded()) as! [String:Any];raw[key]=[:];XCTAssertThrowsError(try StatefulNormalizationLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw),task:t))}
        XCTAssertThrowsError(try StatefulNormalizationLearningSubmission(graph:graph(.bn1Current),parameters:parameters(t),gradientNodeID:"op",task:.bn1Current))
        var bad=graph(t);bad.nodes[1].config["momentum"] = .null;XCTAssertThrowsError(try StatefulNormalizationLearningSubmission(graph:bad,parameters:parameters(t),gradientNodeID:"op",task:t))
        XCTAssertNoThrow(try StatefulNormalizationLearningSubmission.validatedGraph(.init(name:"Pending",nodes:[],edges:[]),task:t,complete:false))
    }
    func testDraftProjectionDoesNotWeakenStrictSubmittedStateOrHideUnknownValues()throws {
        let task=StatefulNormalizationLearningTask.bn1Remembered,bank=parameters(task)
        var g=graph(task);g.nodes[1].config["bias"] = .bool(false)
        XCTAssertThrowsError(try StatefulNormalizationLearningSubmission(graph:g,parameters:bank,gradientNodeID:"op",task:task))
        let active=try StatefulNormalizationLearningSubmission.fromDraft(graph:g,parameters:bank,gradientNodeID:"op",task:task)
        XCTAssertEqual(Set(active.parameters["op"]!.keys),["weight"])
        var raw=try JSONSerialization.jsonObject(with:active.encoded()) as! [String:Any]
        raw["parameters"]=["op":["weight":["shape":[3],"values":[0.5,-1,2]],"bias":["shape":[3],"values":[1,-0.5,0.25]]]]
        XCTAssertThrowsError(try StatefulNormalizationLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw),task:task))
        g.nodes[1].config["affine"] = .bool(false)
        XCTAssertThrowsError(try StatefulNormalizationLearningSubmission(graph:g,parameters:bank,gradientNodeID:"op",task:task))
        let disabled=try StatefulNormalizationLearningSubmission.fromDraft(graph:g,parameters:bank,gradientNodeID:"op",task:task)
        XCTAssertTrue(disabled.parameters.isEmpty);XCTAssertNil(disabled.gradientNodeID)
        var bad=bank;bad["op"]?["running_mean"] = .init(shape:[3],values:[0,0,0])
        XCTAssertThrowsError(try StatefulNormalizationLearningSubmission.fromDraft(graph:g,parameters:bad,gradientNodeID:"op",task:task))
    }
}
