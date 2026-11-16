import XCTest
@testable import KwipMLCore

final class SchedulerLearningTests:XCTestCase {
    func graph(_ task:SchedulerLearningTask)->GraphDocument {
        .init(schemaVersion:7,name:"Schedule adjustments",nodes:[.init(id:"schedule",type:task.componentID,name:"Rate schedule",x:0,y:0,config:task.configuration),.init(id:"optimizer",type:"optimizer_sgd",name:"Supplied SGD",x:0,y:100,config:SchedulerLearningTask.suppliedOptimizerValues),.init(id:"train",type:"train",name:"Supplied context",x:200,y:0,config:[:])],edges:[.init(id:"scheduler-wire",source:"schedule",sourcePort:"out",target:"train",targetPort:"scheduler"),.init(id:"optimizer-wire",source:"optimizer",sourcePort:"out",target:"train",targetPort:"optimizer")])
    }
    func testIndependentUsedRatesTicksAndResumeSuffix()throws {
        for task in SchedulerLearningTask.allCases {
            let p=try SchedulerAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41)
            XCTAssertEqual(p.fixtures.count,4);let expected=p.fixtures[0].expected
            XCTAssertEqual(expected["usedRates"]?.shape,[8,1]);XCTAssertEqual(expected["updatedParameters"]?.shape,[8,3])
            XCTAssertEqual(expected["scheduleTicks"]?.values,task == .oneCycle ? [1,2,3,4,5,6,7,8]:[0,1,1,2,2,3,3,4])
            XCTAssertEqual(expected["resumedParameters"]?.values,Array(try XCTUnwrap(expected["updatedParameters"]?.values).suffix(12)))
            let used=try XCTUnwrap(expected["usedRates"]?.values)
            if task == .step {XCTAssertEqual(used,[0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05])}
            if task == .linear {XCTAssertEqual(used[0],0.025,accuracy:1e-12);XCTAssertEqual(used[2],0.05,accuracy:1e-12);XCTAssertEqual(used[6],0.1,accuracy:1e-12)}
            if task == .plateau {XCTAssertEqual(used[4],0.05,accuracy:1e-12);XCTAssertEqual(used[6],0.05,accuracy:1e-12)}
            if task == .oneCycle {XCTAssertEqual(used[0],0.01,accuracy:1e-12);XCTAssertEqual(used[3],0.1,accuracy:1e-12);XCTAssertEqual(used[7],0.001,accuracy:1e-12)}
        }
    }
    func testStrictFixedPublicContextAndPartialWiring()throws {
        var g=graph(.step);g.edges=[]
        XCTAssertNoThrow(try SchedulerLearningSubmission.validatedGraph(g,task:.step,complete:false));XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
        g=graph(.step);g.nodes[1].config["lr"] = .number(0.2);XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
        g=graph(.step);g.nodes[1].config.removeValue(forKey:"momentum");XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
        g=graph(.step);g.nodes[2].config=["seed":41];XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
        g=graph(.step);g.edges[0].targetPort="optimizer";XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
        g=graph(.step);g.edges[1].id=g.edges[0].id;XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
        g=graph(.step);g.schemaVersion=6;XCTAssertThrowsError(try SchedulerLearningSubmission(graph:g,task:.step))
    }
    func testAllFiveRouteDraftsAndPendingTextWithoutForwardTensorSubmission()throws {
        for task in SchedulerLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue));XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
            XCTAssertEqual(draft.graph.schemaVersion,7);draft.graph=graph(task)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:41).id,"assessment."+task.rawValue)
            let bytes=try draft.submissionData();XCTAssertEqual(try SchedulerLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertTrue(SchedulerLearningCanvas.inspect(draft.graph,task:task).isValid);XCTAssertTrue(SchedulerLearningCanvas.inspect(draft.graph,task:task).tensorLayouts.isEmpty)
            draft.pendingFields=["schedule.gamma":"0.3"];XCTAssertThrowsError(try draft.submissionData());XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()).pendingFields,draft.pendingFields)
            XCTAssertThrowsError(try draft.submission())
        }
    }
    func testColdReceiptsHaveSourceIdentityAndNoPrivateProjection()throws {
        for task in SchedulerLearningTask.allCases {
            let plan=try SchedulerAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173),bytes=try SchedulerLearningSubmission(graph:graph(task),task:task).encoded()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:24)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let id=UUID().uuidString,r=try TensorLearningReceipt(projectID:id,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
                let cold=try TensorLearningReceipt.decode(r.encoded());XCTAssertEqual(cold.attempt.grade.status,.passed)
                XCTAssertEqual(cold.schedulerFormativeProjection(expectedProjectID:id,currentSubmission:bytes,currentMode:mode) == nil,mode == .assessment)
                XCTAssertNil(cold.schedulerFormativeProjection(expectedProjectID:id,currentSubmission:bytes,currentMode:.assessment))
                XCTAssertNil(cold.schedulerFormativeProjection(expectedProjectID:UUID().uuidString,currentSubmission:bytes,currentMode:mode))
            }
        }
    }
}
