import XCTest
@testable import KwipMLCore

final class OptimizerLearningRoutingTests:XCTestCase {
    func graph(_ task:OptimizerLearningTask)->GraphDocument {
        var g=GraphDocument(name:"Training recipe");g.schemaVersion=6
        g.nodes=[.init(id:"recipe",type:task.componentID,name:"Update settings",x:0,y:0,config:task.configuration),.init(id:"train",type:"train",name:"Supplied context",x:200,y:0,config:[:])]
        g.edges=[.init(id:"recipe-edge",source:"recipe",sourcePort:"out",target:"train",targetPort:"optimizer")];return g
    }
    func testFiveRecipesRouteThroughDraftAndCanonicalSubmission()throws {
        for task in OptimizerLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:41)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            var d=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided);d.graph=graph(task)
            try d.validate();let data=try d.submissionData()
            XCTAssertEqual(try OptimizerLearningSubmission.decode(data,task:task).encoded(),data)
            XCTAssertTrue(OptimizerLearningCanvas.inspect(d.graph,task:task).isValid)
            XCTAssertTrue(OptimizerLearningCanvas.inspect(d.graph,task:task).tensorLayouts.isEmpty)
            d.pendingFields=["recipe.lr":"0.2"];XCTAssertThrowsError(try d.submissionData())
        }
    }
    func testIndependentStepsKeepNoGradientParametersAndTypedContextStrict()throws {
        let reference=try OptimizerLessonReference(initial:.init(shape:[3],values:[1,-2,0.5]),gradients:.init(shape:[4,3],values:[0.5,-0.25,1,-1,0.75,-0.5,0.25,1.5,0.75,1,-0.5,-1]),active:.init(shape:[4,2],values:[1,1,1,0,1,1,0,1]),task:.sgd)
        let values=try XCTUnwrap(reference.expected["updatedParameters"]?.values)
        XCTAssertEqual(values[0],0.94,accuracy:1e-12);XCTAssertEqual(values[1],-1.955,accuracy:1e-12)
        XCTAssertEqual(values[2],values[5]);XCTAssertEqual(values[6],values[9]);XCTAssertEqual(values[7],values[10])
        XCTAssertEqual(reference.expected["updateCounts"]?.values,[1,1,2,1,3,2,3,3])
        var g=graph(.sgd);g.edges[0].targetPort="in";XCTAssertThrowsError(try OptimizerLearningSubmission(graph:g,task:.sgd))
        g=graph(.sgd);g.nodes[1].config=["seed":41];XCTAssertThrowsError(try OptimizerLearningSubmission(graph:g,task:.sgd))
        g=graph(.sgd);g.edges=[];XCTAssertNoThrow(try OptimizerLearningSubmission.validatedGraph(g,task:.sgd,complete:false));XCTAssertThrowsError(try OptimizerLearningSubmission(graph:g,task:.sgd))
    }
    func testColdReceiptReplaysMeasuredUpdatesAndNeverProjectsPrivateAnswers()throws {
        for task in OptimizerLearningTask.allCases {
            let plan=try OptimizerAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173),data=try OptimizerLearningSubmission(graph:graph(task),task:task).encoded()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(data),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:12)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let id=UUID().uuidString,r=try TensorLearningReceipt(projectID:id,operation:task.rawValue,mode:mode,submissionData:data,plan:plan,evidence:evidence)
                let cold=try TensorLearningReceipt.decode(r.encoded());XCTAssertEqual(cold.attempt.grade.status,.passed)
                XCTAssertEqual(cold.optimizerFormativeProjection(expectedProjectID:id,currentSubmission:data,currentMode:mode) == nil,mode == .assessment)
                XCTAssertNil(cold.optimizerFormativeProjection(expectedProjectID:id,currentSubmission:data,currentMode:.assessment))
            }
        }
    }
}
