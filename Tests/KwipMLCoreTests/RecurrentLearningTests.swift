import XCTest
@testable import KwipMLCore

final class RecurrentLearningTests:XCTestCase {
    func graph(_ task:RecurrentLearningTask)->GraphDocument {
        var nodes=task.sourceRoles.map {GraphNode(id:$0,type:"input",name:$0,config:["fixture":.string($0)])}
        nodes.append(.init(id:"op",type:task.componentID,name:task.componentTitle,config:task.configuration))
        nodes+=task.outputRoles.map {GraphNode(id:"output-"+$0,type:"output",name:$0,config:["role":.string($0)])}
        let edges=task.sourceRoles.map {GraphEdge(id:"in-"+$0,source:$0,sourcePort:"out",target:"op",targetPort:$0=="input" ? "in":$0)}+task.outputRoles.map {GraphEdge(id:"out-"+$0,source:"op",sourcePort:$0=="result" ? "out":$0,target:"output-"+$0,targetPort:"in")}
        return .init(schemaVersion:2,name:"Remember measurements",nodes:nodes,edges:edges)
    }
    func testIndependentPositiveRNNRecurrenceAndWeightedAllRoleInfluences()throws {
        let task=RecurrentLearningTask.rnn
        var inputs:[String:LearningTensor]=["input":.init(shape:[1,3,2],values:[1,2,3,4,5,6]),"hidden":.init(shape:[1,1,2],values:[0.5,1]),"upstream.result":.init(shape:[1,3,2],values:Array(repeating:1,count:6)),"upstream.hidden":.init(shape:[1,1,2],values:[1,1])]
        for (name,shape) in task.parameterShapes {inputs["parameter."+name] = .init(shape:shape,values:shape.count==2 ? [1,0,0,1]:[0,0])}
        let expected=try RecurrentLessonReference(inputs:inputs,task:task).expected
        XCTAssertEqual(expected["result"]?.values,[1.5,3,4.5,7,9.5,13]);XCTAssertEqual(expected["hidden"]?.values,[9.5,13])
        for (got,want) in zip(expected["inputGradient"]!.values,[4.0,4,3,3,2,2]) {XCTAssertEqual(got,want,accuracy:1e-7)}
        for got in expected["hiddenGradient"]!.values {XCTAssertEqual(got,4,accuracy:1e-7)}
        XCTAssertEqual(Set(expected.keys),Set(task.outputRoles+task.sourceRoles.map {$0+"Gradient"}+task.parameterShapes.keys.map {"parameter."+$0+"Gradient"}))
    }
    func testSixIndependentFourChangedFixturePlansAndDistinctOutputRoles()throws {
        for task in RecurrentLearningTask.allCases {
            let a=try RecurrentAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:17),b=try RecurrentAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41)
            XCTAssertEqual(a.fixtures.count,4);XCTAssertNotEqual(a.fixtures,b.fixtures)
            for f in a.fixtures {XCTAssertTrue(f.expected.values.allSatisfy(\.wellFormed));XCTAssertTrue(Set(task.outputRoles).isSubset(of:f.expected.keys));for key in task.sourceRoles+task.parameterShapes.keys.map({"parameter."+$0}) {XCTAssertEqual(f.expected[key+"Gradient"]?.shape,f.inputs[key]?.shape)}}
            XCTAssertTrue(task.instructions.contains("No code"));XCTAssertTrue(task.instructions.contains("State is information"))
        }
    }
    func testStrictNamedRolesCanonicalRoundtripPartialDraftsAndCheatRefusal()throws {
        for task in RecurrentLearningTask.allCases {
            let source=graph(task),value=try RecurrentLearningSubmission(graph:source,task:task),bytes=try value.encoded()
            XCTAssertEqual(try RecurrentLearningSubmission.decode(bytes,task:task),value)
            XCTAssertNoThrow(try RecurrentLearningSubmission.validatedGraph(.init(),task:task,complete:false))
            var missing=source;missing.edges.removeLast();XCTAssertThrowsError(try RecurrentLearningSubmission(graph:missing,task:task))
            var role=source;role.nodes[role.nodes.count-1].config["role"] = .string("result");XCTAssertThrowsError(try RecurrentLearningSubmission(graph:role,task:task))
            var order=source;order.edges[0].source="hidden";XCTAssertThrowsError(try RecurrentLearningSubmission(graph:order,task:task))
            var port=source;port.edges[port.edges.count-1].sourcePort="in";XCTAssertThrowsError(try RecurrentLearningSubmission(graph:port,task:task))
            var json=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any]);json["weights"]=[];XCTAssertThrowsError(try RecurrentLearningSubmission.decode(JSONSerialization.data(withJSONObject:json),task:task))
        }
    }
    func testSharedTaskDraftSubmissionAndPrivateReceiptProjectionRouting()throws {
        for task in RecurrentLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue));XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:17);XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided);draft.graph=graph(task)
            let canonical=try RecurrentLearningSubmission(graph:graph(task),task:task).encoded();XCTAssertEqual(try draft.submissionData(),canonical)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            draft.pendingFields["op.hidden_size"]="3";XCTAssertThrowsError(try draft.submissionData())
        }
    }
    func testSyntheticColdReceiptSourceIdentityAndPrivateProjectionGuards()throws {
        for task in RecurrentLearningTask.allCases {
            let plan=try RecurrentAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173),bytes=try RecurrentLearningSubmission(graph:graph(task),task:task).encoded()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:48)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let id=UUID().uuidString,r=try TensorLearningReceipt(projectID:id,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
                let cold=try TensorLearningReceipt.decode(r.encoded());XCTAssertEqual(cold.attempt.grade.status,.passed)
                XCTAssertEqual(cold.recurrentFormativeProjection(expectedProjectID:id,currentSubmission:bytes,currentMode:mode)==nil,mode == .assessment)
                XCTAssertNil(cold.recurrentFormativeProjection(expectedProjectID:id,currentSubmission:bytes,currentMode:.assessment))
                XCTAssertNil(cold.recurrentFormativeProjection(expectedProjectID:UUID().uuidString,currentSubmission:bytes,currentMode:mode))
                var changed=graph(task);changed.name="Changed source";let other=try RecurrentLearningSubmission(graph:changed,task:task).encoded()
                XCTAssertNil(cold.recurrentFormativeProjection(expectedProjectID:id,currentSubmission:other,currentMode:mode))
            }
        }
    }

}
