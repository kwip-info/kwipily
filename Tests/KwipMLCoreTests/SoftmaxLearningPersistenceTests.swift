import XCTest
@testable import KwipMLCore

final class SoftmaxLearningPersistenceTests:XCTestCase {
    func testDraftAndReceiptRoundTripRemainTaskBound()throws {
        let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"s",type:"softmax",config:["dim":-1]),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"s"),.init(source:"s",target:"o")])
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:"softmax-last-axis",mode:.assessment)
        draft.graph=graph
        let restored=try TensorLearningDraft.decode(draft.encoded())
        XCTAssertEqual(restored,draft)
        let submission=try restored.submission().encoded()
        let plan=try SoftmaxAssessment.plan(axis:-1,runtimeVersion:"test",seed:42)
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:100)
        let receipt=try TensorLearningReceipt(projectID:draft.projectID,operation:draft.operation,mode:draft.mode,submissionData:submission,plan:plan,evidence:evidence)
        let reopened=try TensorLearningReceipt.decode(receipt.encoded())
        XCTAssertEqual(try reopened.verifiedReplay(expectedProjectID:draft.projectID),plan)
        var wrong=reopened;wrong.operation="softmax-first-sample-axis"
        XCTAssertThrowsError(try wrong.verifiedReplay(expectedProjectID:draft.projectID))
        draft.operation="transpose"
        XCTAssertThrowsError(try draft.encoded(),"Existing transform lessons cannot silently gain Softmax")
        XCTAssertThrowsError(try TensorLearningSubmission(graph:graph))
    }
    func testCanvasRegistryAndValidationAreTaskScoped()throws {
        let registry=TensorLearningCanvas.registry(operation:"softmax-last-axis")
        XCTAssertEqual(registry.component(type:"softmax"),try ComponentRegistry.builtIn().component(type:"softmax"))
        XCTAssertNil(TensorLearningCanvas.registry.component(type:"softmax"))
        var graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"s",type:"softmax",config:["dim":-1]),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"s"),.init(source:"s",target:"o")])
        let valid=TensorLearningCanvas.inspect(graph,allowSoftmax:true)
        XCTAssertTrue(valid.isValid)
        XCTAssertEqual(valid.tensorLayouts["o"]?.displayDimensions,["N","2","3"])
        XCTAssertFalse(TensorLearningCanvas.inspect(graph).isValid)
        graph.nodes[1].config["dim"] = -3
        let invalid=TensorLearningCanvas.inspect(graph,allowSoftmax:true)
        XCTAssertFalse(invalid.isValid)
        XCTAssertNil(invalid.tensorLayouts["o"])
    }

}
