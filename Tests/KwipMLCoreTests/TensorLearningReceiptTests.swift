import XCTest
@testable import KwipMLCore

final class TensorLearningReceiptTests:XCTestCase {
    func fixture() throws -> TensorLearningReceipt {
        let plan=try TensorTransformAssessment.plan(operation:"transpose",runtimeVersion:"test",seed:123)
        let graph=GraphDocument(nodes:[GraphNode(id:"i",type:"input"),GraphNode(id:"t",type:"transpose",config:["dim0":1,"dim1":2]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"t"),GraphEdge(source:"t",target:"o")])
        let bytes=try TensorLearningSubmission(graph:graph).encoded()
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:12)
        return try TensorLearningReceipt(projectID:UUID().uuidString,operation:"transpose",mode:.assessment,submissionData:bytes,plan:plan,evidence:evidence)
    }
    func testBoundReplayRoundTripAndRejectsChangedIdentityOrEvidence() throws {
        let receipt=try fixture()
        XCTAssertEqual(try receipt.verifiedReplay(expectedProjectID:receipt.projectID).id,"assessment.tensor-transpose")
        let decoded=try TensorLearningReceipt.decode(receipt.encoded())
        XCTAssertEqual(decoded.attempt,receipt.attempt)
        XCTAssertThrowsError(try decoded.verifiedReplay(expectedProjectID:UUID().uuidString))
        var changed=receipt;changed.fixtureSeed+=1
        XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
        changed=receipt;changed.operation="reshape"
        XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
        changed=receipt;changed.attempt.assessmentVersion.patch=1
        XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
        changed=receipt;changed.evidence.fixtures[0].tensors["output"]!.values[0]+=1
        XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
        changed=receipt;changed.submissionData=Data("{}".utf8)
        XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
        changed=receipt;changed.attempt.grade.summary="fabricated success"
        XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
    }
    func testReceiptDecoderRejectsUnknownEnvelopeAndOversizedData()throws {
        let receipt=try fixture()
        var object=try JSONSerialization.jsonObject(with:receipt.encoded()) as! [String:Any]
        object["trusted"]=true
        XCTAssertThrowsError(try TensorLearningReceipt.decode(JSONSerialization.data(withJSONObject:object)))
        XCTAssertThrowsError(try TensorLearningReceipt.decode(Data(repeating:32,count:2_000_001)))
    }
}
