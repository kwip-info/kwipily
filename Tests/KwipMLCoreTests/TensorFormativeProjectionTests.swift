import XCTest
@testable import KwipMLCore

final class TensorFormativeProjectionTests:XCTestCase {
    func testProjectionRequiresMatchingSourceAndNonAssessmentOrigin()throws {
        var receipt=try TensorLearningReceiptTests().fixture()
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.guided))
        receipt.attempt.mode = .practice
        let projection=try XCTUnwrap(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.guided))
        XCTAssertEqual(projection.fixtures.count,4)
        XCTAssertEqual(projection.fixtures[0].input.shape,[1,2,3])
        XCTAssertEqual(projection.fixtures[0].measured["output"]?.shape,[1,3,2])
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.assessment))
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:UUID().uuidString,currentSubmission:receipt.submissionData,currentMode:.guided))
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:Data("changed".utf8),currentMode:.guided))
        receipt.attempt.grade.summary="changed"
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.guided))
    }
    func testObservationSlicesRespectBatchBoundariesAndShape()throws {
        let tensor=LearningTensor(shape:[2,2,3],values:(0..<12).map(Double.init))
        XCTAssertEqual(tensor.batchObservation(at:0),LearningTensor(shape:[2,3],values:[0,1,2,3,4,5]))
        XCTAssertEqual(tensor.batchObservation(at:1),LearningTensor(shape:[2,3],values:[6,7,8,9,10,11]))
        XCTAssertNil(tensor.batchObservation(at:2));XCTAssertNil(tensor.batchObservation(at:-1))
        XCTAssertNil(LearningTensor(shape:[2,2,3],values:[1]).batchObservation(at:0))
        XCTAssertEqual(LearningTensor(shape:[1,3],values:[1,2,3]).batchObservation(at:0)?.shape,[3])
    }
}
