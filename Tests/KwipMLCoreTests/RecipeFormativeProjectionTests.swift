import XCTest
@testable import KwipMLCore

final class RecipeFormativeProjectionTests:XCTestCase {
    private func receipt(operation:String,mode:LearningMode)throws->TensorLearningReceipt {
        let plan=try TensorLearningTask.plan(operation:operation,runtimeVersion:"test",seed:29)
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:operation,mode:mode)
        if let task=DataLoadingLearningTask(rawValue:operation) {draft.graph=try DataLoadingExamples.lessonGraph(task:task)}
        let bytes=try draft.submissionData()
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:1000)
        return try .init(projectID:draft.projectID,operation:operation,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
    }
    func testAllTenPublicReceiptsShowActualSourceFieldsWithoutFabricatedUpstream()throws {
        for operation in DataLoadingLearningTask.allCases.map(\.rawValue)+ProgramExportLearningTask.allCases.map(\.rawValue) {
            let receipt=try receipt(operation:operation,mode:.practice)
            let projection=try XCTUnwrap(receipt.recipeFormativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.practice))
            let plan=try receipt.verifiedReplay(expectedProjectID:receipt.projectID)
            XCTAssertEqual(projection.operation,operation)
            XCTAssertEqual(projection.fixtures.count,4)
            XCTAssertEqual(projection.fixtures[0].inputs,plan.fixtures[0].inputs)
            XCTAssertNil(projection.fixtures[0].inputs["upstream"])
            XCTAssertEqual(projection.fixtures[0].measured,receipt.evidence.fixtures[0].tensors)
            XCTAssertEqual(projection.fixtures[0].expected,plan.fixtures[0].expected)
        }
    }
    func testAssessmentOriginModeSwitchStaleSourceProjectAndEvidenceRemainPrivate()throws {
        for operation in [DataLoadingLearningTask.continuation.rawValue,ProgramExportLearningTask.sizeRange.rawValue] {
            let privateReceipt=try receipt(operation:operation,mode:.assessment)
            for mode in [LearningMode.guided,.practice,.assessment] {
                XCTAssertNil(privateReceipt.recipeFormativeProjection(expectedProjectID:privateReceipt.projectID,currentSubmission:privateReceipt.submissionData,currentMode:mode))
            }
            var publicReceipt=try receipt(operation:operation,mode:.guided)
            let id=publicReceipt.projectID,bytes=publicReceipt.submissionData
            XCTAssertNil(publicReceipt.recipeFormativeProjection(expectedProjectID:id,currentSubmission:bytes,currentMode:.assessment))
            XCTAssertNil(publicReceipt.recipeFormativeProjection(expectedProjectID:UUID().uuidString,currentSubmission:bytes,currentMode:.guided))
            XCTAssertNil(publicReceipt.recipeFormativeProjection(expectedProjectID:id,currentSubmission:bytes+Data([32]),currentMode:.guided))
            publicReceipt.fixtureSeed &+= 1
            XCTAssertNil(publicReceipt.recipeFormativeProjection(expectedProjectID:id,currentSubmission:bytes,currentMode:.guided))
        }
    }
}
