import XCTest
@testable import KwipMLCore

final class EmbeddingBagFormativeProjectionTests:XCTestCase {
    private func receipt(_ task:EmbeddingBagLearningTask,mode:LearningMode = .practice)throws->TensorLearningReceipt {
        let plan=try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"test",seed:11)
        let bytes=try EmbeddingBagLearningSubmission(graph:EmbeddingBagExamples.lessonGraph(task:task)).encoded()
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:1000)
        return try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
    }
    func testSixProjectionsPreserveRealSourceRowsLengthsWeightsAndMeasuredInfluence()throws {
        for task in EmbeddingBagLearningTask.allCases {
            let receipt=try receipt(task),plan=try receipt.verifiedReplay(expectedProjectID:receipt.projectID)
            let projected=try XCTUnwrap(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.practice))
            XCTAssertEqual(projected.fixtures.count,4)
            for (shown,original) in zip(projected.fixtures,plan.fixtures) {
                XCTAssertEqual(shown.input,original.inputs["sourceRows"])
                XCTAssertEqual(shown.inputLabel,"Category IDs by observation group")
                XCTAssertEqual(shown.sourceLengths,original.inputs["lengths"])
                XCTAssertEqual(shown.sourceWeights,original.inputs["sourceWeights"])
                XCTAssertEqual(shown.sourceTable,original.inputs["table"])
                XCTAssertEqual(shown.measured["sourceWeightsGradient"],receipt.evidence.fixtures.first {$0.fixtureID == shown.id}?.tensors["sourceWeightsGradient"])
                XCTAssertNil(shown.measured["inputGradient"]);XCTAssertNil(shown.measured["indicesGradient"]);XCTAssertNil(shown.measured["offsetsGradient"])
                XCTAssertTrue(shown.sharedParameters.isEmpty,"Source lengths and weights are not learned table parameters")
                let choices=TensorMeasuredOutput.available(in:shown)
                XCTAssertEqual(choices.contains {$0.rawValue == "sourceWeightsGradient"},try task.configuration.usesPerSampleWeights)
            }
        }
    }
    func testWeightInfluenceUsesSourceCoordinatesIncludingUnusedCells()throws {
        let receipt=try receipt(.packedWeightedSum)
        let projected=try XCTUnwrap(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.guided))
        let empty=try XCTUnwrap(projected.fixtures.last)
        XCTAssertEqual(empty.sourceLengths?.values,[0,0,0])
        XCTAssertEqual(empty.measured["output"]?.shape,[3,3])
        XCTAssertEqual(empty.measured["sourceWeightsGradient"]?.values,Array(repeating:0,count:9))
        let choice=try XCTUnwrap(TensorMeasuredOutput.available(in:empty).first {$0.rawValue == "sourceWeightsGradient"})
        XCTAssertFalse(choice.isParameter)
        XCTAssertEqual(choice.title,"Item-weight influence")
        XCTAssertEqual(choice.displayTensor(try XCTUnwrap(empty.measured["sourceWeightsGradient"]),observation:1)?.shape,[3])
    }
    func testAssessmentOriginsCurrentAssessmentStaleSourcesAndChangedEvidenceStayHidden()throws {
        var receipt=try receipt(.packedWeightedSum)
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.assessment))
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:UUID().uuidString,currentSubmission:receipt.submissionData,currentMode:.practice))
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:Data("stale".utf8),currentMode:.practice))
        let privateReceipt=try self.receipt(.packedWeightedSum,mode:.assessment)
        XCTAssertNil(privateReceipt.formativeProjection(expectedProjectID:privateReceipt.projectID,currentSubmission:privateReceipt.submissionData,currentMode:.guided))
        receipt.evidence.fixtures[0].tensors["output"]!.values[0]+=1
        XCTAssertNil(receipt.formativeProjection(expectedProjectID:receipt.projectID,currentSubmission:receipt.submissionData,currentMode:.practice))
    }
}
