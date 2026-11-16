import XCTest
@testable import KwipMLCore
final class StatefulNormalizationLearningRoutingTests:XCTestCase {
    func draft(_ task:StatefulNormalizationLearningTask)->TensorLearningDraft {var d=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided);d.graph=StatefulNormalizationLearningTests().graph(task);if task.affine {d.parameters=StatefulNormalizationLearningTests().parameters(task);d.gradientNodeID="op"};return d}
    func testTenTaskPlansDraftsAndPalettesRequireTheReviewedStateEnvelope()throws {
        for task in StatefulNormalizationLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue));XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let d=draft(task),bytes=try d.submissionData();XCTAssertEqual(try StatefulNormalizationLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(d.encoded()),d);XCTAssertThrowsError(try d.submission())
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:7),try StatefulNormalizationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:7))
            XCTAssertEqual(TensorLearningCanvas.registry(operation:task.rawValue).components.filter {$0.id==task.componentID}.count,1)
        }
    }
    func testPartialLearnedParametersPendingControlsAndUnknownStateStayBounded()throws {
        var d=draft(.bn1Remembered);d.parameters=nil;d.gradientNodeID=nil
        XCTAssertNoThrow(try d.validate());XCTAssertEqual(try TensorLearningDraft.decode(d.encoded()),d);XCTAssertThrowsError(try d.submissionData())
        d=draft(.bn1Remembered);d.pendingFields=["op.momentum":"0.75"];XCTAssertNoThrow(try d.validate());XCTAssertThrowsError(try d.submissionData());d.pendingFields=[:]
        d.parameters?["op"]?["weight"] = .init(shape:[1,3],values:[1,1,1]);XCTAssertThrowsError(try d.validate())
        d=draft(.bn1Remembered);d.gradientNodeID="x";XCTAssertThrowsError(try d.validate())
        d=draft(.bn1Current);d.parameters=["op":["running_mean":.init(shape:[3],values:[0,0,0])]];XCTAssertThrowsError(try d.validate())
        d=draft(.bn1Remembered);d.graph.edges=[];XCTAssertNoThrow(try d.validate());XCTAssertThrowsError(try d.submissionData())
    }
    func testSyntheticColdContractReplayAndPrivateProjectionRemainTaskBound()throws {
        for task in StatefulNormalizationLearningTask.allCases {
            let d=draft(task),bytes=try d.submissionData(),plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:23)
            // Synthetic Core contract fixture; real worker measurements remain Python/App evidence.
            let e=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:4096)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:d.projectID,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:e)
                XCTAssertEqual(try TensorLearningReceipt.decode(receipt.encoded()).verifiedReplay(expectedProjectID:d.projectID),plan)
                XCTAssertEqual(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:mode)==nil,mode == .assessment)
                if mode == .assessment {XCTAssertNil(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:.guided))}
            }
        }
    }
    func testShapeOnlyLessonInspectionPreservesFloat64AndDiagnosesWrongChannels()throws {
        for task in StatefulNormalizationLearningTask.allCases {
            let d=draft(task),r=TensorLearningCanvas.inspect(d.graph,statefulNormalizationTask:task)
            XCTAssertTrue(r.issues.isEmpty,r.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(r.tensorContracts["out"]?.layout.sample.dimensions,task.sampleShape)
            XCTAssertEqual(r.tensorContracts["out"]?.dtype,.float64)
        }
        var d=draft(.bn1Current);d.graph.nodes[1].config["num_features"] = .number(2)
        let bad=TensorLearningCanvas.inspect(d.graph,statefulNormalizationTask:.bn1Current)
        XCTAssertFalse(bad.issues.isEmpty);XCTAssertNil(bad.tensorContracts["out"])
    }
}
