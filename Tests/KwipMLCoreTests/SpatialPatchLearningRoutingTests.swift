import XCTest
@testable import KwipMLCore
final class SpatialPatchLearningRoutingTests:XCTestCase {
    func draft(_ task:SpatialPatchLearningTask)->TensorLearningDraft {var d=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided);d.graph=SpatialPatchLearningTests().graph(task);return d}
    func testAllFourPlansDraftsAndPalettesRouteWithoutGenericEnvelopeEscape()throws {
        for task in SpatialPatchLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue));XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let d=draft(task),bytes=try d.submissionData();XCTAssertEqual(try SpatialPatchLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(d.encoded()),d);XCTAssertThrowsError(try d.submission())
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:7),try SpatialPatchAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:7))
            XCTAssertEqual(TensorLearningCanvas.registry(operation:task.rawValue).components.filter {$0.id == task.componentID}.count,1)
        }
    }
    func testPendingPatchGeometryAndStateOwnershipStayStrict()throws {
        var d=draft(.unfoldSpaced);d.pendingFields=["op.stride":"1, 1"]
        XCTAssertNoThrow(try d.validate());XCTAssertThrowsError(try d.submissionData());d.pendingFields=[:]
        d.parameters=["op":["weight":.init(shape:[1],values:[1])]];XCTAssertThrowsError(try d.validate());d.parameters=nil
        d.gradientNodeID="op";XCTAssertThrowsError(try d.validate());d.gradientNodeID=nil
        d.graph.edges=[];XCTAssertNoThrow(try d.validate());XCTAssertThrowsError(try d.submissionData())
    }
    func testColdReceiptReplayRetainsExactTaskAndPrivateProjectionRules()throws {
        for task in SpatialPatchLearningTask.allCases {
            let d=draft(task),bytes=try d.submissionData(),plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:23)
            // Contract replay fixture only; actual measurements are independently tested in Python/App.
            let e=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:4096)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:d.projectID,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:e)
                XCTAssertEqual(try TensorLearningReceipt.decode(receipt.encoded()).verifiedReplay(expectedProjectID:d.projectID),plan)
                XCTAssertEqual(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:mode) == nil,mode == .assessment)
                if mode == .assessment {XCTAssertNil(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:.guided))}
            }
        }
    }
    func testIncompleteFoldDefaultsPersistButCompleteSubmissionRefusesUntilConfigured()throws {
        for task in [SpatialPatchLearningTask.foldOverlap,.foldHoles] {
            var d=draft(task)
            d.graph.nodes[1].config=try SpatialPatchConfiguration(componentID:"fold",values:[:]).values
            XCTAssertNoThrow(try d.validate(),"Connect before configuring; preserve a safe repairable draft")
            XCTAssertEqual(try TensorLearningDraft.decode(d.encoded()).graph,d.graph)
            XCTAssertThrowsError(try d.submissionData(),"Actual incoming column count must match configured output geometry before assessment")
            d.graph.nodes[1].config=task.configuration
            XCTAssertNoThrow(try d.submissionData())
        }
    }

    func testLessonInspectorUsesFloat64ShapeOnlyAndDiagnosesUnconfiguredFold()throws {
        for task in SpatialPatchLearningTask.allCases {
            let d=draft(task),result=TensorLearningCanvas.inspect(d.graph,spatialPatchTask:task)
            XCTAssertTrue(result.issues.isEmpty,result.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(result.tensorContracts["out"]?.layout.sample.dimensions,task.outputSampleShape)
            XCTAssertEqual(result.tensorContracts["out"]?.dtype,.float64)
        }
        var d=draft(.foldOverlap);d.graph.nodes[1].config=try SpatialPatchConfiguration(componentID:"fold",values:[:]).values
        let bad=TensorLearningCanvas.inspect(d.graph,spatialPatchTask:.foldOverlap)
        XCTAssertFalse(bad.issues.isEmpty);XCTAssertNil(bad.tensorContracts["out"])
    }

}
