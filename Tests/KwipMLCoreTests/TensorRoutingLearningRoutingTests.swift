import XCTest
@testable import KwipMLCore

final class TensorRoutingLearningRoutingTests:XCTestCase {
    private func draft(_ task:TensorRoutingLearningTask)->TensorLearningDraft {
        var d=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
        d.graph=TensorRoutingLearningTests().graph(task);return d
    }
    func testSixTasksRouteCanonicalDraftsPlansAndPalettes()throws {
        for task in TensorRoutingLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue));XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let d=draft(task),bytes=try d.submissionData()
            XCTAssertEqual(try TensorRoutingLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(d.encoded()),d);XCTAssertThrowsError(try d.submission())
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:19),try TensorRoutingAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:19))
            let palette=TensorLearningCanvas.registry(operation:task.rawValue)
            XCTAssertEqual(palette.components.filter {$0.id == task.componentID}.count,1)
            if task.kind == "split" {XCTAssertEqual(palette.components.filter {$0.id == "concat"}.count,1)}
        }
    }
    func testPendingSourceAddressDraftCannotGainStateOrInventActualPart()throws {
        var d=draft(.chunk);d.pendingFields=["op.chunks":"4"]
        XCTAssertNoThrow(try d.validate());XCTAssertThrowsError(try d.submissionData());d.pendingFields=[:]
        d.parameters=["op":["weight":.init(shape:[1],values:[1])]];XCTAssertThrowsError(try d.validate())
        d.parameters=nil;d.graph.edges[1].sourcePort="part5";XCTAssertThrowsError(try d.validate())
        d.graph.edges=[];XCTAssertNoThrow(try d.validate(),"Unconnected drafts remain repairable")
        XCTAssertThrowsError(try d.submissionData())
    }
    func testSixReceiptsReplayColdAndPrivateProjectionStaysUnavailable()throws {
        for task in TensorRoutingLearningTask.allCases {
            let d=draft(task),bytes=try d.submissionData(),plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:23)
            // Replay contract fixture only. Real measurements are tested separately in worker/CLI/App.
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:512)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:d.projectID,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
                XCTAssertEqual(try TensorLearningReceipt.decode(receipt.encoded()).verifiedReplay(expectedProjectID:d.projectID),plan)
                XCTAssertEqual(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:mode) == nil,mode == .assessment)
                if mode == .assessment {XCTAssertNil(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:.guided))}
                var wrong=receipt;wrong.operation=task == .chunk ? TensorRoutingLearningTask.gather.rawValue:TensorRoutingLearningTask.chunk.rawValue
                XCTAssertThrowsError(try wrong.verifiedReplay(expectedProjectID:d.projectID))
            }
        }
    }
    func testCanvasUsesNamedPartContractsAndInt64SelectorRoleWithoutChangingProductionDtype()throws {
        for task in TensorRoutingLearningTask.allCases {
            let d=draft(task),result=TensorLearningCanvas.inspect(d.graph,tensorRoutingTask:task)
            XCTAssertTrue(result.issues.isEmpty,result.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(result.tensorContracts["out"]?.layout.sample.dimensions,task.outputSampleShape)
            XCTAssertEqual(result.tensorContracts["out"]?.dtype,.float64)
            if task.kind == "split" || task.kind == "chunk" {
                XCTAssertNil(result.tensorContracts["op"],"Named parts must not be disguised as an out port")
                let count=task == .chunk ? 5:3
                for part in 0..<count {XCTAssertNotNil(result.outputTensorContracts[.init("op","part\(part)")])}
                XCTAssertNil(result.outputTensorContracts[.init("op","part\(count)")])
            }
            if task == .gather {XCTAssertEqual(result.tensorContracts["indices"]?.dtype,.int64)}
        }
        var wrong=draft(.gather).graph
        wrong.edges[0].targetPort="indices";wrong.edges[1].targetPort="source"
        XCTAssertFalse(TensorLearningCanvas.inspect(wrong,tensorRoutingTask:.gather).issues.isEmpty)
        wrong=draft(.chunk).graph;wrong.edges[1].sourcePort="part5"
        XCTAssertFalse(TensorLearningCanvas.inspect(wrong,tensorRoutingTask:.chunk).issues.isEmpty)
    }

}
