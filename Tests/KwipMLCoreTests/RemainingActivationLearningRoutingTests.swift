import XCTest
@testable import KwipMLCore

final class RemainingActivationLearningRoutingTests:XCTestCase {
    private func draft(_ task:RemainingActivationLearningTask)->TensorLearningDraft {
        var d=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
        d.graph = .init(nodes:[.init(id:"x",type:"input"),.init(id:"activation",type:task.componentID,config:task.configuration),.init(id:"out",type:"output")],edges:[.init(id:"in",source:"x",target:"activation"),.init(id:"out",source:"activation",target:"out")])
        if let slopes=task.slopes {d.parameters=["activation":["weight":slopes]];d.gradientNodeID="activation"}
        return d
    }
    func testEveryTaskRoutesCanonicalDraftsAndPlansWithoutGenericEnvelopeEscape()throws {
        for task in RemainingActivationLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue));XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let d=draft(task),bytes=try d.submissionData()
            XCTAssertEqual(try RemainingActivationLearningSubmission.decode(bytes,task:task).encoded(),bytes)
            XCTAssertEqual(try TensorLearningDraft.decode(d.encoded()),d)
            XCTAssertThrowsError(try d.submission())
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:19),try RemainingActivationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:19))
            XCTAssertEqual(TensorLearningCanvas.registry(operation:task.rawValue).components.filter {$0.id == task.componentID}.count,1)
        }
    }
    func testPReLUDraftParameterOwnershipPendingEditsAndDeletionStayStrict()throws {
        var d=draft(.channelSlopes)
        XCTAssertNoThrow(try d.validate());d.pendingFields=["activation.parameter.weight.0":"0.9"]
        XCTAssertThrowsError(try d.submissionData());d.pendingFields=[:]
        d.gradientNodeID="out";XCTAssertThrowsError(try d.validate())
        d.gradientNodeID="activation";d.parameters?["activation"]?["weight"] = .init(shape:[1],values:[0.1]);XCTAssertThrowsError(try d.validate())
        d=draft(.channelSlopes);d.graph.nodes.removeAll {$0.id == "activation"};d.graph.edges=[]
        XCTAssertThrowsError(try d.validate(),"Removing a block cannot leave its parameter state owned by another node")
        d.parameters=[:];d.gradientNodeID=nil;XCTAssertNoThrow(try d.validate(),"An incomplete draft can be repaired before a full submission")
        var parameterless=draft(.dropoutTraining);parameterless.parameters=["activation":["weight":.init(shape:[1],values:[1])]]
        XCTAssertThrowsError(try parameterless.validate())
    }
    func testAllFiveReceiptsReplayAndPrivateProjectionRemainsUnavailable()throws {
        for task in RemainingActivationLearningTask.allCases {
            let d=draft(task),bytes=try d.submissionData(),plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:23)
            // Contract replay only: true worker measurements are independently tested in Python/App.
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:512)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let receipt=try TensorLearningReceipt(projectID:d.projectID,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
                XCTAssertEqual(try TensorLearningReceipt.decode(receipt.encoded()).verifiedReplay(expectedProjectID:d.projectID),plan)
                XCTAssertEqual(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:mode) == nil,mode == .assessment)
                if mode == .assessment {XCTAssertNil(receipt.formativeProjection(expectedProjectID:d.projectID,currentSubmission:bytes,currentMode:.guided))}
                var wrong=receipt;wrong.operation=task == .sharedSlope ? RemainingActivationLearningTask.channelSlopes.rawValue:RemainingActivationLearningTask.sharedSlope.rawValue
                XCTAssertThrowsError(try wrong.verifiedReplay(expectedProjectID:d.projectID))
            }
        }
    }
    func testLessonCanvasInfersFloat64ShapeAndRefusesChannelOrBatchAxisMismatch()throws {
        for task in RemainingActivationLearningTask.allCases {
            let d=draft(task),result=TensorLearningCanvas.inspect(d.graph,remainingActivationTask:task)
            XCTAssertTrue(result.issues.isEmpty,result.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(result.tensorLayouts["out"]?.sample.dimensions,[2,3])
            XCTAssertEqual(result.tensorContracts["out"]?.dtype,.float64,"Lesson shape admission must not widen production dtype support")
            if task.isPReLU {
                var wrong=d.graph;wrong.nodes[1].config["num_parameters"]=3
                XCTAssertFalse(TensorLearningCanvas.inspect(wrong,remainingActivationTask:task).issues.isEmpty)
            }
            if task == .logScores {
                var wrong=d.graph;wrong.nodes[1].config["dim"] = -3
                XCTAssertFalse(TensorLearningCanvas.inspect(wrong,remainingActivationTask:task).issues.isEmpty)
            }
        }
    }
}
