import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class LossLearningCanvasTests:XCTestCase {
    @MainActor func testEightRestrictedRegistriesUseRealFieldsObjectiveControlAndTypedFixturePorts()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        for task in LossLearningTask.allCases {
            let session=TensorLearningDraftSession(repository:.init(directory:root))
            XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
            XCTAssertTrue(session.update {$0.graph=LossLearningCanvas.preparedGraph(task:task)})
            let canvas=TensorLearningCanvasAdapter(session:session)
            XCTAssertNotNil(canvas.registry.component(type:task.componentID))
            XCTAssertEqual(canvas.registry.component(type:"objective_check")?.inputs,task.sourceRoles+["objective"])
            XCTAssertEqual(canvas.registry.component(type:"objective_check")?.outputs,["raw","objective","inputInfluence"])
            XCTAssertNil(canvas.registry.component(type:"linear"))
            XCTAssertTrue(canvas.validation.isValid,canvas.validation.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"objective",port:"out",direction:.output).kind,.objectiveRecipe)
            XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:"check",port:"objective",direction:.input).kind,.objectiveRecipe)
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:"prediction",port:"out").contains("float64"))
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:"check",port:"inputInfluence").contains("float64"))
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:"check",port:"objective").contains("recipe"))
            if [.crossEntropy,.nll,.ctc].contains(task) {XCTAssertTrue(canvas.canvasPortDescription(nodeID:"target",port:"out").contains("int64"))}
            if task == .ctc {XCTAssertTrue(canvas.canvasPortDescription(nodeID:"input_lengths",port:"out").contains("[N]"))}
            XCTAssertFalse(canvas.canAddType(task.componentID));XCTAssertFalse(canvas.canAddType("input"));XCTAssertFalse(canvas.canAddType("output"))
        }
    }
    @MainActor func testPartialRoleDragConnectionsAndRecipeDraftApplyDiscardUndoColdAllModes()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        for mode in [LearningMode.guided,.practice,.assessment] {
            let session=TensorLearningDraftSession(repository:.init(directory:root))
            let project=UUID().uuidString,task=LossLearningTask.crossEntropy
            XCTAssertTrue(session.create(projectID:project,operation:task.rawValue,mode:mode))
            let canvas=TensorLearningCanvasAdapter(session:session)
            canvas.addNode(type:"input",x:0,y:0);canvas.addNode(type:"input",x:0,y:100)
            XCTAssertEqual(Set(canvas.graph.nodes.compactMap {$0.config["fixture"]?.stringValue}),Set(task.sourceRoles))
            canvas.addNode(type:task.componentID,x:200,y:0)
            XCTAssertNotNil(canvas.graph.nodes.first {$0.type == task.componentID})
            XCTAssertTrue(session.update {$0.graph=LossLearningCanvas.preparedGraph(task:task)})
            XCTAssertTrue(canvas.setPending(nodeID:"objective",field:"label_smoothing",text:"0.3"))
            XCTAssertEqual(canvas.graph.nodes.first {$0.id == "objective"}?.config["label_smoothing"],.number(0.2))
            XCTAssertThrowsError(try session.draft!.submissionData())
            XCTAssertTrue(canvas.applyPending(nodeID:"objective",field:"label_smoothing"))
            XCTAssertEqual(canvas.graph.nodes.first {$0.id == "objective"}?.config["label_smoothing"],.number(0.3))
            canvas.undo();XCTAssertEqual(canvas.graph.nodes.first {$0.id == "objective"}?.config["label_smoothing"],.number(0.2))
            canvas.discardPending(nodeID:"objective",field:"label_smoothing")
            XCTAssertTrue(canvas.setPending(nodeID:"objective",field:"class_weights",text:"1, 2, 0.75"))
            XCTAssertTrue(canvas.applyPending(nodeID:"objective",field:"class_weights"))
            XCTAssertEqual(canvas.graph.nodes.first {$0.id == "objective"}?.config["class_weights"],.array([.number(1),.number(2),.number(0.75)]))
            XCTAssertTrue(session.save())
            let cold=TensorLearningDraftSession(repository:.init(directory:root));XCTAssertTrue(cold.open(projectID:project))
            XCTAssertEqual(try cold.draft!.submissionData(),try session.draft!.submissionData())
            XCTAssertTrue(cold.update {$0.graph.edges.removeLast()})
            let coldCanvas=TensorLearningCanvasAdapter(session:cold);XCTAssertFalse(coldCanvas.validation.isValid)
            XCTAssertThrowsError(try cold.draft!.submissionData())
        }
    }
    @MainActor func testLongLossNumberListDraftPreservesCanonicalLimitsAndPriorShortFields()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:LossLearningTask.crossEntropy.rawValue,mode:.practice))
        XCTAssertTrue(session.update {$0.graph=LossLearningCanvas.preparedGraph(task:.crossEntropy)})
        let canvas=TensorLearningCanvasAdapter(session:session),text=Array(repeating:"0.125",count:128).joined(separator:", ")
        XCTAssertGreaterThan(text.count,512)
        XCTAssertTrue(canvas.setPending(nodeID:"objective",field:"class_weights",text:text))
        XCTAssertEqual(canvas.pendingText(nodeID:"objective",field:"class_weights"),text)
        XCTAssertTrue(session.save())
        let cold=TensorLearningDraftSession(repository:.init(directory:root));XCTAssertTrue(cold.open(projectID:session.draft!.projectID))
        XCTAssertEqual(TensorLearningCanvasAdapter(session:cold).pendingText(nodeID:"objective",field:"class_weights"),text)
        XCTAssertFalse(canvas.setPending(nodeID:"objective",field:"label_smoothing",text:String(repeating:"1",count:513)))
        XCTAssertFalse(canvas.setPending(nodeID:"objective",field:"class_weights",text:String(repeating:"0",count:32769)))
        var old=TensorLearningDraft(projectID:UUID().uuidString,operation:"reshape",mode:.practice)
        old.pendingFields["shape"]=text;XCTAssertThrowsError(try old.validate())
    }

    func testFormativeProjectionWithholdsPrivateOriginStaleSourceAndDifferentProject()throws {
        for task in LossLearningTask.allCases {
            let plan=try LossLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:47)
            let bytes=try LossLearningSubmission(graph:LossLearningCanvas.preparedGraph(task:task),task:task).encoded()
            let digest=SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:digest,fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:128)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let project=UUID().uuidString,receipt=try TensorLearningReceipt(projectID:project,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
                let projection=receipt.lossFormativeProjection(expectedProjectID:project,currentSubmission:bytes,currentMode:mode)
                XCTAssertEqual(projection == nil,mode == .assessment)
                XCTAssertNil(receipt.lossFormativeProjection(expectedProjectID:project,currentSubmission:bytes,currentMode:.assessment))
                XCTAssertNil(receipt.lossFormativeProjection(expectedProjectID:UUID().uuidString,currentSubmission:bytes,currentMode:.guided))
                XCTAssertNil(receipt.lossFormativeProjection(expectedProjectID:project,currentSubmission:bytes+Data([32]),currentMode:.guided))
                if mode == .assessment {XCTAssertNil(receipt.lossFormativeProjection(expectedProjectID:project,currentSubmission:bytes,currentMode:.guided))}
                else {XCTAssertEqual(projection?.fixtures.count,4);XCTAssertEqual(Set(projection?.fixtures[0].measured.keys.map {$0} ?? []),Set(plan.fixtures[0].expected.keys))}
            }
        }
    }

}
