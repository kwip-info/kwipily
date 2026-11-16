import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class StatefulNormalizationInactiveParameterTests:XCTestCase {
    func testScaleAndShiftTogglesPreserveEditedDraftValuesButSubmitOnlyActiveState()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:TensorLearningDraftRepository(directory:root)),task=StatefulNormalizationLearningTask.bn1Remembered
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
        let graph=GraphDocument(name:"Remembered channel values",nodes:[.init(id:"x",type:"input"),.init(id:"op",type:task.componentID,config:task.configuration),.init(id:"out",type:"output")],edges:[.init(id:"a",source:"x",target:"op"),.init(id:"b",source:"op",target:"out")])
        XCTAssertTrue(session.update {$0.graph=graph})
        let canvas=TensorLearningCanvasAdapter(session:session)
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:"op"))
        let bank=["weight":try XCTUnwrap(task.scale),"bias":try XCTUnwrap(task.bias)]
        XCTAssertTrue(canvas.setNormalizationParameters(nodeID:"op",values:bank));XCTAssertTrue(canvas.setGradientTarget("op"))
        var config=task.configuration;config["bias"] = .bool(false)
        XCTAssertTrue(canvas.configure(nodeID:"op",values:config),canvas.statusMessage)
        XCTAssertEqual(session.draft?.parameters?["op"],bank,"Disabled shift values are parked in the editable draft, not deleted")
        var submitted=try? StatefulNormalizationLearningSubmission.decode(session.draft!.submissionData(),task:task)
        XCTAssertNotNil(submitted)
        XCTAssertEqual(Set(submitted?.parameters["op"]?.keys.map {$0} ?? []),["weight"]);XCTAssertEqual(submitted?.parameters["op"]?["weight"],bank["weight"])
        config["bias"] = .bool(true);XCTAssertTrue(canvas.configure(nodeID:"op",values:config))
        XCTAssertEqual(session.draft?.parameters?["op"],bank)
        config["affine"] = .bool(false);XCTAssertTrue(canvas.configure(nodeID:"op",values:config),canvas.statusMessage)
        XCTAssertEqual(session.draft?.parameters?["op"],bank);XCTAssertEqual(session.draft?.gradientNodeID,"op","Park the chosen target for reactivation")
        submitted=try? StatefulNormalizationLearningSubmission.decode(session.draft!.submissionData(),task:task)
        XCTAssertNotNil(submitted)
        XCTAssertEqual(submitted?.parameters.isEmpty,true);XCTAssertNil(submitted?.gradientNodeID)
        let cold=try TensorLearningDraft.decode(session.draft!.encoded());XCTAssertEqual(cold.parameters?["op"],bank);XCTAssertEqual(cold.gradientNodeID,"op")
        config["affine"] = .bool(true);XCTAssertTrue(canvas.configure(nodeID:"op",values:config))
        submitted=try? StatefulNormalizationLearningSubmission.decode(session.draft!.submissionData(),task:task)
        XCTAssertNotNil(submitted)
        XCTAssertEqual(submitted?.parameters["op"],bank);XCTAssertEqual(submitted?.gradientNodeID,"op")
        canvas.undo();XCTAssertEqual(session.draft?.graph.nodes[1].config["affine"],.bool(false));XCTAssertEqual(session.draft?.parameters?["op"],bank)
        canvas.redo();XCTAssertEqual(session.draft?.graph.nodes[1].config["affine"],.bool(true));XCTAssertEqual(session.draft?.parameters?["op"],bank)
    }
    func testPendingShiftTextSurvivesDisablingItsControlsAndColdDraft()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:TensorLearningDraftRepository(directory:root)),task=StatefulNormalizationLearningTask.bn1Remembered
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
        XCTAssertTrue(session.update {$0.graph = .init(name:"Pending channel edits",nodes:[.init(id:"op",type:task.componentID,config:task.configuration)],edges:[])})
        let canvas=TensorLearningCanvasAdapter(session:session);XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:"op"))
        XCTAssertTrue(canvas.setPendingParameter(nodeID:"op",name:"bias",index:1,text:"-0.75"))
        var config=task.configuration;config["bias"] = .bool(false)
        XCTAssertTrue(canvas.configure(nodeID:"op",values:config));XCTAssertEqual(canvas.pendingParameter(nodeID:"op",name:"bias",index:1),"-0.75")
        config["affine"] = .bool(false);XCTAssertTrue(canvas.configure(nodeID:"op",values:config))
        XCTAssertEqual(canvas.pendingParameter(nodeID:"op",name:"bias",index:1),"-0.75")
        let cold=try TensorLearningDraft.decode(session.draft!.encoded());XCTAssertEqual(cold.pendingFields,session.draft?.pendingFields)
        config["affine"] = .bool(true);config["bias"] = .bool(true);XCTAssertTrue(canvas.configure(nodeID:"op",values:config))
        XCTAssertTrue(canvas.applyPendingParameter(nodeID:"op",name:"bias",index:1));XCTAssertEqual(session.draft?.parameters?["op"]?["bias"]?.values[1],-0.75)
    }
    func testChannelResizeParksOldVectorsUntilExplicitReinitialization()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:TensorLearningDraftRepository(directory:root)),task=StatefulNormalizationLearningTask.bn1Remembered
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
        XCTAssertTrue(session.update {$0.graph = .init(name:"Resized channel values",nodes:[.init(id:"x",type:"input"),.init(id:"op",type:task.componentID,config:task.configuration),.init(id:"out",type:"output")],edges:[.init(id:"a",source:"x",target:"op"),.init(id:"b",source:"op",target:"out")])})
        let canvas=TensorLearningCanvasAdapter(session:session);XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:"op"));XCTAssertTrue(canvas.setGradientTarget("op"))
        let original=try XCTUnwrap(session.draft?.parameters?["op"])
        var config=task.configuration;config["num_features"] = .integer(4)
        XCTAssertTrue(canvas.configure(nodeID:"op",values:config),canvas.statusMessage);XCTAssertEqual(session.draft?.parameters?["op"],original)
        XCTAssertEqual(try TensorLearningDraft.decode(session.draft!.encoded()).parameters?["op"],original)
        XCTAssertThrowsError(try session.draft!.submissionData(),"Connected source still has three channels")
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:"op"));XCTAssertEqual(session.draft?.parameters?["op"]?["weight"]?.shape,[4])
        config["num_features"] = .integer(3);XCTAssertTrue(canvas.configure(nodeID:"op",values:config))
        XCTAssertEqual(session.draft?.parameters?["op"]?["weight"]?.shape,[4]);XCTAssertThrowsError(try session.draft!.submissionData(),"Stale parked width cannot reach execution")
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:"op"))
        let submission=try StatefulNormalizationLearningSubmission.decode(session.draft!.submissionData(),task:task)
        XCTAssertEqual(submission.parameters["op"]?["weight"]?.shape,[3]);XCTAssertEqual(submission.gradientNodeID,"op")
    }
}
