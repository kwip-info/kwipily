import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorRoutingDiscardStatusTests:XCTestCase {
    @MainActor func testDiscardAfterUnavailableChunkPartRefusalRestoresCommittedStatusWithoutChangingEvidenceOrUndo()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:TensorRoutingLearningTask.chunk.rawValue,mode:.guided))
        XCTAssertTrue(session.update {$0.graph=routingLessonGraph(.chunk)})
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.evidenceState={.current}
        let original=canvas.graph
        canvas.moveNodes(ids:["op"],dx:40,dy:10)
        let committed=canvas.graph
        let submission=try XCTUnwrap(session.draft).submissionData()
        XCTAssertTrue(canvas.canUndo)
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"chunks",text:"4"))
        XCTAssertFalse(canvas.applyPending(nodeID:"op",field:"chunks"))
        XCTAssertTrue(canvas.statusMessage.contains("part4"),canvas.statusMessage)
        XCTAssertEqual(canvas.pendingText(nodeID:"op",field:"chunks"),"4")
        XCTAssertEqual(canvas.graph,committed)

        canvas.discardPending(nodeID:"op",field:"chunks")

        XCTAssertEqual(canvas.statusMessage,"Unapplied change discarded.")
        XCTAssertNil(canvas.pendingText(nodeID:"op",field:"chunks"))
        XCTAssertEqual(canvas.configurationTextBinding(nodeID:"op",field:"chunks").wrappedValue,"6")
        XCTAssertEqual(canvas.graph,committed)
        XCTAssertEqual(try XCTUnwrap(session.draft).submissionData(),submission)
        XCTAssertTrue(canvas.validation.isValid)
        XCTAssertEqual(canvas.previewStatus(for:"out"),"Measured output")
        XCTAssertTrue(canvas.graph.edges.contains {$0.source == "op" && $0.sourcePort == "part4"})
        canvas.undo()
        XCTAssertEqual(canvas.graph,original)
        canvas.redo()
        XCTAssertEqual(canvas.graph,committed)
    }

    @MainActor func testDiscardPreservesOtherDraftFieldsAndDoesNotAcknowledgeWhileBusy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:TensorRoutingLearningTask.chunk.rawValue,mode:.guided))
        XCTAssertTrue(session.update {$0.graph=routingLessonGraph(.chunk)})
        let canvas=TensorLearningCanvasAdapter(session:session)
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"chunks",text:"4"))
        XCTAssertFalse(canvas.applyPending(nodeID:"op",field:"chunks"))
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"dim",text:"1"))
        let error=canvas.statusMessage
        canvas.assessmentIsRunning={true}
        canvas.discardPending(nodeID:"op",field:"chunks")
        XCTAssertEqual(canvas.statusMessage,error)
        XCTAssertEqual(canvas.pendingText(nodeID:"op",field:"chunks"),"4")
        canvas.assessmentIsRunning={false}
        canvas.discardPending(nodeID:"op",field:"chunks")
        XCTAssertEqual(canvas.statusMessage,"Unapplied change discarded.")
        XCTAssertNil(canvas.pendingText(nodeID:"op",field:"chunks"))
        XCTAssertEqual(canvas.pendingText(nodeID:"op",field:"dim"),"1")
        XCTAssertThrowsError(try XCTUnwrap(session.draft).submissionData())
    }
}
