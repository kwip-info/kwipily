import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowGlobalResultsPresentationTests: XCTestCase {
    func testStatusDoesNotTreatFailureOrPartialCaptureAsSuccess() {
        XCTAssertEqual(WorkflowGlobalResultsPresentation.tone("completed"), .complete)
        XCTAssertEqual(WorkflowGlobalResultsPresentation.tone("failed"), .problem)
        XCTAssertEqual(WorkflowGlobalResultsPresentation.tone("cancelled"), .attention)
        XCTAssertEqual(WorkflowGlobalResultsPresentation.tone("preparing"), .active)
        XCTAssertEqual(WorkflowGlobalResultsPresentation.tone("idle"), .neutral)
        XCTAssertEqual(WorkflowGlobalResultsPresentation.tone("unknown"), .neutral)
    }
    func testArtifactInventoryIsRecordedRatherThanFreshVerification() {
        let rows = WorkflowGlobalResultsPresentation.artifacts(hashes: ["weights.pt":"abc", "model.json":"def"], completed: true)
        XCTAssertEqual(rows.map(\.name), ["model.json", "weights.pt"])
        XCTAssertEqual(rows.map(\.state), ["Recorded", "Recorded"])
        XCTAssertTrue(WorkflowGlobalResultsPresentation.artifacts(hashes: ["partial.pt":"abc"], completed: false).isEmpty)
    }
    func testPlainMessageRowsRetainRawEvidenceAndNeverInventAttribution() {
        let raw = ["Captured preparation output · source", "{\"event\":\"error\",\"message\":\"bad input\"}"]
        let rows = WorkflowGlobalResultsPresentation.messages(raw, query: "BAD")
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?.sequence, 2)
        XCTAssertEqual(rows.first?.message, raw[1])
        XCTAssertEqual(WorkflowGlobalResultsPresentation.messages(raw).map(\.message), raw)
        XCTAssertTrue(WorkflowGlobalResultsPresentation.messages(raw, query: "missing").isEmpty)
    }
    func testPlainMessageDisplayHasFiniteRetention() {
        let rows = WorkflowGlobalResultsPresentation.messages((0..<800).map(String.init))
        XCTAssertEqual(rows.count, 200)
        XCTAssertEqual(rows.first?.sequence, 601)
        XCTAssertEqual(rows.last?.message, "799")
    }
    func testPausedPlainMessageSnapshotDoesNotFollowNewCaptureOrDiscardEvidence() {
        var captured = ["first raw message"]
        let paused = captured
        captured.append("later raw message")
        XCTAssertEqual(WorkflowGlobalResultsPresentation.messages(paused).map(\.message), ["first raw message"])
        XCTAssertEqual(WorkflowGlobalResultsPresentation.messages(captured).map(\.message), ["first raw message", "later raw message"])
        XCTAssertEqual(captured.count, 2)
    }

    func testRollingWindowCannotRelabelSelectedRawEvidence() throws {
        let before=(0..<200).map {"message \($0)"}
        let selected=try XCTUnwrap(WorkflowGlobalResultsPresentation.messages(before).first)
        let after=Array(before.dropFirst())+["new message"]
        XCTAssertNil(WorkflowGlobalResultsPresentation.retainedSelection(selected,in:after))
        XCTAssertEqual(WorkflowGlobalResultsPresentation.retainedSelection(selected,in:before),selected,"A paused display retains its exact snapshot.")
        XCTAssertEqual(WorkflowGlobalResultsPresentation.retainedSelection(selected,in:Array(before.prefix(50))),selected,"Appending without rebasing does not unnecessarily clear selection.")
    }

    func testCompletedPreparationInventoryUsesOnlyObservedPortMetadata() {
        let outputs:[String:[String:JSONValue]]=[
            "source":["table":.object(["kind":.string("table"),"preview":.object(["rowCount":.integer(12)]),"lineage":.object(["dataVersion":.string("captured-version")])])],
            "fit":["state":.object(["kind":.string("state"),"preview":.object(["columns":.array([])])])],
            "unknown":["output":.object([:])]]
        let rows=WorkflowGlobalResultsPresentation.capturedPorts(outputs,completed:true)
        XCTAssertEqual(rows.map(\.nodeID),["fit","source","unknown"])
        XCTAssertEqual(rows.map(\.kind),["state","table","Not recorded"])
        XCTAssertEqual(rows.map(\.rowCount),[nil,12,nil])
        XCTAssertEqual(rows.map(\.state),["Recorded","Recorded","Recorded"])
        XCTAssertEqual(rows[1].dataVersion,"captured-version")
        XCTAssertTrue(WorkflowGlobalResultsPresentation.capturedPorts(outputs,completed:false).isEmpty,"Partial observations are not a completed artifact inventory.")
    }
    func testCapturedPortInventoryKeepsEveryPortAndRejectsInventedCounts() {
        let outputs:[String:[String:JSONValue]]=["split":[
            "validation":.object(["kind":.string("table"),"preview":.object(["rowCount":.integer(0)])]),
            "training":.object(["kind":.string("table"),"preview":.object(["rowCount":.string("100")])])]]
        let rows=WorkflowGlobalResultsPresentation.capturedPorts(outputs,completed:true)
        XCTAssertEqual(rows.map(\.port),["training","validation"])
        XCTAssertEqual(rows.map(\.rowCount),[nil,0])
        XCTAssertEqual(Set(rows.map(\.id)).count,2)
    }

}
