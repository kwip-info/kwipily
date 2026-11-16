import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSelectionPresentationTests: XCTestCase {
    private let a = WorkflowNodeAddress(domain: .data, localID: "same")
    private let b = WorkflowNodeAddress(domain: .model, localID: "same")
    private let c = WorkflowNodeAddress(domain: .control, localID: "other")
    private func document() -> WorkflowDocument {
        .init(projectID: "project", nodes: [
            .init(address: a, payload: .dataOperator(operatorID: "kwip.data.sourceFiles", config: [:])),
            .init(address: b, payload: .modelPrimitive(type: "linear", configurationVersion: 1, config: [:])),
            .init(address: c, payload: .train(config: [:]))], edges: [
                .init(id: "inside", from: .init(address: a, port: "table"), to: .init(address: b, port: "in"), kind: .table),
                .init(id: "outside", from: .init(address: b, port: "out"), to: .init(address: c, port: "model"), kind: .tensor)])
    }
    func testSelectedConnectionsAreScopedAndClassifiedWithoutDuplicateSelfEdges() {
        let d = document(), s = WorkflowSelection(nodeAddresses: [a, b], edgeIDs: ["inside"])
        XCTAssertEqual(WorkflowSelectionPresentation.connections(in: d, selection: s).map(\.id), ["inside", "outside"])
        XCTAssertEqual(WorkflowSelectionPresentation.relation(d.edges[0], selection: s), "Internal")
        XCTAssertEqual(WorkflowSelectionPresentation.relation(d.edges[1], selection: s), "Outgoing")
        XCTAssertEqual(WorkflowSelectionPresentation.relation(d.edges[1], selection: .init(nodeAddresses: [c])), "Incoming")
        XCTAssertEqual(WorkflowSelectionPresentation.relation(d.edges[1], selection: .init(edgeIDs: ["outside"])), "Selected")
    }
    func testBackToSelectionRefusesRemovedMembersOrAnotherSession() {
        let d = document(), session = UUID(), bookmark = WorkflowSelectionBookmark(projectID: "project", sessionID: UUID(), selection: .init(nodeAddresses: [a, b]))
        XCTAssertFalse(bookmark.canRestore(projectID: "project", sessionID: session, document: d))
        XCTAssertTrue(bookmark.canRestore(projectID: "project", sessionID: bookmark.sessionID, document: d))
        var changed = d; changed.nodes.removeAll { $0.address == b }
        XCTAssertFalse(bookmark.canRestore(projectID: "project", sessionID: bookmark.sessionID, document: changed))
        XCTAssertFalse(bookmark.canRestore(projectID: "other", sessionID: bookmark.sessionID, document: d))
    }
    @MainActor func testDiagnosticsNeverSubstituteRunWideOrUnselectedEvidence() {
        let console = WorkflowRunConsole()
        console.append(phase: .training, level: .error, message: "global")
        console.append(phase: .training, level: .warning, message: "data", component: a)
        console.append(phase: .training, level: .error, message: "model", component: b)
        console.append(phase: .training, level: .info, message: "information", component: a)
        XCTAssertEqual(WorkflowSelectionPresentation.diagnostics(console.entries, selection: .init(nodeAddresses: [a])).map(\.message), ["data"])
    }
    func testOutputUsesMeasuredCountNotPreviewSampleSizeAndNeverSumsPorts() {
        let snapshot = WorkflowCapturedDataOutput(projectID: "project", sessionID: UUID(), address: a,
            runID: UUID(), requestIdentity: "capture", capturedRevision: 1, bindingRevision: 1,
            phase: .completed, matchesCurrentConstruction: true, matchesCurrentBindings: true,
            ports: ["table": .object(["kind": .string("table"), "preview": .object(["rowCount": .integer(500_000), "rows": .array([.array([.integer(1)])])])])])
        XCTAssertEqual(WorkflowSelectionPresentation.outputDescription(snapshot), "table · 500000 records")
        XCTAssertEqual(WorkflowSelectionPresentation.outputDescription(nil), "Not captured")
    }
    func testCurrentConfigurationFindingsUseExactProjectedIDsNotRunWideMessages() {
        let findings = [ValidationIssue(id: "global", message: "global"),
            ValidationIssue(id: "data", message: "data", nodeID: "data:same"),
            ValidationIssue(id: "model", message: "model", nodeID: "model:same")]
        XCTAssertEqual(WorkflowSelectionPresentation.configurationIssues(findings, displayIDs: ["data:same"]).map(\.id), ["data"])
    }
    func testCaptureStateAndOwnershipRefuseFalseCurrentOrForeignEvidence() {
        let session = UUID()
        func capture(_ phase: WorkflowExecutionPhase = .completed, current: Bool = true, ports: [String:JSONValue] = ["out": .string("measured")]) -> WorkflowCapturedDataOutput {
            .init(projectID: "project", sessionID: session, address: a, runID: UUID(), requestIdentity: "capture",
                capturedRevision: 1, bindingRevision: 1, phase: phase, matchesCurrentConstruction: current,
                matchesCurrentBindings: true, ports: ports)
        }
        XCTAssertEqual(WorkflowSelectionPresentation.captureState(capture()), "Current")
        XCTAssertEqual(WorkflowSelectionPresentation.captureState(capture(.failed)), "Partial capture")
        XCTAssertEqual(WorkflowSelectionPresentation.captureState(capture(current: false)), "Stale")
        XCTAssertEqual(WorkflowSelectionPresentation.captureState(capture(ports: [:])), "Not captured")
        XCTAssertNil(WorkflowSelectionPresentation.owned(capture(), projectID: "foreign", sessionID: session, address: a))
        XCTAssertNil(WorkflowSelectionPresentation.owned(capture(), projectID: "project", sessionID: UUID(), address: a))
        XCTAssertNil(WorkflowSelectionPresentation.owned(capture(), projectID: "project", sessionID: session, address: b))
        XCTAssertNotNil(WorkflowSelectionPresentation.owned(capture(), projectID: "project", sessionID: session, address: a))
    }

}
