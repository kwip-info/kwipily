import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainabilityInspectorContextTests: XCTestCase {
    private let choices = [InitializationOwnerChoice(node: "expanded/first", parameter: "weight", name: "first.weight", shape: [2, 3]), InitializationOwnerChoice(node: "expanded/first", parameter: "bias", name: "first.bias", shape: [2])]
    private func empty() throws -> [String: JSONValue] { try TrainabilityConfiguration().values }
    func testTypedOwnersModesApplyDiscardAndPending() throws {
        var pending = false
        var committed: [String: JSONValue]?
        let baseline = try empty()
        let context = TrainabilityInspectorContext(configuration: baseline, owners: choices, origin: "editor", isCurrent: { true }, isBusy: { false }, apply: { expected, updated in
            XCTAssertEqual(expected, baseline); committed = updated; return true
        }, pending: { pending = $0 })
        XCTAssertTrue(context.addOwner(choices[0].id))
        XCTAssertTrue(context.setMode("frozen", at: 0))
        XCTAssertTrue(pending); XCTAssertTrue(context.canApply)
        XCTAssertFalse(context.addOwner(choices[0].id))
        XCTAssertFalse(context.setMode("eval", at: 0))
        XCTAssertTrue(context.apply())
        let saved = try TrainabilityConfiguration(values: XCTUnwrap(committed))
        guard case .object(let savedRow) = saved.owners.first else { return XCTFail("Missing committed owner") }
        XCTAssertEqual(savedRow["mode"], .string("frozen"))
        XCTAssertFalse(pending); XCTAssertFalse(context.hasChanges)
        XCTAssertTrue(context.setMode("trainable", at: 0)); XCTAssertTrue(context.discard())
        XCTAssertEqual(context.draftConfiguration, committed)
        XCTAssertTrue(context.removeOwner(at: 0)); XCTAssertTrue(pending)
        context.releasePendingEdits(); XCTAssertFalse(pending)
    }
    func testUnresolvedOwnerRetainedUntilExplicitRepair() throws {
        let raw: [String: JSONValue] = ["owners": .array([.object(["node": .string("removed"), "parameter": .string("weight"), "mode": .string("frozen")])])]
        let context = TrainabilityInspectorContext(configuration: raw, owners: choices, origin: "repair", isCurrent: { true }, isBusy: { false }, apply: { _, _ in true }, pending: { _ in })
        XCTAssertEqual(context.rows.count, 1); XCTAssertEqual(context.draftConfiguration, raw)
        XCTAssertNotNil(context.validationMessage); XCTAssertFalse(context.canApply)
        XCTAssertTrue(context.setOwner(choices[1].id, at: 0))
        XCTAssertNil(context.validationMessage); XCTAssertTrue(context.apply())
        guard case .object(let repaired) = context.rows.first else { return XCTFail("Missing repaired owner") }
        XCTAssertEqual(repaired["parameter"], .string("bias"))
    }
    func testStaleBusyAndFailedCommitKeepDraft() throws {
        var current = true, busy = false, succeeds = false
        let context = TrainabilityInspectorContext(configuration: try empty(), owners: choices, origin: "guard", isCurrent: { current }, isBusy: { busy }, apply: { _, _ in succeeds }, pending: { _ in })
        XCTAssertTrue(context.addOwner(choices[0].id))
        let draft = context.draftConfiguration
        XCTAssertFalse(context.apply()); XCTAssertEqual(context.draftConfiguration, draft); XCTAssertNotNil(context.rejection)
        busy = true; XCTAssertFalse(context.setMode("frozen", at: 0)); XCTAssertFalse(context.discard())
        busy = false; current = false; succeeds = true
        XCTAssertFalse(context.apply()); XCTAssertFalse(context.removeOwner(at: 0)); XCTAssertEqual(context.draftConfiguration, draft)
    }
    func testBoundedColdDraftAndForeignEnvelopeRefusal() throws {
        let make = { TrainabilityInspectorContext(configuration: try! self.empty(), owners: self.choices, origin: "cold", isCurrent: { true }, isBusy: { false }, apply: { _, _ in true }, pending: { _ in }) }
        let first = make(); XCTAssertTrue(first.addOwner(choices[0].id)); XCTAssertTrue(first.setMode("frozen", at: 0))
        let data = try XCTUnwrap(first.serializedDraft())
        let cold = make(); XCTAssertTrue(cold.restorePending(data)); XCTAssertEqual(cold.draftConfiguration, first.draftConfiguration); XCTAssertTrue(cold.hasChanges)
        XCTAssertFalse(cold.restorePending(Data(repeating: 32, count: 131073)))
        XCTAssertFalse(cold.restorePending(Data("{\"schemaVersion\":1,\"configuration\":{\"owners\":[]},\"foreign\":true}".utf8)))
        XCTAssertEqual(cold.draftConfiguration, first.draftConfiguration)
    }
}
