import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientPolicyInspectorContextTests: XCTestCase {
    private let output = GradientEndpointChoice(node: "branch", port: "hidden", name: "Named branch result", dtype: .float32, shape: [2], isModelInput: false)
    private let input = GradientEndpointChoice(node: "input", port: "out", name: "Model observations", dtype: .float32, shape: [2], isModelInput: true)
    private let integer = GradientEndpointChoice(node: "indices", port: "out", name: "Integer observations", dtype: .int64, shape: [2], isModelInput: true)
    private let owner = InitializationOwnerChoice(node: "first", parameter: "weight", name: "layers.first.weight", shape: [2, 2])
    private func request(id: String = "effect", outputNode: String = "branch", targetNode: String = "input") -> JSONValue {
        .object(["id": .string(id), "output": .object(["node": .string(outputNode), "port": .string("hidden")]), "partition": .string("validation"), "row_start": 0, "row_count": 1, "reduction": .string("sum"), "upstream": .null, "targets": .array([.object(["kind": .string("input"), "node": .string(targetNode), "port": .string("out")])]), "unavailable": .string("error")])
    }
    private func config(_ requests: [JSONValue] = []) throws -> [String: JSONValue] { try GradientPolicyConfiguration(values: ["requests": .array(requests)]).values }
    private func context(_ source: [String: JSONValue], current: @escaping () -> Bool = { true }, busy: @escaping () -> Bool = { false }, validator: @escaping ([String: JSONValue]) throws -> Void = { _ in }, apply: @escaping ([String: JSONValue], [String: JSONValue]) -> Bool = { _, _ in true }, pending: @escaping (Bool) -> Void = { _ in }) -> GradientPolicyInspectorContext {
        .init(configuration: source, endpoints: [output, input, integer], owners: [owner], origin: "captured-editor", isCurrent: current, isBusy: busy, validate: validator, apply: apply, pending: pending)
    }
    func testOrderedTypedRequestsCommitExpectedSourceAndDiscardDoesNotMutateIt() throws {
        let source = try config(); var saved = source; var pending = false; var calls = 0
        let editor = context(source, apply: { expected, updated in XCTAssertEqual(expected, saved); saved = updated; calls += 1; return true }, pending: { pending = $0 })
        XCTAssertTrue(editor.addRequest()); XCTAssertEqual(editor.requests.count, 1)
        XCTAssertEqual(saved, source); XCTAssertTrue(pending)
        XCTAssertTrue(editor.setRequestID("first-effect", at: 0))
        XCTAssertTrue(editor.setOutput(output.id, at: 0)); XCTAssertTrue(editor.setPartition("training", at: 0))
        XCTAssertTrue(editor.setRowStart(2, at: 0)); XCTAssertTrue(editor.setRowCount(2, at: 0))
        XCTAssertTrue(editor.addParameterTarget(owner.id, at: 0))
        XCTAssertTrue(editor.addRequest()); XCTAssertTrue(editor.setRequestID("second-effect", at: 1))
        XCTAssertTrue(editor.moveRequest(from: 1, to: 0)); XCTAssertEqual(editor.requests.first?["id"], .string("second-effect"))
        XCTAssertTrue(editor.setAccumulationSteps(3)); XCTAssertTrue(editor.canApply)
        XCTAssertTrue(editor.apply()); XCTAssertEqual(calls, 1); XCTAssertFalse(pending)
        XCTAssertEqual(try GradientPolicyConfiguration(values: saved).accumulationSteps, 3)
        XCTAssertEqual(editor.expectedConfiguration, saved)
        XCTAssertTrue(editor.removeRequest(at: 0)); XCTAssertTrue(editor.discard())
        XCTAssertEqual(editor.draftConfiguration, saved); XCTAssertEqual(calls, 1)
        XCTAssertFalse(editor.setOutput("foreign", at: 0)); XCTAssertFalse(editor.addParameterTarget("foreign", at: 0))
    }
    func testUnresolvedReferencesRemainVisibleAndRequireCurrentChoiceRepair() throws {
        let source = try config([request(outputNode: "removed", targetNode: "retired-input")])
        let editor = context(source)
        XCTAssertEqual(editor.requests.first?["output"], .object(["node": .string("removed"), "port": .string("hidden")]))
        XCTAssertNotNil(editor.validationMessage); XCTAssertFalse(editor.canApply)
        XCTAssertTrue(editor.setOutput(output.id, at: 0)); XCTAssertNotNil(editor.validationMessage)
        XCTAssertTrue(editor.removeTarget(at: 0, request: 0)); XCTAssertTrue(editor.addInputTarget(input.id, at: 0))
        XCTAssertNil(editor.validationMessage); XCTAssertTrue(editor.canApply)
        XCTAssertFalse(editor.addInputTarget(output.id, at: 0), "An intermediate output is not a model-input target.")
        XCTAssertTrue(editor.addInputTarget(integer.id, at: 0)); XCTAssertNotNil(editor.validationMessage)
        XCTAssertTrue(editor.setUnavailable("reportUnavailable", at: 0)); XCTAssertNil(editor.validationMessage)
    }
    func testUpstreamFullShapeAndCallerObjectiveValidationAreStrict() throws {
        let source = try config([request()])
        let editor = context(source, validator: { value in try GradientPolicyConfiguration(values: value).validateObjectiveCompatibility(reduction: "mean") })
        XCTAssertTrue(editor.setReduction("upstream", at: 0))
        XCTAssertTrue(editor.setUpstream(shape: [1, 2], values: [0.25, -0.5], at: 0)); XCTAssertNil(editor.validationMessage)
        XCTAssertFalse(editor.setUpstream(shape: [1, 2], values: [.infinity, 0], at: 0))
        XCTAssertTrue(editor.setUpstream(shape: [2, 1], values: [0.25, -0.5], at: 0)); XCTAssertNotNil(editor.validationMessage)
        XCTAssertTrue(editor.setUpstream(shape: [1, 2], values: [0.25, -0.5], at: 0))
        XCTAssertTrue(editor.setGradientScale("sum")); XCTAssertNotNil(editor.validationMessage); XCTAssertFalse(editor.canApply)
        XCTAssertTrue(editor.setGradientScale("mean")); XCTAssertTrue(editor.canApply)
    }
    func testRawNumericDraftColdSnapshotAndDiscardRetainOriginalBaseline() throws {
        let source = try config([request()]); let warm = context(source)
        let key = "request.0.row_start"
        XCTAssertTrue(warm.setRawNumericDraft(key, text: "-", origin: warm.numericOrigin(key)))
        XCTAssertTrue(warm.hasUncommittedNumbers); XCTAssertFalse(warm.canApply)
        XCTAssertFalse(warm.commitNumericDraft(key))
        XCTAssertFalse(warm.setRawNumericDraft(key, text: "3", origin: "retired"))
        let snapshot = try warm.snapshotData(); XCTAssertLessThanOrEqual(snapshot.count, 131_072)
        let cold = context(source); XCTAssertTrue(cold.restorePending(snapshot))
        XCTAssertEqual(cold.expectedConfiguration, source); XCTAssertEqual(cold.rawNumericDrafts[key], "-")
        XCTAssertTrue(cold.setRawNumericDraft(key, text: "3", origin: cold.numericOrigin(key)))
        XCTAssertTrue(cold.commitNumericDraft(key)); XCTAssertEqual(cold.requests[0]["row_start"], 3)
        XCTAssertTrue(cold.canApply); XCTAssertTrue(cold.discard())
        XCTAssertEqual(cold.draftConfiguration, source); XCTAssertTrue(cold.rawNumericDrafts.isEmpty)
        XCTAssertFalse(cold.restorePending(Data(repeating: 32, count: 131_073)))
        XCTAssertFalse(cold.restorePending(Data("{}".utf8)))
        XCTAssertFalse(cold.setRawNumericDraft("unknown", text: "1", origin: cold.numericOrigin("unknown")))
    }
    func testIncompleteNameAndTargetDraftsSurviveColdRestorationForExplicitRepair() throws {
        let source=try config([request()]),editor=context(source)
        XCTAssertTrue(editor.setRequestID("",at:0))
        XCTAssertEqual(editor.requests[0]["id"],.string(""));XCTAssertNotNil(editor.validationMessage);XCTAssertFalse(editor.canApply)
        XCTAssertTrue(editor.removeTarget(at:0,request:0))
        let snapshot=try editor.snapshotData(),cold=context(source)
        XCTAssertTrue(cold.restorePending(snapshot));XCTAssertEqual(cold.requests[0]["id"],.string(""))
        XCTAssertEqual(cold.requests[0]["targets"],.array([]));XCTAssertFalse(cold.canApply)
        XCTAssertTrue(cold.setRequestID("repaired",at:0));XCTAssertTrue(cold.addInputTarget(input.id,at:0))
        XCTAssertTrue(cold.canApply);XCTAssertTrue(cold.apply())
    }
    func testInfluenceShapeAndValuesCanBeResizedTogetherWithoutLosingEitherRawDraft() throws {
        let source=try config([request()]),editor=context(source)
        XCTAssertTrue(editor.setReduction("upstream",at:0));XCTAssertTrue(editor.setRowCount(2,at:0))
        let shape="request.0.upstream.shape",values="request.0.upstream.values"
        XCTAssertTrue(editor.setRawNumericDraft(shape,text:"2, 2",origin:editor.numericOrigin(shape)))
        XCTAssertTrue(editor.setRawNumericDraft(values,text:"1, 2, 3, 4",origin:editor.numericOrigin(values)))
        let snapshot=try editor.snapshotData(),cold=context(source)
        XCTAssertTrue(cold.restorePending(snapshot))
        XCTAssertEqual(cold.rawNumericDrafts[shape],"2, 2");XCTAssertEqual(cold.rawNumericDrafts[values],"1, 2, 3, 4")
        XCTAssertTrue(editor.commitNumericDraft(shape));XCTAssertFalse(editor.hasUncommittedNumbers)
        XCTAssertNil(editor.validationMessage);XCTAssertTrue(editor.canApply)
        guard case .object(let upstream)=editor.requests[0]["upstream"] else {return XCTFail("Missing influence array")}
        XCTAssertEqual(upstream["shape"],.array([2,2]));guard case .array(let measured)=upstream["values"] else {return XCTFail("Missing influence values")};XCTAssertEqual(measured.compactMap(\.doubleValue),[1,2,3,4])
    }
    func testBusyStaleAndExpectedConfigurationConflictCannotCommitOrClearDraft() throws {
        var current = true, busy = false, calls = 0
        let source = try config([request()])
        let editor = context(source, current: { current }, busy: { busy }, apply: { _, _ in calls += 1; return false })
        XCTAssertTrue(editor.setAccumulationSteps(2)); XCTAssertFalse(editor.apply()); XCTAssertNotNil(editor.rejection)
        XCTAssertEqual(calls, 1); XCTAssertEqual(editor.expectedConfiguration, source); XCTAssertTrue(editor.hasChanges)
        busy = true; XCTAssertFalse(editor.setAccumulationSteps(3)); XCTAssertFalse(editor.apply()); XCTAssertFalse(editor.discard())
        busy = false; current = false
        XCTAssertFalse(editor.setGradientScale("sum")); XCTAssertFalse(editor.apply()); XCTAssertFalse(editor.discard())
        XCTAssertEqual(calls, 1)
    }
}
