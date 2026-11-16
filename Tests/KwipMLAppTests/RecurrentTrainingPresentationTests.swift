import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RecurrentTrainingPresentationTests: XCTestCase {
    private let types = ["rnn", "lstm", "gru", "rnn_cell", "lstm_cell", "gru_cell"]
    private func tensor(_ shape: [Int], dtype: String = "float32", values: [JSONValue] = [1]) -> JSONValue {
        .object(["shape": .array(shape.map { .integer(Int64($0)) }), "dtype": .string(dtype), "values": .array(values)])
    }
    private func fixture(_ type: String) throws -> (GraphDocument, [String: JSONValue]) {
        let graph = try ExampleFlows.recurrentClassifier(componentID: type)
        let cell = type.hasSuffix("_cell")
        var previews: [String: JSONValue] = ["input": tensor([8,2]), "head": tensor([8,2]), "output": tensor([8,2])]
        if !cell { previews["sequence"] = tensor([8,2,1]); previews["flat"] = tensor([8,4]) }
        var outputs: [String: JSONValue] = ["out": tensor(cell ? [8,2] : [8,2,2]), "hidden": tensor(cell ? [8,2] : [8,1,2])]
        if type == "lstm" || type == "lstm_cell" { outputs["cell"] = tensor(cell ? [8,2] : [8,1,2]) }
        previews["memory"] = .object(["outputs": .object(outputs)])
        return (graph, previews)
    }
    func testAllSixCapturedNamedRolesAcceptIncludingUnusedStates() throws {
        for type in types {
            let (graph, previews) = try fixture(type)
            XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph, sampleCapture: nil), type)
        }
    }
    func testEveryNamedRoleRefusesMalformedUnusedStatesAndAliasShapes() throws {
        for type in types {
            let (graph, original) = try fixture(type)
            for mutation in 0..<8 {
                var previews = original
                guard case .object(var row) = previews["memory"], case .object(var outputs) = row["outputs"] else { return XCTFail() }
                let role = type.hasPrefix("lstm") ? "cell" : "hidden"
                switch mutation {
                case 0: outputs.removeValue(forKey: "hidden")
                case 1: outputs["unexpected"] = tensor([8,2])
                case 2: outputs[role] = tensor([7,2])
                case 3: outputs[role] = tensor([8,3])
                case 4: outputs[role] = tensor(type.hasSuffix("_cell") ? [8,2] : [8,1,2], dtype: "int64")
                case 5: outputs[role] = tensor(type.hasSuffix("_cell") ? [8,2] : [8,1,2], values: Array(repeating: 1, count: 9))
                case 6: outputs["out"] = tensor([8,2]) // wrong sequence shape; cells changed below
                    if type.hasSuffix("_cell") { outputs["out"] = tensor([8,3]) }
                default: row["shape"] = .array([8,2])
                }
                row["outputs"] = .object(outputs); previews["memory"] = .object(row)
                XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph, sampleCapture: nil), "\(type) mutation \(mutation)")
            }
        }
    }
    func testCellResultAndHiddenAreExactCaptureAliases() throws {
        for type in ["rnn_cell", "lstm_cell", "gru_cell"] {
            let (graph, original) = try fixture(type)
            for values: [JSONValue] in [[2], [1, 1], []] {
                var previews = original
                guard case .object(var row) = previews["memory"], case .object(var outputs) = row["outputs"] else { return XCTFail() }
                outputs["hidden"] = tensor([8,2], values: values)
                row["outputs"] = .object(outputs); previews["memory"] = .object(row)
                XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph, sampleCapture: nil), type)
            }
        }
    }
    func testCapturePolicyAndStatusApplyToAllRecurrentOutputs() throws {
        for type in types {
            let (graph, original) = try fixture(type)
            for capture in [false, true] {
                let policy = TrainingSampleCapturePolicy(activationValues: capture, predictionRows: false)
                var previews = original
                for key in previews.keys {
                    guard case .object(var row) = previews[key] else { return XCTFail() }
                    row["captureStatus"] = .string(capture ? "captured" : "omittedByPolicy")
                    if case .object(var outputs) = row["outputs"] {
                        for role in outputs.keys { if case .object(var leaf) = outputs[role] { leaf["values"] = .array(capture ? [1] : []); outputs[role] = .object(leaf) } }
                        row["outputs"] = .object(outputs)
                    } else { row["values"] = .array(capture ? [1] : []) }
                    previews[key] = .object(row)
                }
                XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph, sampleCapture: policy))
                XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(original, graph: graph, sampleCapture: policy))
            }
        }
    }
    func testSuppliedHiddenUsesNamedSourceAndRefusesDifferentObservationCount() throws {
        let graph = GraphDocument(nodes: [
            .init(id: "input", type: "input", config: ["features": 2]),
            .init(id: "first", type: "gru_cell"),
            .init(id: "second", type: "lstm_cell", config: ["state_mode": .string("supplied")])
        ], edges: [
            .init(id: "a", source: "input", sourcePort: "out", target: "first", targetPort: "in"),
            .init(id: "b", source: "input", sourcePort: "out", target: "second", targetPort: "in"),
            .init(id: "h", source: "first", sourcePort: "hidden", target: "second", targetPort: "hidden"),
            .init(id: "c", source: "input", sourcePort: "out", target: "second", targetPort: "cell")
        ])
        var previews: [String: JSONValue] = ["input": tensor([8,2]), "first": .object(["outputs": .object(["out": tensor([8,2]), "hidden": tensor([8,2])])]), "second": .object(["outputs": .object(["out": tensor([8,2]), "hidden": tensor([8,2]), "cell": tensor([8,2])])])]
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph, sampleCapture: nil))
        previews["first"] = .object(["outputs": .object(["out": tensor([8,2]), "hidden": tensor([7,2])])])
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph, sampleCapture: nil))
    }
}
