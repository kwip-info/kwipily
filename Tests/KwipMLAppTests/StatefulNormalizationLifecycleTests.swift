import XCTest
import KwipMLCore
@testable import KwipMLApp

final class StatefulNormalizationLifecycleTests: XCTestCase {
    func testBufferTransferIsRequiredOnlyByCapturedTrackedComponents() throws {
        for id in ["batch_norm1d", "instance_norm2d"] {
            let graph = GraphDocument(name: "Tracked channels", nodes: [.init(id: "op", type: id, config: ["track_running_stats": true])])
            let all = ["parameters", "buffers", "optimizer", "randomGenerator", "epoch", "metrics"]
            let value: [String: JSONValue] = ["lineage": .object(["mode": .string("newTraining"), "parentRunID": .null, "restoredState": .array([]), "resetState": .array(all.map(JSONValue.string))]), "settings": .object(["initialEpoch": 0, "epochsExecuted": 2])]
            XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(value, mode: .fresh, targetEpochs: 2, parent: nil, graph: graph))
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(value, mode: .fresh, targetEpochs: 2, parent: nil))
            var untracked = graph; untracked.nodes[0].config["track_running_stats"] = false
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(value, mode: .fresh, targetEpochs: 2, parent: nil, graph: untracked))
            var missing = value
            if case .object(var lineage) = missing["lineage"] {lineage["resetState"] = .array(all.filter { $0 != "buffers" }.map(JSONValue.string));missing["lineage"] = .object(lineage)}
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(missing, mode: .fresh, targetEpochs: 2, parent: nil, graph: graph))
        }
    }
}
