import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RemainingDataObservationTests: XCTestCase {
    private let sha = String(repeating: "a", count: 64)
    private var identity: PreparationObservationIdentity { .init(requestID: "request", graphSHA256: sha, sourceBindingsSHA256: sha, revision: "1") }
    private func ledger(_ id: String, _ config: [String: JSONValue]) throws -> PreparationObservationLedger {
        try .init(identity: identity, graph: .init(nodes: [.init(id: "s", operatorID: "kwip.data.sourceCSV"), .init(id: "t", operatorID: id, config: config)], edges: [.init(from: .init(node: "s", port: "table"), to: .init(node: "t", port: "table"))]), outputKinds: ["s": ["table": "table"], "t": ["table": "table"]], expectedSourceDigests: ["s": sha])
    }
    private func event(_ node: String, source: String, role: String = "unassigned", partition: String = "", columns: [String] = ["id"]) -> [String: JSONValue] {
        ["event": .string("dataflowPreview"), "nodeID": .string(node), "execution": .object(["nodeID": .string(node), "status": .string("executed")]), "outputs": .object(["table": .object([
            "kind": .string("table"), "lineage": .object(["role": .string(role), "sourceDigest": .string(source), "membershipDigest": .string(sha), "dataVersion": .string(sha), "partitionDigest": .string(partition), "fittedStateDigests": .array([])]),
            "preview": .object(["partition": .string(role), "dataVersion": .string(sha), "rowCount": .integer(1), "columns": .array(columns.map { .object(["name": .string($0), "dtype": .string("string"), "nullable": .bool(false)]) }), "rows": .array([.array(columns.map { _ in .string("x") })]), "diagnostics": .array([]), "diagnosticCount": .integer(0)])
        ])])]
    }
    func testSplitRowsAuthenticatesQualifiedSourceAndDerivedOrigin() throws {
        var config = SplitTextPracticeExample.configuration; config["outputMode"] = .string("rows")
        var value = try ledger("kwip.data.splitText", config)
        XCTAssertTrue(value.nodeStarted("s", identity: identity)); XCTAssertTrue(value.consume(event("s", source: sha), identity: identity))
        XCTAssertTrue(value.nodeStarted("t", identity: identity))
        XCTAssertFalse(value.consume(event("t", source: sha, columns: ["id", "street"]), identity: identity))
        // Independently computed with Python json.dumps and SHA256.
        XCTAssertTrue(value.consume(event("t", source: "1eb9cbbd65d066b48262c9a64bf626e337f7df25ebf4c7f5136dd7c43e3e9444", columns: ["id", "street"]), identity: identity))
        let output = try XCTUnwrap(value.observation(at: .init(node: "t", port: "table"), current: identity))
        XCTAssertEqual(output.columns[0].origin.nodeID, "s")
        XCTAssertEqual(output.columns[1].origin.nodeID, "t")
    }
    func testRowsRejectsUpstreamPartitionEvenWithMatchingSourceHash() throws {
        var config = SplitTextPracticeExample.configuration; config["outputMode"] = .string("rows")
        var value = try ledger("kwip.data.splitText", config)
        _ = value.nodeStarted("s", identity: identity)
        XCTAssertTrue(value.consume(event("s", source: sha, role: "training", partition: sha), identity: identity))
        _ = value.nodeStarted("t", identity: identity)
        XCTAssertFalse(value.consume(event("t", source: "1eb9cbbd65d066b48262c9a64bf626e337f7df25ebf4c7f5136dd7c43e3e9444"), identity: identity))
    }
    func testReshapeSourceDigestRetainsExplicitFloatJSONRepresentations() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        let defaults = try XCTUnwrap(registry.operatorDefinition(id: "kwip.data.reshapeTable")).defaultConfig
        let fixtures: [(Double, String)] = [
            (1.0, "a57c54805c6f70d2a07d1ae5cb7582261953f93c2171b0cd17bfcacf9aa081b9"),
            (-0.0, "fc15899e488825b905f42f32b68f596562d0e042daa6824b220729a712556e29"),
            (1e-5, "2b96f52439434a987b095d890572b0988fc4a44924dab5b1a2dcbb4aaac0479a"),
            (1e16, "332a53992a42afe31b32f583eb6008979acb227263c6067cae54f520f86df51d"),
            (1e-4, "1e01247e78a1aa2ba5fbec12202782649623fba8857063359d21ae9726e15f0f"),
            (1.25e-20, "bb834fdf39e4bafd51be27bf4deab0b8e0ce25e3b4b9838169de243c8bdb1bb9")]
        for (number, digest) in fixtures {
            var config = defaults; config["fillValue"] = .number(number)
            var value = try ledger("kwip.data.reshapeTable", config)
            _ = value.nodeStarted("s", identity: identity); XCTAssertTrue(value.consume(event("s", source: sha), identity: identity))
            _ = value.nodeStarted("t", identity: identity); XCTAssertTrue(value.consume(event("t", source: digest), identity: identity), "Float \(number)")
        }
    }
}
