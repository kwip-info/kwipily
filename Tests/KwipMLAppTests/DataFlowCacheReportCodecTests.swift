import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowCacheReportCodecTests: XCTestCase {
    private let fixtureDigest = String(repeating:"a",count:64)
    private let pin = String(repeating:"b",count:64)
    private var graph:DataFlowDocument { .init(nodes:[
        .init(id:"source",operatorID:"kwip.data.sourceCSV"),
        .init(id:"cast",operatorID:"kwip.data.cast"),
        .init(id:"split",operatorID:"kwip.data.split")]) }
    private func decision(_ id:String,_ status:String,_ reason:String)->[String:Any] {
        ["nodeID":id,"cacheKey":fixtureDigest,"status":status,"reason":reason]
    }
    private func report(_ policy:String)->[String:Any] {
        let reuse = policy == "reuse"
        return ["schemaVersion":1,"policy":policy,"implementationDigest":fixtureDigest,
            "sourceManifestSHA256":reuse ? pin as Any : NSNull(),
            "nodes":policy == "off" ? [] : [decision("source","executed","sourceRevalidated"),decision("cast",reuse ? "reused":"executed",reuse ? "verifiedHit":"notMaterialized"),decision("split","executed",reuse ? "identityChanged":"notMaterialized")],
            "executedNodeIDs":reuse ? ["source","split"]:["source","cast","split"],
            "reusedNodeIDs":reuse ? ["cast"]:[]]
    }
    private func bytes(_ object:[String:Any])throws->Data {try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])}
    func testLargerCacheReportRequiresCapturedPolicyAndPreservesLegacyRefusal() throws {
        let workload = try ExecutionWorkloadPolicy(preset: .balanced)
        var cache = report("materialize")
        cache["materializedBytesLimit"] = 300_000_000
        cache["materializedBytes"] = 250_000_000
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(cache), graph: graph))
        let expectation = DataFlowCacheExpectation(policy: .materialize, sourceManifestSHA256: nil, materializedBytesLimit: 300_000_000, workload: workload)
        XCTAssertEqual(try DataFlowCacheReportCodec.decodeVerified(bytes(cache), graph: graph, expectation: expectation).materializedBytes, 250_000_000)
        let policy = try JSONSerialization.jsonObject(with: JSONEncoder().encode(workload))
        var manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "cache": cache,
            "tableStorageVersion": 1, "workloadPolicy": policy, "artifacts": [["kind": "tableSegment", "path": "table-pack-" + String(repeating: "a", count: 32) + ".kwipdata", "bytes": 250_000_000]]]
        XCTAssertEqual(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest), graph: graph).materializedBytes, 250_000_000)
        let narrower = try workload.overriding(["retainedBytes": 100])
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest), graph: graph, expectation: .init(policy: .materialize, sourceManifestSHA256: nil, workload: narrower)))
        manifest.removeValue(forKey: "tableStorageVersion")
        manifest.removeValue(forKey: "workloadPolicy")
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest), graph: graph))
    }
    func testPinnedAllowanceRequiresPairedMeasuredFieldsAndExactInventoryBytes() throws {
        var cache = report("materialize")
        cache["materializedBytesLimit"] = 100
        cache["materializedBytes"] = 30
        let expected = DataFlowCacheExpectation(policy: .materialize, sourceManifestSHA256: nil, materializedBytesLimit: 100)
        let decoded = try DataFlowCacheReportCodec.decodeVerified(bytes(cache), graph: graph, expectation: expected)
        XCTAssertEqual(decoded.materializedBytesLimit, 100)
        XCTAssertEqual(decoded.materializedBytes, 30)
        var manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "cache": cache,
            "artifacts": [["kind": "table", "bytes": 20], ["kind": "state", "bytes": 10], ["kind": "graph", "bytes": 50]]]
        XCTAssertEqual(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest), graph: graph, expectation: expected), decoded)
        manifest["artifacts"] = [["kind": "table", "bytes": 29]]
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest), graph: graph, expectation: expected))
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(report("materialize")), graph: graph, expectation: expected))
        for mutation: (inout [String: Any]) -> Void in [
            { $0.removeValue(forKey: "materializedBytes") }, { $0["materializedBytes"] = true },
            { $0["materializedBytes"] = -1 }, { $0["materializedBytes"] = 101 },
            { $0["materializedBytesLimit"] = 0 }, { $0["materializedBytesLimit"] = 200_000_001 },
            { $0["materializedBytesLimit"] = 99 }
        ] {
            var changed = cache; mutation(&changed)
            XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(changed), graph: graph, expectation: expected))
        }
        var off = report("off"); off["materializedBytesLimit"] = 100; off["materializedBytes"] = 0
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(off), graph: graph))
        let legacy = try DataFlowCacheReportCodec.decodeVerified(bytes(report("materialize")), graph: graph)
        XCTAssertNil(legacy.materializedBytes)
    }

    func testUnicodeNodeIDsUseWorkerScalarCountInsteadOfUTF8Bytes() throws {
        for unit in ["é", "😀"] {
            let accepted = String(repeating:unit,count:128)
            let unicodeGraph = DataFlowDocument(nodes:[.init(id:accepted,operatorID:"kwip.data.sourceCSV")])
            var object = report("off")
            object["executedNodeIDs"] = [accepted]
            let value = try DataFlowCacheReportCodec.decodeVerified(bytes(object),graph:unicodeGraph,
                expectation:.init(policy:.off,sourceManifestSHA256:nil))
            XCTAssertEqual(value.executedNodeIDs,[accepted])
            let rejected = accepted + unit
            let overlongGraph = DataFlowDocument(nodes:[.init(id:rejected,operatorID:"kwip.data.sourceCSV")])
            object["executedNodeIDs"] = [rejected]
            XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(object),graph:overlongGraph))
        }
        // A grapheme-cluster count would accept 129 scalars as 65 visible characters.
        let decomposed = String(repeating:"e\u{301}",count:64) + "e"
        let graph = DataFlowDocument(nodes:[.init(id:decomposed,operatorID:"kwip.data.sourceCSV")])
        var object = report("off");object["executedNodeIDs"] = [decomposed]
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(object),graph:graph))
    }

    func testOffHasEmptyDecisionsButAllNodesExecuted()throws {
        let value = try DataFlowCacheReportCodec.decodeVerified(bytes(report("off")),graph:graph,
            expectation:.init(policy:.off,sourceManifestSHA256:nil))
        XCTAssertEqual(value.policy,"off");XCTAssertTrue(value.nodes.isEmpty)
        XCTAssertEqual(Set(value.executedNodeIDs),Set(graph.nodes.map(\.id)))
        XCTAssertTrue(value.reusedNodeIDs.isEmpty)
        XCTAssertEqual(value.implementationDigest,fixtureDigest)
    }
    func testMaterializeAndReuseHaveCompleteDistinctDecisionEvidence()throws {
        for policy in [DataFlowCachePolicy.materialize,.reuse] {
            let value = try DataFlowCacheReportCodec.decodeVerified(bytes(report(policy.rawValue)),graph:graph,
                expectation:.init(policy:policy,sourceManifestSHA256:policy == .reuse ? pin:nil))
            XCTAssertEqual(value.nodes.count,3)
            XCTAssertEqual(value.nodes.first?.cacheKey,fixtureDigest)
            XCTAssertEqual(value.sourceManifestSHA256,policy == .reuse ? pin:nil)
        }
    }
    func testRejectsMissingDuplicateForeignAndContradictoryNodeEvidence()throws {
        let modifications:[(inout [String:Any])->Void] = [
            {$0["executedNodeIDs"] = ["source"]},
            {$0["executedNodeIDs"] = ["source","source","split"]},
            {$0["executedNodeIDs"] = ["foreign","split"]},
            {$0["reusedNodeIDs"] = ["cast","source"]},
            {$0["nodes"] = [self.decision("source","executed","sourceRevalidated")]},
            {$0["nodes"] = [self.decision("source","reused","verifiedHit"),self.decision("cast","reused","verifiedHit"),self.decision("split","executed","identityChanged")]},
            {$0["nodes"] = [self.decision("source","executed","notMaterialized"),self.decision("cast","reused","verifiedHit"),self.decision("split","executed","identityChanged")]},
            {$0["nodes"] = [self.decision("source","executed","sourceRevalidated"),self.decision("cast","reused","identityChanged"),self.decision("split","executed","identityChanged")]}
        ]
        for modify in modifications {var object=report("reuse");modify(&object);XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(object),graph:graph))}
        var materialized=report("materialize")
        materialized["nodes"]=[decision("source","executed","sourceRevalidated"),decision("cast","executed","identityChanged"),decision("split","executed","notMaterialized")]
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(materialized),graph:graph))
    }
    func testStrictSchemaPolicyPinsAndBounds()throws {
        let modifications:[(inout [String:Any])->Void] = [
            {$0["extra"] = 1},{$0["schemaVersion"] = true},{$0["implementationDigest"] = "invalid"},
            {$0.removeValue(forKey:"implementationDigest")},{$0.removeValue(forKey:"sourceManifestSHA256")},
            {$0["sourceManifestSHA256"] = NSNull()},{$0["policy"] = "unknown"},
            {$0["nodes"] = Array(repeating:self.decision("cast","reused","verifiedHit"),count:129)}
        ]
        for modify in modifications {var object=report("reuse");modify(&object);XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(object),graph:graph))}
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(report("reuse")),graph:graph,expectation:.init(policy:.reuse,sourceManifestSHA256:fixtureDigest)))
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(report("materialize")),graph:graph,expectation:.init(policy:.off,sourceManifestSHA256:nil)))
        var off=report("off");off["sourceManifestSHA256"]=pin
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(bytes(off),graph:graph))
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeVerified(Data(repeating:32,count:256_001),graph:graph))
    }
    func testManifestUsesStoredReportAndRequiresExactObservedReportMatch()throws {
        let observed = try DataFlowCacheReportCodec.decodeVerified(bytes(report("reuse")),graph:graph)
        let manifest:[String:Any] = ["schemaVersion":1,"kind":"dataflowExecution","status":"completed","cache":report("reuse")]
        XCTAssertEqual(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest),graph:graph,expectation:.init(policy:.reuse,sourceManifestSHA256:pin),observedReport:observed),observed)
        var changed=report("reuse");changed["implementationDigest"]=pin
        let different = try DataFlowCacheReportCodec.decodeVerified(bytes(changed),graph:graph)
        XCTAssertThrowsError(try DataFlowCacheReportCodec.decodeManifest(bytes(manifest),graph:graph,expectation:.init(policy:.reuse,sourceManifestSHA256:pin),observedReport:different))
    }
}
