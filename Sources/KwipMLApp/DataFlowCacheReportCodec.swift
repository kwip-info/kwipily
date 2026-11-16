import Foundation
import KwipMLCore

struct DataFlowCacheExpectation: Equatable {
    let policy: DataFlowCachePolicy
    let sourceManifestSHA256: String?
    var materializedBytesLimit: Int? = nil
    var workload: ExecutionWorkloadPolicy? = nil
}

/// Verifies a historical execution report, not whether its artifacts are reusable
/// with today's sources/runtime. The existing report type also supports old partial
/// UI projections; only this decoder promotes a complete worker report to evidence.
enum DataFlowCacheReportCodec {
    static func decodeVerified(_ bytes: Data, graph: DataFlowDocument,
                               expectation: DataFlowCacheExpectation? = nil) throws -> DataFlowCacheReport {
        try require(!bytes.isEmpty && bytes.count <= 256_000, "Cache report exceeds its byte limit.")
        let object = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        let keys: Set<String> = ["schemaVersion", "policy", "implementationDigest", "sourceManifestSHA256", "nodes", "executedNodeIDs", "reusedNodeIDs"]
        try require(Set(object.keys) == keys || Set(object.keys) == keys.union(["materializedBytesLimit", "materializedBytes"]), "Cache report fields are missing or unsupported.")
        guard case .integer(1) = object["schemaVersion"],
              case .array(let decisions) = object["nodes"] else {
            throw DataFlowContractError("Unsupported cache report schema.")
        }
        try require(decisions.count <= 128, "Cache report decision count exceeds its bound.")
        for decision in decisions {
            guard case .object(let fields) = decision else { throw DataFlowContractError("Invalid cache decision.") }
            try require(Set(fields.keys) == Set(["nodeID", "cacheKey", "status", "reason"]), "Cache decision fields are missing or unsupported.")
            guard case .string(let key) = fields["cacheKey"], hash(key) else { throw DataFlowContractError("Invalid cache decision key.") }
        }
        let value = try JSONDecoder().decode(DataFlowCacheReport.self, from: bytes)
        guard let policy = DataFlowCachePolicy(rawValue: value.policy), let implementation = value.implementationDigest,
              hash(implementation) else { throw DataFlowContractError("Invalid cache policy or implementation digest.") }
        let retainedMaximum: Int64 = expectation?.workload?.limit(.retainedBytes) ?? 200_000_000
        if object["materializedBytesLimit"] != nil {
            guard policy != .off, case .integer(let limit) = object["materializedBytesLimit"],
                case .integer(let count) = object["materializedBytes"],
                (1...retainedMaximum).contains(limit), (0...limit).contains(count) else {
                throw DataFlowContractError("Invalid preparation cache allowance or measured byte count.")
            }
        }
        if let limit = expectation?.materializedBytesLimit {
            try require(value.materializedBytesLimit == limit, "Cache allowance differs from the captured request or is missing.")
        }
        if policy == .reuse {
            guard case .string(let pin) = object["sourceManifestSHA256"], hash(pin) else {
                throw DataFlowContractError("Cache reuse requires its selected manifest pin.")
            }
        } else {
            guard case .null = object["sourceManifestSHA256"] else {
                throw DataFlowContractError("Only cache reuse may identify a selected manifest.")
            }
        }
        if let expectation {
            try require(expectation.policy == policy && expectation.sourceManifestSHA256 == value.sourceManifestSHA256,
                        "Cache report differs from requested policy or selected manifest pin.")
        }
        let ids = graph.nodes.map(\.id), expected = Set(ids)
        try require((1...128).contains(ids.count) && expected.count == ids.count && ids.allSatisfy(validID), "Invalid graph identities for cache report verification.")
        let executed = Set(value.executedNodeIDs), reused = Set(value.reusedNodeIDs)
        try require(value.executedNodeIDs.count <= 128 && value.reusedNodeIDs.count <= 128 && executed.count == value.executedNodeIDs.count && reused.count == value.reusedNodeIDs.count,
                    "Cache report has duplicate or excessive execution identities.")
        try require(executed.isDisjoint(with: reused) && executed.union(reused) == expected,
                    "Cache execution identities do not completely and uniquely cover the captured graph.")
        let sources = Set(graph.nodes.filter { DataFlowSourceOperators.contains($0.operatorID) }.map(\.id))
        try require(sources.isSubset(of: executed), "Sources must be executed again, never reused.")
        if policy == .off {
            try require(value.nodes.isEmpty && reused.isEmpty && executed == expected,
                        "Disabled caching must execute all nodes without cache decisions.")
            return value
        }
        let decisionIDs = value.nodes.map(\.nodeID)
        try require(Set(decisionIDs) == expected && decisionIDs.count == expected.count,
                    "Cache decisions do not completely and uniquely cover the captured graph.")
        if policy == .materialize { try require(reused.isEmpty, "Materialization cannot report reused nodes.") }
        for decision in value.nodes {
            let didReuse = reused.contains(decision.nodeID)
            try require(decision.status == (didReuse ? "reused" : "executed"), "Cache decision status differs from its execution index.")
            if sources.contains(decision.nodeID) {
                try require(!didReuse && decision.reason == "sourceRevalidated", "Source cache reason must disclose revalidation.")
            } else if didReuse {
                try require(policy == .reuse && decision.reason == "verifiedHit", "Reused node must report a verified cache hit.")
            } else {
                try require(decision.reason == "notMaterialized" || (policy == .reuse && decision.reason == "identityChanged"),
                            "Executed node has a cache reason inconsistent with policy.")
            }
        }
        return value
    }

    /// Extracts from the caller's already-read manifest bytes. This verifies cache
    /// provenance only: phase inventory/graph/hash validation remains authoritative.
    static func decodeManifest(_ bytes: Data, graph: DataFlowDocument,
                               expectation: DataFlowCacheExpectation? = nil,
                               observedReport: DataFlowCacheReport? = nil) throws -> DataFlowCacheReport {
        try require(!bytes.isEmpty && bytes.count <= 2_000_000, "Preparation manifest exceeds its byte limit.")
        let object = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        guard case .integer(1) = object["schemaVersion"], case .string("dataflowExecution") = object["kind"],
              case .string("completed") = object["status"], case .object(let report) = object["cache"] else {
            throw DataFlowContractError("Preparation manifest lacks completed cache provenance.")
        }
        var capturedExpectation = expectation
        var retainedMaximum = 200_000_000
        if let version = object["tableStorageVersion"] {
            guard version == .integer(1), let policy = object["workloadPolicy"] else { throw DataFlowContractError("Cache storage policy is missing.") }
            retainedMaximum = try PreparationArtifactPolicy(manifest: bytes).totalBytes
            let workload = try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONEncoder().encode(policy))
            if let expected = expectation?.workload { try require(workload == expected, "Cache storage policy differs from the captured request.") }
            if capturedExpectation == nil {
                guard let raw = report["policy"]?.stringValue, let policy = DataFlowCachePolicy(rawValue: raw) else { throw DataFlowContractError("Invalid cache policy.") }
                capturedExpectation = .init(policy: policy, sourceManifestSHA256: report["sourceManifestSHA256"]?.stringValue)
            }
            capturedExpectation?.workload = workload
        } else { try require(expectation?.workload == nil, "Cache storage omitted the captured policy.") }
        let value = try decodeVerified(JSONEncoder().encode(report), graph: graph, expectation: capturedExpectation)
        if let count = value.materializedBytes {
            guard case .array(let artifacts) = object["artifacts"] else { throw DataFlowContractError("Cache byte evidence requires its artifact inventory.") }
            var total = 0
            for artifact in artifacts {
                guard case .object(let item) = artifact, case .string(let kind) = item["kind"] else { throw DataFlowContractError("Invalid cache artifact inventory.") }
                if kind == "table" || kind == "state" || kind == "tableSegment" {
                    guard case .integer(let bytes) = item["bytes"], (0...Int64(retainedMaximum)).contains(bytes), Int64(total) <= Int64(retainedMaximum) - bytes else {
                        throw DataFlowContractError("Invalid cache artifact byte count.")
                    }
                    total += Int(bytes)
                }
            }
            try require(count == total, "Measured cache bytes differ from the pinned artifact inventory.")
        }
        if let observedReport { try require(value == observedReport, "Observed cache report differs from the stored manifest report.") }
        return value
    }
    private static func hash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    private static func validID(_ value: String) -> Bool { !value.isEmpty && value.unicodeScalars.count <= 128 && !value.contains("\0") }
    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw DataFlowContractError(message) }
    }
}
