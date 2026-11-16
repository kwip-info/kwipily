import Foundation
import KwipMLCore

/// Completeness is independent of successful execution. All claims are bound to
/// the archived immutable demand plan, never inferred from a displayed sample.
enum WorkflowPreparationCoverage {
    struct Receipt: Equatable {
        let value: JSONValue
        let sampledNodeIDs: Set<String>
        let completeNodeIDs: Set<String>
        var isSampled: Bool { !sampledNodeIDs.isEmpty }
        func isComplete(_ node: String) -> Bool { completeNodeIDs.contains(node) }
    }
    static func verify(manifest: Data, graph: Data, plan: Data?, selectedNodes: Set<String>? = nil,
                       allowSample: Bool = false, expectedSampling: JSONValue? = nil,
                       expectedCompleteNodes: Set<String>? = nil,observedTablePorts:[String:Set<String>]? = nil) throws -> Receipt? {
        let envelope = try object(manifest), graphValue = try JSONDecoder().decode(JSONValue.self, from: graph)
        let document = try DataFlowCodec.decodeGraph(graph)
        guard envelope["graphDigest"] == .string(WorkflowArchiveCodec.hash(graph)) else { throw failure("Coverage graph differs from its pinned manifest.") }
        let planValue = try plan.map(object)
        if let bytes = plan, let captured = planValue {
            guard envelope["preparationPlanDigest"] == .string(WorkflowArchiveCodec.hash(try canonicalUTF8(bytes))),
                  captured["graph"] == graphValue, captured["graphDigest"] == envelope["graphDigest"],
                  captured["schemaVersion"] == 1 || captured["schemaVersion"] == 2 || captured["schemaVersion"] == 3 else {
                throw failure("Coverage plan differs from its pinned preparation.")
            }
        }
        if planValue?["schemaVersion"] == 3 {
            guard let policy = planValue?["workloadPolicy"], envelope["workloadPolicy"] == policy else {
                throw failure("Preparation workload policy differs from its pinned plan.")
            }
            _ = try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONEncoder().encode(policy))
        } else if envelope["workloadPolicy"] != nil || planValue?["workloadPolicy"] != nil {
            throw failure("Historical preparation plans cannot claim a new workload policy.")
        }
        guard let raw = envelope["executionCoverage"] else {
            guard planValue?["schemaVersion"] != 2, planValue?["schemaVersion"] != 3, planValue?["demand"] == nil else { throw failure("Preparation demand is missing its completeness receipt.") }
            if let expectedSampling {
                guard case .object(let policy) = expectedSampling, try isCompletePolicy(policy) else { throw failure("Captured sampling requires an explicit completeness receipt.") }
            }
            return nil // Historical preparations predate sampling and are complete.
        }
        guard case .object(let coverage) = raw, let captured = planValue,
              coverage["schemaVersion"] == 1, coverage["graphDigest"] == envelope["graphDigest"],
              coverage["selection"] == .string("sourceOrderPrefix") else { throw failure("Invalid preparation coverage.") }
        let demandKeys: Set<String> = ["schemaVersion", "graphDigest", "sampling", "selection", "completeNodeIDs", "sampledNodeIDs", "percentageRequiresCardinalityScan"]
        guard Set(coverage.keys) == demandKeys.union(["status", "nodes"]) else { throw failure("Unknown preparation coverage fields.") }
        let complete = try strings(coverage["completeNodeIDs"]), sampled = try strings(coverage["sampledNodeIDs"])
        let all = Set(document.nodes.map(\.id))
        guard Set(complete).isDisjoint(with: sampled), Set(complete + sampled) == all else { throw failure("Coverage does not identify every preparation node exactly once.") }
        let order = try topologicalOrder(document)
        guard complete == order.filter({ Set(complete).contains($0) }), sampled == order.filter({ Set(sampled).contains($0) }) else { throw failure("Coverage node order differs from the captured graph.") }
        guard case .object(let policy) = coverage["sampling"] else { throw failure("Missing sampling policy.") }
        let fullPolicy = try isCompletePolicy(policy)
        let requiresCount = policy["mode"] == .string("percentage") && !fullPolicy && !sampled.isEmpty
        guard coverage["percentageRequiresCardinalityScan"] == .bool(requiresCount),
              coverage["status"] == .string(sampled.isEmpty ? "complete" : "sampled"),
              !fullPolicy || sampled.isEmpty else { throw failure("Coverage conflicts with its effective sampling policy.") }
        for edge in document.edges where complete.contains(edge.to.node) {
            guard complete.contains(edge.from.node) else { throw failure("A complete consumer has sampled ancestors.") }
        }
        for node in document.nodes where ["kwip.data.view", "kwip.data.outputDelimited"].contains(node.operatorID) {
            guard complete.contains(node.id) else { throw failure("Table View and full output require complete ancestry.") }
        }
        if captured["schemaVersion"] == 2 || captured["schemaVersion"] == 3 {
            guard case .object(let demand) = captured["demand"], Set(demand.keys) == demandKeys,
                  demand == coverage.filter({ demandKeys.contains($0.key) }) else { throw failure("Execution coverage differs from its immutable demand plan.") }
        } else {
            guard captured["demand"] == nil, sampled.isEmpty, fullPolicy else { throw failure("Historical preparation plans cannot claim sampled execution.") }
        }
        guard case .object(let nodes) = coverage["nodes"], Set(nodes.keys) == all else { throw failure("Missing per-node completeness evidence.") }
        var tablePorts: [String: Set<String>] = [:]
        if case .array(let artifacts) = envelope["artifacts"] {
            for case .object(let artifact) in artifacts where artifact["kind"] == .string("table") {
                guard case .string(let node) = artifact["nodeID"], case .string(let port) = artifact["port"], all.contains(node) else { throw failure("Invalid table coverage port.") }
                tablePorts[node, default: []].insert(port)
            }
        }
        if let observedTablePorts {
            guard envelope["observationRetention"] == .string("requiredOnly"),Set(observedTablePorts.keys)==all else {throw failure("Observation coverage requires the verified required-only output contract.")}
        }
        for node in document.nodes {
            guard case .object(let evidence) = nodes[node.id], Set(evidence.keys) == ["completeness", "rowCounts"],
                  evidence["completeness"] == .string(complete.contains(node.id) ? "complete" : "sampled"),
                  case .object(let counts) = evidence["rowCounts"], Set(counts.keys) == (observedTablePorts?[node.id] ?? tablePorts[node.id,default:[]]),
                  counts.values.allSatisfy({ if case .integer(let count) = $0 { return (0...5_000_000).contains(count) }; return false }) else {
                throw failure("Per-node coverage differs from its artifact inventory.")
            }
        }
        if let expectedSampling {
            guard coverage["sampling"] == expectedSampling else { throw failure("Preparation coverage differs from the captured sampling request.") }
            var requested = expectedCompleteNodes ?? []
            guard requested.isSubset(of: all) else { throw failure("Captured complete-data targets are unknown.") }
            requested.formUnion(document.nodes.filter { ["kwip.data.view", "kwip.data.outputDelimited"].contains($0.operatorID) }.map(\.id))
            if fullPolicy { requested = all }
            var pending = Array(requested)
            while let node = pending.popLast() {
                for edge in document.edges where edge.to.node == node && !requested.contains(edge.from.node) {
                    requested.insert(edge.from.node); pending.append(edge.from.node)
                }
            }
            guard requested == Set(complete) else { throw failure("Preparation coverage differs from captured complete-data targets.") }
        }
        let receipt = Receipt(value: raw, sampledNodeIDs: Set(sampled), completeNodeIDs: Set(complete))
        if let selectedNodes {
            guard !selectedNodes.isEmpty, selectedNodes.isSubset(of: all) else { throw failure("Unknown complete-data selection.") }
            var required = selectedNodes, pending = Array(selectedNodes)
            while let node = pending.popLast() {
                for edge in document.edges where edge.to.node == node && !required.contains(edge.from.node) {
                    required.insert(edge.from.node); pending.append(edge.from.node)
                }
            }
            guard required.isSubset(of: receipt.completeNodeIDs) else { throw failure("This model requires complete prepared data and fitted states. Prepare its selected branch completely.") }
        } else if !allowSample && receipt.isSampled {
            throw failure("Sampled preparation cannot satisfy a complete-data consumer.")
        }
        return receipt
    }
    private static func object(_ bytes: Data) throws -> [String: JSONValue] {
        guard bytes.count <= 2_000_000, case .object(let value) = try JSONDecoder().decode(JSONValue.self, from: bytes) else { throw failure("Invalid bounded coverage metadata.") }
        return value
    }
    private static func strings(_ value: JSONValue?) throws -> [String] {
        guard case .array(let entries) = value, entries.count <= 128 else { throw failure("Invalid coverage node identities.") }
        let strings = try entries.map { entry -> String in
            guard case .string(let value) = entry, !value.isEmpty, value.count <= 128 else { throw failure("Invalid coverage node identity.") }; return value
        }
        guard Set(strings).count == strings.count else { throw failure("Duplicate coverage node identity.") }; return strings
    }
    private static func isCompletePolicy(_ policy: [String: JSONValue]) throws -> Bool {
        guard policy["schemaVersion"] == 1 else { throw failure("Unsupported sampling policy.") }
        switch policy["mode"] {
        case .some(.string("all")): guard Set(policy.keys) == ["schemaVersion", "mode"] else { throw failure("Inactive sampling fields.") }; return true
        case .some(.string("rowCount")): guard Set(policy.keys) == ["schemaVersion", "mode", "rowCount"], case .integer(let count) = policy["rowCount"], (1...5_000_000).contains(count) else { throw failure("Invalid sample row count.") }; return false
        case .some(.string("percentage")):
            guard Set(policy.keys) == ["schemaVersion", "mode", "percentage"] else { throw failure("Inactive sampling fields.") }
            let value: Double
            switch policy["percentage"] { case .integer(let count): value = Double(count); case .number(let number): value = number; default: throw failure("Invalid sample percentage.") }
            guard value.isFinite, value > 0, value <= 100 else { throw failure("Invalid sample percentage.") }; return value == 100
        default: throw failure("Unknown sampling policy.")
        }
    }
    private static func topologicalOrder(_ graph: DataFlowDocument) throws -> [String] {
        var order: [String] = []
        while order.count < graph.nodes.count {
            let ready = graph.nodes.filter { node in !order.contains(node.id) && graph.edges.filter({ $0.to.node == node.id }).allSatisfy({ order.contains($0.from.node) }) }.map(\.id)
            guard !ready.isEmpty else { throw failure("Cyclic preparation coverage graph.") }; order += ready
        }
        return order
    }
    /// Python's plan identity uses UTF-8 strings and preserves numeric tokens.
    /// Converting through JSONEncoder would change e.g. 1.0 to 1 and lose its pin.
    static func canonicalUTF8(_ data: Data) throws -> Data {
        _ = try object(data)
        let bytes = Array(data); var result = Data(), cursor = 0
        while cursor < bytes.count {
            let byte = bytes[cursor]
            if [9, 10, 13, 32].contains(byte) { cursor += 1; continue }
            guard byte == 34 else { result.append(byte); cursor += 1; continue }
            let start = cursor; cursor += 1
            while cursor < bytes.count {
                if bytes[cursor] == 92 { cursor += 2; continue }
                if bytes[cursor] == 34 { break }; cursor += 1
            }
            guard cursor < bytes.count else { throw failure("Unterminated coverage plan string.") }
            let string = try JSONDecoder().decode(String.self, from: Data(bytes[start...cursor])); cursor += 1
            result.append(34)
            for scalar in string.unicodeScalars {
                switch scalar.value {
                case 34: result.append(contentsOf: [92,34])
                case 92: result.append(contentsOf: [92,92])
                case 8: result.append(contentsOf: [92,98])
                case 12: result.append(contentsOf: [92,102])
                case 10: result.append(contentsOf: [92,110])
                case 13: result.append(contentsOf: [92,114])
                case 9: result.append(contentsOf: [92,116])
                case 0...31: result.append(contentsOf: String(format: "\\u%04x", scalar.value).utf8)
                default: result.append(contentsOf: String(scalar).utf8)
                }
            }
            result.append(34)
        }
        return result
    }
    private static func failure(_ message: String) -> WorkflowArchiveError { WorkflowArchiveError(message) }
}
