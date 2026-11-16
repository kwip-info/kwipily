import Foundation
import KwipMLCore

struct WorkflowComparisonField: Identifiable, Equatable {
    let path: String
    let baseline: String
    let candidate: String
    var id: String { path }
}

/// JSON Pointer paths preserve owner-defined keys without collisions. Values retain
/// JSON types; absent fields are different from explicit null or empty containers.
enum WorkflowComparisonFields {
    static func differences(_ baseline: JSONValue, _ candidate: JSONValue) throws -> [WorkflowComparisonField] {
        let a = try flatten(baseline), b = try flatten(candidate)
        return Set(a.keys).union(b.keys).sorted().compactMap { path in
            guard a[path] != b[path] else { return nil }
            return .init(path: path, baseline: a[path] ?? "Not present", candidate: b[path] ?? "Not present")
        }
    }
    static func json<T: Encodable>(_ value: T) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value))
    }
    private static func flatten(_ value: JSONValue, path: String = "") throws -> [String: String] {
        func escape(_ key: String) -> String { key.replacingOccurrences(of: "~", with: "~0").replacingOccurrences(of: "/", with: "~1") }
        switch value {
        case .object(let fields) where !fields.isEmpty:
            var result: [String: String] = [:]
            for key in fields.keys.sorted() { result.merge(try flatten(fields[key]!, path: path + "/" + escape(key))) { _, new in new } }
            return result
        case .array(let values) where !values.isEmpty:
            var result: [String: String] = [:]
            for (index, item) in values.enumerated() { result.merge(try flatten(item, path: path + "/\(index)")) { _, new in new } }
            return result
        default:
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            return [path.isEmpty ? "/" : path: String(decoding: try encoder.encode(value), as: UTF8.self)]
        }
    }
    static func configuration(_ value: VerifiedWorkflowOutcome) throws -> JSONValue {
        let plan = value.plan
        let nodes = Dictionary(uniqueKeysWithValues: plan.dataflow.nodes.map { ($0.id, JSONValue.object(["operator": .string($0.operatorID), "config": .object($0.config)])) })
        let sources = Dictionary(uniqueKeysWithValues: value.capture.sources.map { ($0.address.domain.rawValue + ":" + $0.address.localID, JSONValue.string($0.sha256)) })
        return .object([
            "modelIdentity": .string(plan.model.semanticSHA256),
            "settings": try json(plan.settings),
            "dataComponents": .object(nodes),
            "dataConnections": try json(plan.dataflow.edges.sorted { $0.id < $1.id }),
            "sourceContent": .object(sources),
            "trainingOutput": try json(plan.training), "validationOutput": try json(plan.validation),
            "features": try json(plan.featureColumns), "target": plan.targetColumn.map(JSONValue.string) ?? .null,
            "trainingMode": try json(plan.trainingMode),
            "cachePolicy": try json(plan.preparationCachePolicy),
            "checkpointParent": try json(value.capture.checkpointParent),
            "modelRegistry": .string(value.capture.modelRegistrySHA256),
            "dataRegistry": .string(value.capture.dataRegistrySHA256),
            "runtimeEvidence": .string(value.capture.runtimeEvidence.sha256)
        ])
    }
}
