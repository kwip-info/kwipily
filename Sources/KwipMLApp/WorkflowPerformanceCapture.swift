import Foundation
import KwipMLCore

/// One verified representative forward. RSS samples never imply layer allocation or peak memory.
struct WorkflowPerformanceCapture: Equatable {
    struct Record: Equatable, Identifiable {
        let nodeID: String
        var id: String { nodeID }
        let durationNanoseconds: Int64?
        let timingUnavailableReason: String?
        let residentBytesBefore: Int64?
        let residentBytesAfter: Int64?
        let memoryUnavailableReason: String?
    }
    let device: String
    let records: [Record]
    static let maximumBytes = 131_072
    static func decode(_ data: Data, metadata: JSONValue, runID: String, graphHash: String,
                       device: String, graph: GraphDocument, policy: TrainingSampleCapturePolicy) throws -> Self {
        func refuse() -> WorkflowPhaseError { .init("Performance observations differ from their captured identity, policy, bounds or operation order.") }
        guard policy.performanceMeasurements, (1...maximumBytes).contains(data.count),
              !runID.isEmpty, runID.count <= 128, WorkflowArchiveCodec.isHash(graphHash),
              ["cpu", "mps"].contains(device), case .object(let pin) = metadata,
              Set(pin.keys) == ["schemaVersion", "file", "bytes", "sha256"],
              pin["schemaVersion"] == 1, pin["file"] == .string("performance.json"),
              pin["bytes"] == .integer(Int64(data.count)), pin["sha256"] == .string(WorkflowPhaseArtifacts.hash(data)) else { throw refuse() }
        let value = try JSONDecoder().decode([String:JSONValue].self, from: data)
        let encodedPolicy = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(policy))
        guard Set(value.keys) == ["schemaVersion", "runID", "graphHash", "device", "stage", "selection", "maximumRows", "policy", "timingMethod", "memoryMethod", "records"],
              value["schemaVersion"] == 1, value["runID"] == .string(runID), value["graphHash"] == .string(graphHash),
              value["device"] == .string(device), value["stage"] == .string("validationCapture"), value["selection"] == .string("firstValidationRows"),
              value["maximumRows"] == 8, value["policy"] == encodedPolicy,
              value["timingMethod"] == .string(device == "cpu" ? "hostWall" : "synchronizedHostWall"),
              value["memoryMethod"] == .string("processResidentBytes"), case .array(let rows) = value["records"],
              (1...128).contains(rows.count), rows.count == graph.nodes.count,
              Set(graph.nodes.map(\.id)).count == graph.nodes.count else { throw refuse() }
        let ids = Set(graph.nodes.map(\.id))
        guard graph.edges.allSatisfy({ ids.contains($0.source) && ids.contains($0.target) }) else { throw refuse() }
        func number(_ value: JSONValue?, maximum: Int64, minimum: Int64 = 0) throws -> Int64? {
            if value == .null { return nil }
            guard case .integer(let n) = value, n >= minimum, n <= maximum else { throw refuse() }
            return n
        }
        func reason(_ value: JSONValue?, required: Bool, allowed: Set<String>) throws -> String? {
            if !required, value == .null { return nil }
            guard required, case .string(let text) = value, allowed.contains(text) else { throw refuse() }
            return text
        }
        var visited = Set<String>(), records: [Record] = []
        for row in rows {
            guard case .object(let fields) = row,
                  Set(fields.keys) == ["nodeID", "durationNanoseconds", "timingUnavailableReason", "residentBytesBefore", "residentBytesAfter", "memoryUnavailableReason"],
                  case .string(let id) = fields["nodeID"], ids.contains(id), !visited.contains(id),
                  graph.edges.filter({ $0.target == id }).allSatisfy({ visited.contains($0.source) }) else { throw refuse() }
            let duration = try number(fields["durationNanoseconds"], maximum: 86_400_000_000_000)
            let timing = try reason(fields["timingUnavailableReason"], required: duration == nil,
                                    allowed: device == "mps" ? ["clockUnavailable", "deviceSynchronizationUnavailable"] : ["clockUnavailable"])
            let before = try number(fields["residentBytesBefore"], maximum: Int64.max, minimum: 1)
            let after = try number(fields["residentBytesAfter"], maximum: Int64.max, minimum: 1)
            let memory = try reason(fields["memoryUnavailableReason"], required: before == nil || after == nil, allowed: ["residentMemoryUnavailable"])
            records.append(.init(nodeID: id, durationNanoseconds: duration, timingUnavailableReason: timing,
                                 residentBytesBefore: before, residentBytesAfter: after, memoryUnavailableReason: memory))
            visited.insert(id)
        }
        guard visited == ids else { throw refuse() }
        return .init(device: device, records: records)
    }
}
