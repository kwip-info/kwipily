import Foundation
import KwipMLCore

/// Finite host-phase observations, separate from table capture and training profiles.
/// Deliberately not ObservableObject: one worker event must not redraw the workspace.
struct PreparationPhaseMeasurements {
    static let phases: Set<String> = ["dispatch", "cacheKey", "cacheLookup", "execution", "observation", "release"]
    private(set) var records: [String: [String: JSONValue]] = [:]

    mutating func consume(_ bytes: Data, allowedNodes: Set<String>, enabled: Bool) throws {
        func refuse() -> WorkflowArchiveError { .init("Preparation timing differs from its captured policy, component or finite metadata contract.") }
        guard enabled, bytes.count <= 4096,
              let event = try? JSONDecoder().decode([String: JSONValue].self, from: bytes),
              Set(event.keys) == ["event", "measurement"], event["event"] == .string("dataflowNodeTiming"),
              case .object(let fields) = event["measurement"],
              case .string(let node) = fields["nodeID"], allowedNodes.contains(node), records[node] == nil,
              case .string(let status) = fields["status"], ["completed", "failed", "interrupted"].contains(status),
              case .bool = fields["cacheHit"],
              Set(fields.keys) == Set(["nodeID", "status", "cacheHit", "milliseconds", "wallMilliseconds", "unattributedMilliseconds"] + (status == "completed" ? [] : ["failurePhase"])),
              case .object(let durations) = fields["milliseconds"], Set(durations.keys) == Self.phases else { throw refuse() }
        func duration(_ value: JSONValue?) throws -> Double {
            let number: Double
            switch value {
            case .integer(let n): number = Double(n)
            case .number(let n): number = n
            default: throw refuse()
            }
            guard number.isFinite, number >= 0 else { throw refuse() }
            return number
        }
        let wall = try duration(fields["wallMilliseconds"])
        let unattributed = try duration(fields["unattributedMilliseconds"])
        let total = try durations.values.reduce(0.0) { try $0 + duration($1) }
        guard total.isFinite, total <= wall + 0.01,
              abs(total + unattributed - wall) <= 0.01 else { throw refuse() }
        if status != "completed" {
            guard case .string(let phase) = fields["failurePhase"], Self.phases.union(["boundary"]).contains(phase) else { throw refuse() }
        }
        records[node] = fields
    }
}
