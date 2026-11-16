import Foundation
import KwipMLCore

/// Workload capacity is captured by the operation. Metadata and individual
/// decoder frames retain their independent protocol boundaries.
struct PreparationArtifactPolicy {
    let totalBytes: Int
    let inventoryCount: Int
    private let segmentPaths: Set<String>
    init(manifest data: Data) throws {
        guard data.count <= 2_000_000 else { throw WorkflowFileLeaseError.unsafePath }
        let root = try JSONDecoder().decode([String: JSONValue].self, from: data)
        guard case .array(let inventory) = root["artifacts"] else { throw WorkflowFileLeaseError.unsafePath }
        var segments = Set<String>()
        for case .object(let entry) in inventory where entry["kind"] == .string("tableSegment") {
            guard let path = entry["path"]?.stringValue,
                  path.hasPrefix("table-pack-"), path.hasSuffix(".kwipdata"),
                  path.count == 52,
                  path.dropFirst(11).dropLast(9).allSatisfy({ "0123456789abcdef".contains($0) }),
                  segments.insert(path).inserted else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let version = root["tableStorageVersion"] {
            guard version == .integer(1), let value = root["workloadPolicy"] else { throw WorkflowFileLeaseError.unsafePath }
            let policy = try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONEncoder().encode(value))
            totalBytes = Int(policy.limit(.retainedBytes))
            // The flat inventory's 2 MB metadata frame is still bounded.
            inventoryCount = min(62_500, Int(min(62_500, policy.limit(.planNodes))) * 32 + Int(min(62_500, policy.limit(.collectionMembers))) + 128)
        } else {
            guard segments.isEmpty else { throw WorkflowFileLeaseError.unsafePath }
            totalBytes = 200_000_000; inventoryCount = 300
        }
        if root["tableStorageVersion"] != nil {
            for case .object(let entry) in inventory where entry["kind"] == .string("tableViewIndex") {
                guard let path = entry["path"]?.stringValue, Self.isIndexName(path), segments.insert(path).inserted else {throw WorkflowFileLeaseError.unsafePath}
            }
        }
        segmentPaths = segments
    }
    static func inventoryCount(_ policy: ExecutionWorkloadPolicy) -> Int {
        min(62_500, Int(min(62_500, policy.limit(.planNodes))) * 32 + Int(min(62_500, policy.limit(.collectionMembers))) + 128)
    }
    static func isSegmentName(_ name: String) -> Bool {
        name.hasPrefix("table-pack-") && name.hasSuffix(".kwipdata") && name.utf8.count == 52
            && name.dropFirst(11).dropLast(9).allSatisfy { "0123456789abcdef".contains($0) }
    }
    static func isIndexName(_ name: String) -> Bool {
        name.hasPrefix("view-") && name.hasSuffix(".sqlite3") && (16...18).contains(name.utf8.count)
            && name.dropFirst(5).dropLast(8).allSatisfy { "0123456789".contains($0) }
    }
    static func isCapacityFileName(_ name: String) -> Bool { isSegmentName(name) || isIndexName(name) }
    func fileBytes(_ path: String) -> Int {
        segmentPaths.contains(URL(fileURLWithPath: path).lastPathComponent) ? totalBytes : 50_000_000
    }
}
