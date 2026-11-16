import Foundation
import KwipMLCore

/// Structural admission after the caller has authenticated this manifest and its
/// inventory. The index is re-hashed again by the captured query operation.
enum TableViewArtifactReference {
    static func parse(manifest: [String: JSONValue], directory: URL, expectedNodes: Set<String>) throws -> [String: TableViewQuerySession.Artifact] {
        let views: [String: JSONValue]
        if case .object(let value) = manifest["tableViews"] { views = value }
        else { guard expectedNodes.isEmpty else { throw failure() }; return [:] }
        guard Set(views.keys) == expectedNodes, case .array(let inventory) = manifest["artifacts"],
              case .object(let coverage) = manifest["executionCoverage"], case .object(let nodes) = coverage["nodes"] else { throw failure() }
        let tokenContext=try TokenVocabularyResolver.make(manifest:manifest,directory:directory,endpoint:.init(node:"",port:""))
        let workload = try manifest["workloadPolicy"].map { try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONEncoder().encode($0)) }
        let capacity = TableTransferCapacity(workload: workload)
        var result: [String: TableViewQuerySession.Artifact] = [:]
        for (node, value) in views {
            guard case .object(let view) = value,
                  [JSONValue.integer(1), .integer(2)].contains(view["schemaVersion"] ?? .null), view["kind"] == .string("tableViewIndex"),
                  case .integer(let count) = view["rowCount"], count >= 0,
                  case .array(let columns) = view["columns"], (1...capacity.columns).contains(columns.count), count <= Int64(min(capacity.rows, capacity.cells / columns.count)),
                  case .string(let sourcePin) = view["sourceSHA256"], WorkflowArchiveCodec.isHash(sourcePin),
                  case .object(let index) = view["index"], inventory.contains(.object(index)),
                  index["kind"] == .string("tableViewIndex"), index["nodeID"] == .string(node), index["port"] == .string("table"),
                  index["sourceSHA256"] == .string(sourcePin), case .string(let path) = index["path"],
                  path.hasSuffix(".sqlite3"), try WorkflowArchiveCodec.pathParts(path).count == 1,
                  case .string(let pin) = index["sha256"], WorkflowArchiveCodec.isHash(pin),
                  case .integer(let bytes) = index["bytes"], (1...Int64(capacity.artifactBytes)).contains(bytes),
                  case .object(let nodeCoverage) = nodes[node], nodeCoverage["completeness"] == .string("complete"),
                  case .object(let rowCounts) = nodeCoverage["rowCounts"], rowCounts["table"] == .integer(count),
                  inventory.contains(where: { item in
                      guard case .object(let entry) = item else { return false }
                      return entry["kind"] == .string("table") && entry["nodeID"] == .string(node)
                          && entry["port"] == .string("table") && entry["sha256"] == .string(sourcePin)
                  }) else { throw failure() }
            result[node] = .init(indexURL: directory.appendingPathComponent(path), indexSHA256: pin, sourceSHA256: sourcePin,tokenVocabulary:tokenContext?.at(.init(node:node,port:"table")),workloadPolicy:workload)
        }
        return result
    }
    static func capturedTables(manifest: [String: JSONValue], directory: URL, expectedNodes: Set<String>) throws -> [WorkflowEndpoint: TableViewQuerySession.Artifact] {
        guard case .array(let inventory)=manifest["artifacts"],case .object(let coverage)=manifest["executionCoverage"],case .object(let nodes)=coverage["nodes"] else {throw failure()}
        let tokenContext=try TokenVocabularyResolver.make(manifest:manifest,directory:directory,endpoint:.init(node:"",port:""))
        let workload = try manifest["workloadPolicy"].map { try JSONDecoder().decode(ExecutionWorkloadPolicy.self, from: JSONEncoder().encode($0)) }
        let segments = try inventory.compactMap { value -> WorkflowArchiveFile? in
            guard case .object(let entry) = value, entry["kind"] == .string("tableSegment") else { return nil }
            guard manifest["tableStorageVersion"] == .integer(1), let name=entry["path"]?.stringValue,
                  PreparationArtifactPolicy.isSegmentName(name),case .integer(let count)=entry["bytes"],count>0,
                  count <= (workload?.limit(.retainedBytes) ?? 0),let pin=entry["sha256"]?.stringValue,WorkflowArchiveCodec.isHash(pin) else { throw failure() }
            return .init(path:name,bytes:Int(count),sha256:pin)
        }
        var result:[WorkflowEndpoint: TableViewQuerySession.Artifact]=[:]
        for case .object(let entry) in inventory where entry["kind"] == .string("table") {
            guard case .string(let node)=entry["nodeID"],expectedNodes.contains(node),case .string(let port)=entry["port"],
                  case .string(let path)=entry["path"],path.hasSuffix(".json"),try WorkflowArchiveCodec.pathParts(path).count == 1,
                  case .string(let pin)=entry["sha256"],WorkflowArchiveCodec.isHash(pin),case .integer(let bytes)=entry["bytes"],(1...50_000_000).contains(bytes),
                  case .object(let evidence)=nodes[node],case .object(let counts)=evidence["rowCounts"],case .integer(let count)=counts[port],count >= 0 else {throw failure()}
            let endpoint=WorkflowEndpoint(address:.init(domain:.data,localID:node),port:port)
            guard result[endpoint] == nil else {throw failure()}
            result[endpoint] = .init(indexURL:directory.appendingPathComponent(path),indexSHA256:pin,sourceSHA256:pin,inputKind:"table",tokenVocabulary:tokenContext?.at(.init(node:node,port:port)),retainedSegments:segments,workloadPolicy:workload)
        }
        return result
    }
    private static func failure() -> WorkflowArchiveError { .init("Table View lacks verified complete-table index evidence.") }
}
