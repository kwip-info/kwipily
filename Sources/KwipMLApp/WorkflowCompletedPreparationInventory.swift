import Foundation
import KwipMLCore


/// Common completed-preparation envelope check. Reuse may intentionally target a different prior graph.
enum WorkflowCompletedPreparationInventory {
    static func verifyModelInputs(data:Data,expected:[String]) throws {
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:data)
        guard manifest["modelInputNodes"] == (expected.isEmpty ? nil:.array(expected.sorted().map(JSONValue.string))),
              case .array(let artifacts)=manifest["artifacts"] else {throw WorkflowArchiveError("Model input representation differs from the captured consumer contract.")}
        for case .object(let file) in artifacts where file["kind"] == .string("table") && expected.contains(file["nodeID"]?.stringValue ?? "") {
            guard file["tableStorageVersion"] == nil else {throw WorkflowArchiveError("A canonical model input was replaced with segmented storage.")}
        }
    }
    static func verifyWorkload(data: Data, expected: ExecutionWorkloadPolicy?) throws {
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: data)
        let captured = try expected.map { try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode($0)) }
        guard manifest["workloadPolicy"] == captured else {
            throw WorkflowArchiveError("Preparation workload policy differs from its accepted operation.")
        }
    }

    static func verify(data: Data, files: [WorkflowArchiveFile], prefix: String, expectedGraphDigest: String? = nil, workload: ExecutionWorkloadPolicy? = nil) throws {
        struct Manifest: Decodable { let schemaVersion: Int; let kind: String; let status: String; let graphDigest: String; let artifacts: [WorkflowArchiveFile] }
        guard data.count <= 2_000_000 else { throw WorkflowFileLeaseError.unsafePath }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        let limits = try PreparationArtifactPolicy(manifest: data)
        guard manifest.schemaVersion == 1, manifest.kind == "dataflowExecution", manifest.status == "completed",
              WorkflowArchiveCodec.isHash(manifest.graphDigest), expectedGraphDigest == nil || manifest.graphDigest == expectedGraphDigest,
              (1...limits.inventoryCount).contains(manifest.artifacts.count), Set(manifest.artifacts.map(\.path)).count == manifest.artifacts.count,
              files.contains(.init(path: prefix + "manifest.json", bytes: data.count, sha256: WorkflowArchiveCodec.hash(data))) else { throw WorkflowFileLeaseError.unsafePath }
        let expected = files.filter { $0.path.hasPrefix(prefix) && $0.path != prefix + "manifest.json" }
        guard expected.count == manifest.artifacts.count else { throw WorkflowFileLeaseError.unsafePath }
        var total = 0
        for file in manifest.artifacts {
            guard !file.path.isEmpty, !file.path.contains("/"), !file.path.contains("\0"), file.path != ".", file.path != "..", file.path != "manifest.json",
                  (0...limits.fileBytes(file.path)).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256),
                  expected.contains(.init(path: prefix + file.path, bytes: file.bytes, sha256: file.sha256)) else { throw WorkflowFileLeaseError.unsafePath }
            guard file.bytes <= min(limits.totalBytes, workload.map { Int($0.limit(.retainedBytes)) } ?? limits.totalBytes) - total else { throw WorkflowFileLeaseError.unsafePath }
            total += file.bytes
        }
    }
    /// Call after envelope validation. Only bounded pinned metadata is decoded;
    /// full table/state bytes retain the existing streamed inventory checks.
    static func verifyCoverage(data: Data, files: [WorkflowArchiveFile], prefix: String, directory: URL,
                               selectedNodes: Set<String>? = nil, allowSample: Bool = false,
                               expectedSampling: JSONValue? = nil, expectedCompleteNodes: Set<String>? = nil) throws -> WorkflowPreparationCoverage.Receipt? {
        let envelope = try JSONDecoder().decode([String: JSONValue].self, from: data)
        try FileImageArtifactInventory.verify(manifest:data,files:files,prefix:prefix,directory:directory)
        let planPath = prefix + "preparation-plan.json"
        let hasPlan = files.contains { $0.path == planPath }
        guard envelope["executionCoverage"] != nil || hasPlan || expectedSampling != nil else { return nil }
        func metadata(_ name: String) throws -> Data {
            let path = prefix + name
            guard let file = files.first(where: { $0.path == path }), (0...2_000_000).contains(file.bytes) else {
                throw WorkflowArchiveError("Preparation coverage lacks its pinned graph or plan.")
            }
            let bytes = try DiagnosticFileIO.read(root: directory, parts: WorkflowArchiveCodec.pathParts(path), limit: 2_000_000)
            guard bytes.count == file.bytes, WorkflowArchiveCodec.hash(bytes) == file.sha256 else {
                throw WorkflowArchiveError("Preparation coverage metadata changed.")
            }
            return bytes
        }
        var observedTablePorts:[String:Set<String>]?
        if let retention=envelope["observationRetention"] {
            guard retention == .string("requiredOnly"),case .array(let artifacts)=envelope["artifacts"],
                  !artifacts.contains(where: {if case .object(let item)=$0 {return item["kind"] == .string("previews") || item["path"] == .string("previews.json")};return false}) else {throw WorkflowArchiveError("Invalid preparation observation retention.")}
            let graph=try DataFlowCodec.decodeGraph(metadata("dataflow.json"))
            let registry=try JSONDecoder().decode(DataFlowOperatorRegistry.self,from:metadata("dataflow-operators.json"))
            guard case .array(let selection)=envelope["requiredOutputNodes"],selection.count<=128,selection.allSatisfy({$0.stringValue != nil}) else {throw WorkflowArchiveError("Invalid required preparation node selection.")}
            let required=Set(selection.compactMap(\.stringValue))
            guard required.count==selection.count,required.isSubset(of:Set(graph.nodes.map(\.id))) else {throw WorkflowArchiveError("Unknown required preparation node.")}
            let intermediates=Set(graph.edges.map {$0.from.node})
            let cacheEnabled:Bool;if case .object(let cache)=envelope["cache"] {cacheEnabled=cache["policy"] != .string("off")}else {cacheEnabled=false}
            var expected=Set<DataFlowEndpoint>(),actual=Set<DataFlowEndpoint>(),observed:[String:Set<String>]=[:]
            for node in graph.nodes {
                guard let definition=registry.operatorDefinition(id:node.operatorID)?.configuredForExecution(node.config,connectedInputs:Set(graph.edges.filter {$0.to.node==node.id}.map {$0.to.port})) else {throw WorkflowArchiveError("Preparation retention has an unknown component.")}
                observed[node.id]=Set(definition.outputs.filter {$0.kind == "table"}.map(\.id))
                for port in definition.outputs where port.kind == "state" || cacheEnabled || required.contains(node.id) || !intermediates.contains(node.id) || node.operatorID == "kwip.data.view" {expected.insert(.init(node:node.id,port:port.id))}
            }
            for case .object(let item) in artifacts where item["kind"] == .string("table") || item["kind"] == .string("state") {
                guard let node=item["nodeID"]?.stringValue,let port=item["port"]?.stringValue,actual.insert(.init(node:node,port:port)).inserted else {throw WorkflowArchiveError("Invalid retained preparation endpoint.")}
            }
            observedTablePorts=observed
            guard actual==expected else {throw WorkflowArchiveError("Preparation omitted required outputs or retained optional intermediate copies.")}
        }
        return try WorkflowPreparationCoverage.verify(manifest: data, graph: metadata("dataflow.json"),
            plan: hasPlan ? metadata("preparation-plan.json") : nil, selectedNodes: selectedNodes, allowSample: allowSample,
            expectedSampling: expectedSampling, expectedCompleteNodes: expectedCompleteNodes,observedTablePorts:observedTablePorts)
    }

}
