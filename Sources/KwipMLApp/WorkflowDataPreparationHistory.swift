import Foundation
import Darwin
import Combine
import KwipMLCore

/// Immutable preparation History; reopening verifies captured bytes without
/// re-reading private source files or writing the external destination again.
@MainActor final class WorkflowDataPreparationHistory:ObservableObject {
    struct Entry:Identifiable {
        let id:String
        let completedAt:Date
        var timestamp:Date {completedAt}
        let document:WorkflowDocument
        let requestIdentity:String
        let artifactDirectory:URL
        let manifestSHA256:String
        let savedOutputs:[[String:JSONValue]]
        let artifacts:[WorkflowArchiveFile]
        var coverage: WorkflowPreparationCoverage.Receipt? = nil
        var tableViews: [String: TableViewQuerySession.Artifact] = [:]
    }
    private struct Receipt:Codable {
        let schemaVersion:Int;let projectID:String;let operationID:String;let completedAt:Date
        let document:WorkflowDocument;let requestSHA256:String;let manifestSHA256:String
    }
    @Published private(set) var entries:[Entry]=[]
    @Published private(set) var status=""
    private let root:URL;private let projectID:String
    private var generation=0
    init(root:URL,projectID:String) {
        // The managed worker captures physical paths; Foundation may display the
        // system /var alias. Keep the same physical root for exact request checks.
        if root.isFileURL,let pointer=realpath(root.path,nil) {
            self.root=URL(fileURLWithPath:String(cString:pointer),isDirectory:true);free(pointer)
        } else {self.root=root}
        self.projectID=projectID
    }
    func reload() {Task {await reloadAndWait()}}
    func reloadAndWait()async {
        generation+=1;let selected=generation,root=root,project=projectID
        let result=await Task.detached(priority:.utility) {Self.scan(root:root,projectID:project)}.value
        guard selected==generation else {return};entries=result.0;status=result.1.joined(separator:"\n")
    }
    func record(operationID:String,document:WorkflowDocument)async throws->Entry {
        let root=root,project=projectID
        let task=Task.detached(priority:.utility) {try Self.writeRecord(root:root,projectID:project,operationID:operationID,document:document)}
        let entry=try await withTaskCancellationHandler(operation:{try await task.value},onCancel:{task.cancel()})
        generation+=1;entries.removeAll {$0.id==entry.id};entries.insert(entry,at:0);status="";return entry
    }
    func verify(_ entry:Entry)async throws->Entry {
        let root=root,project=projectID
        let verified=try await Task.detached(priority:.userInitiated) {try Self.readRecord(root:root,projectID:project,operationID:entry.id)}.value
        guard verified.manifestSHA256==entry.manifestSHA256,verified.requestIdentity==entry.requestIdentity else {throw WorkflowArchiveError("Saved preparation evidence changed; reload History.")};return verified
    }
    func outputs(for entry:Entry)async throws->[WorkflowNodeAddress:[String:JSONValue]] {
        let verified=try await verify(entry)
        return try await Task.detached(priority:.userInitiated) {try Self.readOutputs(verified)}.value
    }
    func databaseRecords(for entry:Entry)async throws->[DatabaseOutputLedger.Record] {
        let root=root,project=projectID
        return try await Task.detached(priority:.utility) {
            let records=try DatabaseOutputLedger(root:root,projectID:project).load().filter {$0.preparationID==entry.id}
            if records.isEmpty {return []}
            let manifest=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:entry.artifactDirectory,parts:["manifest.json"],limit:2_000_000))
            guard case .array(let inventory)=manifest["artifacts"] else {throw WorkflowFileLeaseError.unsafePath}
            for r in records {
                guard inventory.contains(where:{if case .object(let file)=$0 {return file["kind"] == .string("table") && file["nodeID"] == .string(r.nodeID) && file["port"] == .string("table") && file["sha256"] == .string(r.sourceSHA256)};return false}) else {throw WorkflowArchiveError("Database receipt input differs from this captured table.")}
                guard let node=entry.document.nodes.first(where:{$0.address.localID==r.nodeID}),case .dataOperator("kwip.data.outputDelimited",let c)=node.payload,case .object(let db)=c[APITableSettings.isAPI(c) ? "api":"database"],try (APITableSettings.isAPI(c) ? APITableSettings.normalized(db):DatabaseOutputSettings.normalized(db))==(APITableSettings.isAPI(c) ? APITableSettings.normalized(r.configuration):r.configuration) else {throw WorkflowArchiveError("Database receipt does not match this captured preparation.")}}
            return records
        }.value
    }
    private nonisolated static func parts(projectID:String,operationID:String)throws->[String] {
        guard UUID(uuidString:projectID)?.uuidString==projectID,UUID(uuidString:operationID)?.uuidString==operationID else {throw WorkflowFileLeaseError.unsafePath}
        return ["workflow-operations",projectID,operationID]
    }
    private nonisolated static func verified(root:URL,projectID:String,operationID:String,document:WorkflowDocument,completedAt:Date)throws->(Entry,String) {
        try Task.checkCancellation();try WorkflowCodec.validate(document)
        guard document.projectID==projectID else {throw WorkflowFileLeaseError.unsafePath}
        let path=try parts(projectID:projectID,operationID:operationID),directory=path.reduce(root) {$0.appendingPathComponent($1)}
        let requestData=try DiagnosticFileIO.read(root:root,parts:path+["request.json"],limit:256_000)
        let request=try WorkflowToolRequest.decode(requestData);try request.validate(directory:directory,verifyInputs:false)
        guard request.toolMode == .dataPreparation,request.projectID==projectID,request.operationID==operationID else {throw WorkflowArchiveError("Saved preparation belongs to another captured operation.")}
        for file in request.inputFiles {try Task.checkCancellation();try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:WorkflowPreparationToolContract.fileLimit(path:file.path))}
        let output=directory.appendingPathComponent("artifacts")
        let raw=try DiagnosticFileIO.read(root:output,parts:["manifest.json"],limit:2_000_000)
        struct Manifest:Decodable {let artifacts:[WorkflowArchiveFile];let graphDigest:String;let outputs:[[String:JSONValue]]?}
        let manifest=try JSONDecoder().decode(Manifest.self,from:raw)
        let manifestFile=WorkflowArchiveFile(path:"manifest.json",bytes:raw.count,sha256:WorkflowArchiveCodec.hash(raw))
        try WorkflowCompletedPreparationInventory.verify(data:raw,files:manifest.artifacts+[manifestFile],prefix:"")
        guard Set(try FileManager.default.contentsOfDirectory(atPath:output.path))==Set(manifest.artifacts.map(\.path)+["manifest.json"]) else {throw WorkflowArchiveError("Preparation archive inventory changed.")}
        let artifactPolicy = try PreparationArtifactPolicy(manifest: raw)
        for file in manifest.artifacts {try Task.checkCancellation();try WorkflowPreviewToolContract.verifyFile(file:file,directory:output,limit:artifactPolicy.fileBytes(file.path))}
        func option(_ name: String) -> String? {
            guard let index = request.arguments.firstIndex(of: name), index + 1 < request.arguments.count else { return nil }
            return request.arguments[index + 1]
        }
        try WorkflowCompletedPreparationInventory.verifyWorkload(data: raw, expected: request.executionBudget.resources?.workloadPolicy)
        let retention=try JSONDecoder().decode([String:JSONValue].self,from:raw)["observationRetention"]
        guard retention == (option("--dataflow-capture-outputs")=="false" ? .string("requiredOnly"):nil) else {throw WorkflowArchiveError("Saved preparation retention differs from its request.")}
        if retention != nil {
            let expected=try option("--dataflow-required-nodes").map {try JSONDecoder().decode([String].self,from:Data($0.utf8))} ?? []
            guard try JSONDecoder().decode([String:JSONValue].self,from:raw)["requiredOutputNodes"] == .array(expected.sorted().map(JSONValue.string)) else {throw WorkflowArchiveError("Required preparation outputs differ from the captured selection.")}
        }
        let sampling = try option("--dataflow-sampling").map { try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8)) }
            ?? .object(["schemaVersion": .integer(1), "mode": .string("all")])
        let completeNodes = try option("--dataflow-complete-nodes").map { try JSONDecoder().decode([String].self, from: Data($0.utf8)) } ?? []
        let coverage = try WorkflowCompletedPreparationInventory.verifyCoverage(data: raw, files: manifest.artifacts + [manifestFile], prefix: "", directory: output, allowSample: true,
            expectedSampling: sampling, expectedCompleteNodes: Set(completeNodes))
        let captured=try DiagnosticFileIO.read(root:directory,parts:["input","graph.json"],limit:2_000_000)
        let graph=try DataFlowCodec.decodeGraph(captured)
        let tableViews = try TableViewArtifactReference.parse(manifest: JSONDecoder().decode([String: JSONValue].self, from: raw), directory: output,
            expectedNodes: Set(graph.nodes.filter { $0.operatorID == "kwip.data.view" }.map(\.id)))
        guard request.inputFiles.contains(.init(path:"input/graph.json",bytes:captured.count,sha256:WorkflowArchiveCodec.hash(captured))),try DataFlowCodec.decodeGraph(DiagnosticFileIO.read(root:output,parts:["dataflow.json"],limit:2_000_000))==graph else {throw WorkflowArchiveError("Preparation graph differs from its captured request.")}
        let expected=DataFlowDocument(nodes:document.nodes.compactMap {node in guard case .dataOperator(let id,let config)=node.payload else {return nil};return DataFlowNode(id:node.address.localID,operatorID:id,config:config)},edges:document.edges.filter {$0.from.address.domain == .data && $0.to.address.domain == .data}.map {.init(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))})
        guard expected==graph else {throw WorkflowArchiveError("Preparation History document does not identify the actual executed data graph.")}
        return (.init(id:operationID,completedAt:completedAt,document:document,requestIdentity:request.requestIdentity,artifactDirectory:output,manifestSHA256:manifestFile.sha256,savedOutputs:manifest.outputs ?? [],artifacts:manifest.artifacts,coverage:coverage,tableViews:tableViews),WorkflowArchiveCodec.hash(requestData))
    }
    private nonisolated static func writeRecord(root:URL,projectID:String,operationID:String,document:WorkflowDocument)throws->Entry {
        let path=try parts(projectID:projectID,operationID:operationID)
        if let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:path+["preparation-history.json"],limit:4_000_000) {
            guard try JSONDecoder().decode(Receipt.self,from:bytes).document==document else {throw WorkflowArchiveError("Another document already owns this preparation receipt.")};return try readRecord(root:root,projectID:projectID,operationID:operationID)
        }
        let now=Date(),(entry,requestSHA)=try verified(root:root,projectID:projectID,operationID:operationID,document:document,completedAt:now)
        let receipt=Receipt(schemaVersion:1,projectID:projectID,operationID:operationID,completedAt:now,document:document,requestSHA256:requestSHA,manifestSHA256:entry.manifestSHA256)
        try Task.checkCancellation();try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(receipt),root:root,parts:path+["preparation-history.json"],limit:4_000_000);return entry
    }
    /// Receipt metadata only; this does not establish that retained artifacts still verify.
    nonisolated static func retainedMetadata(root:URL,projectID:String,operationID:String,budget:WorkflowRetainedHistoryReader.MetadataBudget)throws->WorkflowRetainedHistoryReader.Metadata? {
        let path=try parts(projectID:projectID,operationID:operationID)
        guard let bytes=try budget.read(root:root,parts:path+["preparation-history.json"],limit:4000000) else {return nil}
        let receipt=try JSONDecoder().decode(Receipt.self,from:bytes)
        guard receipt.schemaVersion==1,receipt.projectID==projectID,receipt.operationID==operationID,
              receipt.document.projectID==projectID,receipt.completedAt.timeIntervalSince1970.isFinite,
              WorkflowArchiveCodec.isHash(receipt.manifestSHA256) else {throw WorkflowArchiveError("History receipt identity is invalid.")}
        return .init(family:.preparation,operationID:operationID,completedAt:receipt.completedAt,
                     capturedRevision:receipt.document.revision,manifestSHA256:receipt.manifestSHA256,components:MCPComponentAccess.identifiers(receipt.document))
    }
    nonisolated static func retainedEntry(root:URL,projectID:String,operationID:String)throws->Entry {
        try readRecord(root:root,projectID:projectID,operationID:operationID)
    }
    private nonisolated static func readRecord(root:URL,projectID:String,operationID:String)throws->Entry {
        let path=try parts(projectID:projectID,operationID:operationID)
        let receipt=try JSONDecoder().decode(Receipt.self,from:DiagnosticFileIO.read(root:root,parts:path+["preparation-history.json"],limit:4_000_000))
        guard receipt.schemaVersion==1,receipt.projectID==projectID,receipt.operationID==operationID,receipt.completedAt.timeIntervalSince1970.isFinite else {throw WorkflowArchiveError("Preparation receipt identity is invalid.")}
        let (entry,requestSHA)=try verified(root:root,projectID:projectID,operationID:operationID,document:receipt.document,completedAt:receipt.completedAt)
        guard requestSHA==receipt.requestSHA256,entry.manifestSHA256==receipt.manifestSHA256 else {throw WorkflowArchiveError("Preparation History evidence changed.")};return entry
    }
    private nonisolated static func scan(root:URL,projectID:String)->([Entry],[String]) {
        do {
            guard UUID(uuidString:projectID)?.uuidString==projectID else {throw WorkflowFileLeaseError.unsafePath}
            let folder=root.appendingPathComponent("workflow-operations/\(projectID)")
            guard FileManager.default.fileExists(atPath:folder.path) else {return ([],[])}
            let metadata=try folder.resourceValues(forKeys:[.isSymbolicLinkKey,.isDirectoryKey])
            guard metadata.isSymbolicLink != true,metadata.isDirectory==true else {throw WorkflowFileLeaseError.unsafePath}
            let children=try FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:nil,options:[.skipsHiddenFiles])
            guard children.count<=5000 else {throw WorkflowArchiveError("Preparation History exceeds its scan bound.")}
            var result:[Entry]=[],issues:[String]=[]
            for child in children where UUID(uuidString:child.lastPathComponent)?.uuidString==child.lastPathComponent {
                do {
                    let path=try parts(projectID:projectID,operationID:child.lastPathComponent)
                    if try DiagnosticFileIO.readIfPresent(root:root,parts:path+["preparation-history.json"],limit:4_000_000) != nil {result.append(try readRecord(root:root,projectID:projectID,operationID:child.lastPathComponent))}
                } catch {issues.append(error.localizedDescription)}
            }
            return (result.sorted {$0.completedAt>$1.completedAt},issues)
        } catch {return ([],[error.localizedDescription])}
    }
    nonisolated static func readOutputs(_ entry:Entry)throws->[WorkflowNodeAddress:[String:JSONValue]] {
        var outputs:[WorkflowNodeAddress:[String:JSONValue]]=[:],total=0
        let manifestData=try DiagnosticFileIO.read(root:entry.artifactDirectory,parts:["manifest.json"],limit:2_000_000)
        guard WorkflowArchiveCodec.hash(manifestData) == entry.manifestSHA256 else { throw WorkflowArchiveError("Saved preparation manifest changed while loading outputs.") }
        struct Indexed:Decodable {struct Item:Decodable {let path:String;let nodeID:String?;let port:String?;let kind:String};let graphDigest:String;let artifacts:[Item]}
        let manifest=try JSONDecoder().decode(Indexed.self,from:manifestData)
        let indexed=manifest.artifacts
        if try JSONDecoder().decode([String:JSONValue].self,from:manifestData)["observationRetention"] == .string("requiredOnly") {return [:]}
        if indexed.contains(where: { $0.path == "previews.json" }) {
            guard indexed.filter({ $0.path == "previews.json" }).count == 1,
                  indexed.first(where: { $0.path == "previews.json" })?.kind == "previews" else {
                throw WorkflowArchiveError("Saved preparation preview inventory is invalid.")
            }
            func pinned(_ name:String,limit:Int)throws->Data {
                try Task.checkCancellation()
                guard let file=entry.artifacts.first(where: { $0.path == name }),file.bytes <= limit,
                      indexed.contains(where: { $0.path == name }) else { throw WorkflowArchiveError("Saved preparation lacks bounded preview metadata.") }
                let bytes=try DiagnosticFileIO.read(root:entry.artifactDirectory,parts:[name],limit:limit)
                guard bytes.count == file.bytes,WorkflowArchiveCodec.hash(bytes) == file.sha256 else {
                    throw WorkflowArchiveError("Saved preparation preview metadata changed: " + name)
                }
                return bytes
            }
            let graphBytes=try pinned("dataflow.json",limit:2_000_000)
            // The worker writes dataflow.json with the same canonical encoder used
            // for graphDigest. Decode those same pinned bytes for node authority.
            guard WorkflowArchiveCodec.hash(graphBytes) == manifest.graphDigest else { throw WorkflowArchiveError("Saved preparation preview graph identity differs.") }
            let graph=try DataFlowCodec.decodeGraph(graphBytes)
            let registry=try JSONDecoder().decode(DataFlowOperatorRegistry.self,from:pinned("dataflow-operators.json",limit:2_000_000))
            guard registry.schemaVersion == 1,Set(registry.operators.map(\.id)).count == registry.operators.count else {
                throw WorkflowArchiveError("Saved preparation preview registry is invalid.")
            }
            let raw=try JSONDecoder().decode(JSONValue.self,from:pinned("previews.json",limit:32_000_000))
            guard case .object(let sidecar)=raw,Set(sidecar.keys) == ["schemaVersion","graphDigest","outputs"],
                  sidecar["schemaVersion"] == .integer(1),sidecar["graphDigest"] == .string(manifest.graphDigest),
                  case .object(let nodes)=sidecar["outputs"],Set(nodes.keys) == Set(graph.nodes.map(\.id)) else {
                throw WorkflowArchiveError("Saved preparation previews differ from the captured graph.")
            }
            for node in graph.nodes {
                try Task.checkCancellation()
                guard let definition=registry.operatorDefinition(id:node.operatorID)?.configuredForExecution(node.config,connectedInputs:Set(graph.edges.filter{$0.to.node==node.id}.map{$0.to.port})),
                      case .object(let ports)=nodes[node.id],!definition.outputs.isEmpty,
                      Set(definition.outputs.map(\.id)).count == definition.outputs.count,
                      Set(ports.keys) == Set(definition.outputs.map(\.id)) else {
                    throw WorkflowArchiveError("Saved preparation preview ports differ from the captured component.")
                }
                for port in definition.outputs {
                    guard let value=ports[port.id] else { throw WorkflowArchiveError("Saved preparation preview port is absent.") }
                    try WorkflowPreparationPreviewValidation.validate(value,kind:port.kind,operatorID:node.operatorID)
                }
                outputs[.init(domain:.data,localID:node.id)]=ports
            }
            return outputs
        }
        for file in entry.artifacts where file.path.hasPrefix("node-") {
            try Task.checkCancellation()
            let bytes=try DiagnosticFileIO.read(root:entry.artifactDirectory,parts:[file.path],limit:50_000_000)
            guard let index=indexed.first(where:{$0.path==file.path}),let node=index.nodeID,let port=index.port else {continue}
            let object=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            let value:JSONValue
            if index.kind=="table",case .array(let columns)=object["columns"],case .array(let rows)=object["rows"],case .object(let lineage)=object["lineage"] {
                let previewRows=rows.prefix(8).map {row->JSONValue in guard case .array(let cells)=row else {return row};return .array(cells.map {if case .string(let text)=$0 {return .string(String(String.UnicodeScalarView(text.unicodeScalars.prefix(200))))};return $0})}
                var publicLineage=lineage;publicLineage["observationIDs"]=nil;publicLineage["diagnostics"]=nil
                let diagnostics: [JSONValue];if case .array(let all)=lineage["diagnostics"] {diagnostics=all}else {diagnostics=[]}
                value = .object(["kind":.string("table"),"preview":.object(["columns":.array(columns),"rows":.array(previewRows),"rowCount":.integer(Int64(rows.count)),"partition":lineage["role"] ?? .string("unassigned"),"dataVersion":lineage["dataVersion"] ?? .null,"diagnostics":.array(Array(diagnostics.suffix(16))),"diagnosticCount":.integer(Int64(diagnostics.count))]),"lineage":.object(publicLineage)])
            } else if index.kind=="state",case .object(let fit)=object["fitEvidence"] {
                var lineage=fit;lineage["observationIDs"]=nil;lineage["diagnostics"]=nil
                let keys:Set<String>=["operator","columns","means","scales","fillValues","observedCounts","emptyColumns","vocabularies","contentDigest","fitDataVersion"]
                value = .object(["kind":.string("state"),"summary":.object(object.filter {keys.contains($0.key)}),"lineage":.object(lineage)])
            } else {throw WorkflowArchiveError("Preparation table/state archive is malformed.")}
            total += try WorkflowArchiveCodec.encode(value).count;guard total<=32_000_000 else {throw WorkflowArchiveError("Saved preparation previews exceed display budget.")}
            outputs[.init(domain:.data,localID:node),default:[:]][port]=value
        }
        return outputs
    }
}
