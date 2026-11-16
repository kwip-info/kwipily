import Foundation
import Combine
import KwipMLCore

/// A derived History index over immutable receipts beneath retained operations.
/// Artifact hashing and receipt IO stay off the main actor.
@MainActor final class BackendFitHistoryStore:ObservableObject {
    struct Entry:Identifiable {
        let id:String
        let backendID:String
        let completedAt:Date
        var timestamp:Date {completedAt}
        let evaluation:[String:JSONValue]
        let artifactDirectory:URL
        let manifestSHA256:String
        let workerRequest:Data
        let document:WorkflowDocument
        let artifacts:[WorkflowArchiveFile]
    }
    @Published private(set) var entries:[Entry]=[]
    @Published private(set) var status=""
    private let root:URL
    private let projectID:String
    private var generation=0
    init(root:URL,projectID:String){self.root=root;self.projectID=projectID}
    func reload(){Task {await reloadAndWait()}}
    func reloadAndWait()async {
        generation += 1;let selected=generation
        let root=root,project=projectID
        status="Verifying backend history…"
        let result=await Task.detached(priority:.utility) {Self.scan(root:root,projectID:project)}.value
        guard selected==generation else {return}
        entries=result.0;status=result.1.joined(separator:"\n")
    }
    func record(operationID:String,workerRequest:Data,document:WorkflowDocument)async throws {
        try Task.checkCancellation()
        let root=root,project=projectID,acceptance=ReceiptAcceptance()
        let worker=Task.detached(priority:.utility) {
            try Task.checkCancellation()
            return try Self.writeRecord(root:root,projectID:project,operationID:operationID,workerRequest:workerRequest,document:document,acceptance:acceptance)
        }
        let entry=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{acceptance.cancel();worker.cancel()})
        // Receipt acceptance is the commit boundary. Cancellation after it cannot
        // turn a durable verified completion into a reported non-result.
        generation += 1
        entries.removeAll{$0.id==entry.id};entries.append(entry);entries.sort{$0.completedAt>$1.completedAt};status=""
    }
    /// Recheck immediately before an export or opening saved output.
    func verify(_ entry:Entry)async throws->Entry {
        let root=root,project=projectID,id=entry.id
        let verified=try await Task.detached(priority:.userInitiated) {try Self.readRecord(root:root,projectID:project,operationID:id)}.value
        guard verified.manifestSHA256==entry.manifestSHA256,verified.workerRequest==entry.workerRequest else {throw WorkflowArchiveError("Selected backend History evidence changed; reload History before exporting.")}
        return verified
    }
    /// Serializes cancellation against the beginning of immutable receipt commit.
    /// Locks cover only state transitions; hashing and disk IO never hold the lock.
    private final class ReceiptAcceptance:@unchecked Sendable {
        private let lock=NSLock()
        private var cancelled=false
        private var committing=false
        func cancel(){lock.lock();defer {lock.unlock()};if !committing {cancelled=true}}
        func beginCommit()throws {lock.lock();defer {lock.unlock()};guard !cancelled else {throw CancellationError()};committing=true}
    }
    private struct Receipt:Codable {
        let schemaVersion:Int
        let projectID:String
        let operationID:String
        let completedAt:Date
        let workerRequest:Data
        let document:WorkflowDocument
        let outerRequestSHA256:String
        let manifestSHA256:String
    }
    private nonisolated static func parts(projectID:String,operationID:String)throws->[String] {
        guard UUID(uuidString:projectID)?.uuidString==projectID,UUID(uuidString:operationID)?.uuidString==operationID else {throw WorkflowArchiveError("Backend history requires canonical project and operation identities.")}
        return ["workflow-operations",projectID,operationID]
    }
    private nonisolated static func verified(root:URL,projectID:String,operationID:String,workerRequest:Data,document:WorkflowDocument,completedAt:Date)throws->(Entry,String) {
        let path=try parts(projectID:projectID,operationID:operationID)
        guard workerRequest.count<=100_000,document.projectID==projectID,completedAt.timeIntervalSince1970.isFinite else {throw WorkflowArchiveError("Backend history receipt identity is invalid.")}
        try WorkflowCodec.validate(document)
        let directory=path.reduce(root){$0.appendingPathComponent($1)}
        let bytes=try DiagnosticFileIO.read(root:root,parts:path+["request.json"],limit:256_000)
        let outer=try WorkflowToolRequest.decode(bytes)
        try outer.validate(directory:directory,verifyInputs:false)
        guard outer.toolMode == .backendFit,outer.projectID==projectID,outer.operationID==operationID,
              outer.inputFiles.contains(where:{$0.path=="input/backend-request.json" && $0.bytes==workerRequest.count && $0.sha256==WorkflowArchiveCodec.hash(workerRequest)}) else {throw WorkflowArchiveError("Backend history request differs from the retained operation.")}
        let output=directory.appendingPathComponent("artifacts")
        let backend:String,evaluation:[String:JSONValue],manifestSHA256:String,files:[WorkflowArchiveFile]
        guard let object=try JSONSerialization.jsonObject(with:workerRequest) as? [String:Any],let id=object["backendID"] as? String else {throw WorkflowArchiveError("Backend history request has no backend.")}
        backend=id
        if id=="tensorflow" {
            let result=try TensorFlowWorkflowResult.verify(directory:output,request:TensorFlowPreparedWorkerRequest.decode(workerRequest))
            evaluation=try JSONDecoder().decode([String:JSONValue].self,from:JSONEncoder().encode(result.evaluation));manifestSHA256=result.manifestSHA256;files=result.artifacts
        } else if id=="sklearn" {
            let result=try SklearnWorkflowResult.verify(directory:output,request:SklearnPreparedWorkerRequest.decode(workerRequest))
            let manifest=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:output,parts:["manifest.json"],limit:2_000_000))
            guard case .object(let metrics)=manifest["evaluation"] else {throw WorkflowArchiveError("Missing verified evaluation.")}
            var presentation=metrics
            if let configuration=object["configuration"] as? [String:Any],configuration["task"] as? String == "documentClassification" {presentation["operation"] = .string(configuration["operation"] as? String ?? "fit")}
            if let config=object["configuration"] as? [String:Any],config["task"] as? String == "prebuilt" {presentation["task"] = .string("prebuilt");presentation["operation"] = .string(config["operation"] as? String ?? "fit")}
            evaluation=presentation;manifestSHA256=result.manifestSHA256;files=result.artifacts
        } else {throw WorkflowArchiveError("Unsupported backend History entry.")}
        return (.init(id:operationID,backendID:backend,completedAt:completedAt,evaluation:evaluation,artifactDirectory:output,manifestSHA256:manifestSHA256,workerRequest:workerRequest,document:document,artifacts:files),WorkflowArchiveCodec.hash(bytes))
    }
    private nonisolated static func writeRecord(root:URL,projectID:String,operationID:String,workerRequest:Data,document:WorkflowDocument,acceptance:ReceiptAcceptance)throws->Entry {
        let path=try parts(projectID:projectID,operationID:operationID)
        if let existing=try DiagnosticFileIO.readIfPresent(root:root,parts:path+["backend-history.json"],limit:8_000_000) {
            let receipt=try JSONDecoder().decode(Receipt.self,from:existing)
            guard receipt.workerRequest==workerRequest,receipt.document==document else {throw WorkflowArchiveError("A different backend History receipt already owns this operation.")}
            let entry=try readRecord(root:root,projectID:projectID,operationID:operationID)
            try acceptance.beginCommit()
            return entry
        }
        let now=Date();let (entry,outerSHA)=try verified(root:root,projectID:projectID,operationID:operationID,workerRequest:workerRequest,document:document,completedAt:now)
        let receipt=Receipt(schemaVersion:1,projectID:projectID,operationID:operationID,completedAt:now,workerRequest:workerRequest,document:document,outerRequestSHA256:outerSHA,manifestSHA256:entry.manifestSHA256)
        let bytes=try WorkflowArchiveCodec.encode(receipt)
        try acceptance.beginCommit()
        try DiagnosticFileIO.writeImmutable(bytes,root:root,parts:path+["backend-history.json"],limit:8_000_000)
        return entry
    }
    /// Receipt metadata only; this does not establish that retained artifacts still verify.
    nonisolated static func retainedMetadata(root:URL,projectID:String,operationID:String,budget:WorkflowRetainedHistoryReader.MetadataBudget)throws->WorkflowRetainedHistoryReader.Metadata? {
        let path=try parts(projectID:projectID,operationID:operationID)
        guard let bytes=try budget.read(root:root,parts:path+["backend-history.json"],limit:8000000) else {return nil}
        let receipt=try JSONDecoder().decode(Receipt.self,from:bytes)
        guard receipt.schemaVersion==1,receipt.projectID==projectID,receipt.operationID==operationID,
              receipt.document.projectID==projectID,receipt.completedAt.timeIntervalSince1970.isFinite,
              WorkflowArchiveCodec.isHash(receipt.manifestSHA256) else {throw WorkflowArchiveError("History receipt identity is invalid.")}
        return .init(family:.backend,operationID:operationID,completedAt:receipt.completedAt,
                     capturedRevision:receipt.document.revision,manifestSHA256:receipt.manifestSHA256,components:MCPComponentAccess.identifiers(receipt.document))
    }
    nonisolated static func retainedEntry(root:URL,projectID:String,operationID:String)throws->Entry {
        try readRecord(root:root,projectID:projectID,operationID:operationID)
    }
    private nonisolated static func readRecord(root:URL,projectID:String,operationID:String)throws->Entry {
        let path=try parts(projectID:projectID,operationID:operationID)
        let bytes=try DiagnosticFileIO.read(root:root,parts:path+["backend-history.json"],limit:8_000_000)
        let receipt=try JSONDecoder().decode(Receipt.self,from:bytes)
        guard receipt.schemaVersion==1,receipt.projectID==projectID,receipt.operationID==operationID else {throw WorkflowArchiveError("Backend History receipt belongs to another operation.")}
        let (entry,outerSHA)=try verified(root:root,projectID:projectID,operationID:operationID,workerRequest:receipt.workerRequest,document:receipt.document,completedAt:receipt.completedAt)
        guard outerSHA==receipt.outerRequestSHA256,entry.manifestSHA256==receipt.manifestSHA256 else {throw WorkflowArchiveError("Backend History evidence changed after completion.")}
        return entry
    }
    private nonisolated static func scan(root:URL,projectID:String)->([Entry],[String]) {
        do {
            guard UUID(uuidString:projectID)?.uuidString==projectID else {throw WorkflowArchiveError("Invalid project identity for backend History.")}
            let project=root.appendingPathComponent("workflow-operations/\(projectID)")
            if !FileManager.default.fileExists(atPath:project.path) {return ([],[])}
            let values=try project.resourceValues(forKeys:[.isSymbolicLinkKey,.isDirectoryKey])
            guard values.isSymbolicLink != true,values.isDirectory==true else {throw WorkflowArchiveError("Backend History folder is unsafe.")}
            let children=try FileManager.default.contentsOfDirectory(at:project,includingPropertiesForKeys:[],options:[.skipsHiddenFiles])
            guard children.count<=5000 else {throw WorkflowArchiveError("Backend History exceeds its 5000-operation scan bound.")}
            var entries:[Entry]=[],failures:[String]=[]
            for child in children where UUID(uuidString:child.lastPathComponent)?.uuidString==child.lastPathComponent {
                do {
                    let path=try parts(projectID:projectID,operationID:child.lastPathComponent)
                    guard try DiagnosticFileIO.readIfPresent(root:root,parts:path+["backend-history.json"],limit:8_000_000) != nil else {continue}
                    entries.append(try readRecord(root:root,projectID:projectID,operationID:child.lastPathComponent))
                } catch {failures.append("\(child.lastPathComponent): \(error.localizedDescription)")}
            }
            return (entries.sorted{$0.completedAt>$1.completedAt},failures)
        } catch {return ([],[error.localizedDescription])}
    }
}
