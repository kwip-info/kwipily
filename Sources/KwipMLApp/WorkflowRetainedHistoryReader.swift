import Foundation
import Darwin
import KwipMLCore

/// Read-only retained evidence. Does not own a document, selection, runtime or source grant.
/// Family readers remain the sole receipt decoder and artifact verification authority.
enum WorkflowRetainedHistoryReader {
    enum Family:String,Codable,CaseIterable {case preparation,backend}
    struct Metadata:Encodable,Equatable {
        let family:Family
        let operationID:String
        let completedAt:Date
        let capturedRevision:UInt64
        let manifestSHA256:String
        var components:Set<String>=[]
        enum CodingKeys:String,CodingKey {case family,operationID,completedAt,capturedRevision,manifestSHA256,verification}
        let verification:String = "recordedReceiptOnly"
    }
    struct Issue:Equatable {let operationID:String;let code:String}
    struct Cursor {fileprivate let snapshotID:UUID;fileprivate let offset:Int}
    struct Listing {let records:[Metadata];let issues:[Issue];let nextCursor:Cursor?;let inspectedOperations:Int}
    struct Snapshot {
        let projectID:String
        let family:Family
        let receiptBytesRead:Int
        private let id=UUID()
        fileprivate let rows:[Row]
        fileprivate enum Row {case metadata(Metadata),issue(Issue),absent}
        func page(cursor:Cursor?=nil,limit:Int=50)throws->Listing {
            guard (1...100).contains(limit),cursor == nil || cursor?.snapshotID==id else {throw WorkflowArchiveError("History cursor does not belong to this snapshot.")}
            let offset=cursor?.offset ?? 0
            guard (0...rows.count).contains(offset) else {throw WorkflowArchiveError("History cursor is unavailable.")}
            let selected=rows.dropFirst(offset).prefix(limit)
            let records=selected.compactMap {if case .metadata(let value)=$0{return value};return nil}
            let issues=selected.compactMap {if case .issue(let value)=$0{return value};return nil}
            let next=offset+selected.count
            return .init(records:records,issues:issues,nextCursor:next<rows.count ? .init(snapshotID:id,offset:next):nil,inspectedOperations:selected.count)
        }
    }
    enum ReadFailure:Error {case metadataBudgetExceeded}
    /// Single-threaded per-snapshot admission; malformed decoded receipts still consume bytes.
    final class MetadataBudget {
        private(set) var remaining:Int
        private(set) var consumed=0
        init(_ allowance:Int){remaining=allowance}
        func read(root:URL,parts:[String],limit:Int)throws->Data? {
            guard remaining>=limit else {throw ReadFailure.metadataBudgetExceeded}
            let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:limit)
            if let bytes {remaining-=bytes.count;consumed+=bytes.count}
            return bytes
        }
    }
    struct Provenance:Encodable {
        let receipt:Metadata
        let artifactCount:Int
        let verification:String = "verifiedRetainedArtifacts"
        let scope:String = "Captured operation only; not current graph, source files or exported destinations."
    }
    enum Verified {
        case preparation(WorkflowDataPreparationHistory.Entry)
        case backend(BackendFitHistoryStore.Entry)
        var provenance:Provenance {
            let count:Int
            switch self {case .preparation(let e):count=e.artifacts.count;case .backend(let e):count=e.artifacts.count}
            return .init(receipt:metadata,artifactCount:count)
        }
        var metadata:Metadata {
            switch self {
            case .preparation(let e):return .init(family:.preparation,operationID:e.id,completedAt:e.completedAt,capturedRevision:e.document.revision,manifestSHA256:e.manifestSHA256,components:MCPComponentAccess.identifiers(e.document))
            case .backend(let e):return .init(family:.backend,operationID:e.id,completedAt:e.completedAt,capturedRevision:e.document.revision,manifestSHA256:e.manifestSHA256,components:MCPComponentAccess.identifiers(e.document))
            }
        }
    }
    private static func physicalRoot(_ root:URL)->URL {
        // Match the family store's physical root exactly; Foundation may retain /var aliases.
        guard root.isFileURL,let pointer=realpath(root.path,nil) else {return root}
        defer {free(pointer)}
        return URL(fileURLWithPath:String(cString:pointer),isDirectory:true)
    }
    static func list(root:URL,projectID:String,family:Family,offset:Int=0,limit:Int=50)async throws->Listing {
        // Compatibility for first-page callers only. Continuation requires the same snapshot.
        guard offset==0,(1...100).contains(limit) else {throw WorkflowArchiveError("Open a History snapshot to continue listing.")}
        return try await open(root:root,projectID:projectID,family:family).page(limit:limit)
    }
    static func open(root:URL,projectID:String,family:Family,metadataAllowance:Int=64_000_000)async throws->Snapshot {
        let root=physicalRoot(root)
        return try await WorkflowHistoryVerificationWork.runCancellable {check in
            try check()
            guard UUID(uuidString:projectID)?.uuidString==projectID,(8_000_000...64_000_000).contains(metadataAllowance) else {throw WorkflowArchiveError("Invalid History listing scope.")}
            let scan=try WorkflowHistoryReader.scanDirectories(root:root,projectID:projectID,limit:5000,namespace:"workflow-operations")
            guard !scan.truncated,scan.issues.isEmpty else {throw WorkflowArchiveError("History listing cannot safely inspect all operation identities.")}
            let ids=scan.directories.map(\.uuidString).sorted(),budget=MetadataBudget(metadataAllowance)
            var rows:[Snapshot.Row]=[]
            for id in ids {
                try check()
                do {
                    let metadata:Metadata?
                    switch family {
                    case .preparation:metadata=try WorkflowDataPreparationHistory.retainedMetadata(root:root,projectID:projectID,operationID:id,budget:budget)
                    case .backend:metadata=try BackendFitHistoryStore.retainedMetadata(root:root,projectID:projectID,operationID:id,budget:budget)
                    }
                    rows.append(metadata.map { .metadata($0) } ?? .absent)
                } catch is CancellationError {throw CancellationError()}
                catch is ReadFailure {throw ReadFailure.metadataBudgetExceeded}
                catch {rows.append(.issue(.init(operationID:id,code:"receipt_unavailable")))}
            }
            try check()
            return .init(projectID:projectID,family:family,receiptBytesRead:budget.consumed,rows:rows)
        }
    }
    static func inspect(root:URL,projectID:String,metadata:Metadata)async throws->Verified {
        let root=physicalRoot(root)
        return try await WorkflowHistoryVerificationWork.runCancellable {check in
            try check()
            let result:Verified
            switch metadata.family {
            case .preparation:result = .preparation(try WorkflowDataPreparationHistory.retainedEntry(root:root,projectID:projectID,operationID:metadata.operationID))
            case .backend:result = .backend(try BackendFitHistoryStore.retainedEntry(root:root,projectID:projectID,operationID:metadata.operationID))
            }
            guard result.metadata==metadata else {throw WorkflowArchiveError("History evidence changed since listing. Refresh before inspection.")}
            try check();return result
        }
    }
}
