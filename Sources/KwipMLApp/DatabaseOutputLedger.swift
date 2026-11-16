import Foundation
import KwipMLCore

/// Local durable effect evidence, never a trigger. A leftover intent is unknown.
/// One canonical project owner serializes mutations; no external/global exactly-once claim.
struct DatabaseOutputLedger {
    let root:URL
    let projectID:String
    struct Record:Codable,Equatable,Identifiable {
        let id:String
        let nodeID:String
        let preparationID:String
        let targetIdentity:String
        let configuration:[String:JSONValue]
        let sourceSHA256:String
        let createdAt:Date
        var receipt:[String:JSONValue]
        var acknowledged:Bool=false
    }
    private struct Envelope:Codable {let schemaVersion:Int;let records:[Record]}
    private var parts:[String] {["database-output-records",projectID,"records.json"]}
    func load()throws->[Record] {
        guard UUID(uuidString:projectID)?.uuidString==projectID else {throw WorkflowFileLeaseError.unsafePath}
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:2_000_000) else {return []}
        let value=try JSONDecoder().decode(Envelope.self,from:bytes)
        guard value.schemaVersion==1,value.records.count<=1000,Set(value.records.map(\.id)).count==value.records.count else {throw WorkflowArchiveError("Database write ledger is invalid; existing evidence was preserved.")}
        for r in value.records {
            guard UUID(uuidString:r.id)?.uuidString==r.id,UUID(uuidString:r.preparationID)?.uuidString==r.preparationID,!r.nodeID.isEmpty,r.nodeID.utf8.count<=128,WorkflowArchiveCodec.isHash(r.targetIdentity),WorkflowArchiveCodec.isHash(r.sourceSHA256),["intent","committed","rolledBack","notStarted","unknown"].contains(r.receipt["status"]?.stringValue ?? "") else {throw WorkflowFileLeaseError.unsafePath}
            if r.configuration["kind"] == .string("api") {_ = try APITableSettings.normalized(r.configuration)} else {_ = try DatabaseOutputSettings.normalized(r.configuration)}
        }
        return value.records
    }
    func persist(_ records:[Record])throws {
        guard records.count<=1000 else {throw WorkflowArchiveError("Database write receipt limit reached. Preserve/export existing evidence before further writes.")}
        let bytes=try WorkflowArchiveCodec.encode(Envelope(schemaVersion:1,records:records))
        guard bytes.count<=2_000_000 else {throw WorkflowArchiveError("Database receipt storage allowance reached; existing receipts retained.")}
        let folder=root.appendingPathComponent("database-output-records/"+projectID)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        try DiagnosticFileIO.writeAtomic(bytes,root:root,parts:parts)
    }
    func begin(_ record:Record)throws {
        var all=try load()
        guard !all.contains(where:{$0.id==record.id || !$0.acknowledged && $0.targetIdentity==record.targetIdentity && ["intent","unknown"].contains($0.receipt["status"]?.stringValue ?? "")}) else {throw WorkflowArchiveError("A previous database write has an unknown outcome. Inspect its receipt and acknowledge it before a new write.")}
        all.insert(record,at:0);try persist(all)
    }
    func finish(id:String,receipt:[String:JSONValue])throws {
        guard ["committed","rolledBack","notStarted","unknown"].contains(receipt["status"]?.stringValue ?? "") else {throw WorkflowFileLeaseError.unsafePath}
        var all=try load();guard let i=all.firstIndex(where:{$0.id==id}),all[i].receipt["status"] == .string("intent") else {throw WorkflowArchiveError("Database write receipt cannot replace a terminal operation.")}
        if let supplied=receipt["operationID"],supplied != .string(id) {throw WorkflowFileLeaseError.unsafePath}
        all[i].receipt=receipt;try persist(all)
    }
    func acknowledge(_ id:String)throws {
        var all=try load();guard let i=all.firstIndex(where:{$0.id==id}),["intent","unknown"].contains(all[i].receipt["status"]?.stringValue ?? "") else {throw WorkflowFileLeaseError.unsafePath}
        all[i].acknowledged=true;try persist(all)
    }
}
