import Foundation
import KwipMLCore

struct RunHistoryPolicy:Codable,Equatable {
    var metricsAndDiagnostics:Bool
    var componentOutputs:Bool
    static let all=Self(metricsAndDiagnostics:true,componentOutputs:true)
    static let none=Self(metricsAndDiagnostics:false,componentOutputs:false)
    var enabled:Bool {metricsAndDiagnostics || componentOutputs}
}
/// Optional observation retention is pinned at run acceptance. Required scientific
/// evidence is owned by the existing archives and never removed by this policy.
struct RunHistoryPolicyStore {
    let root:URL;let projectID:String
    struct Record:Codable {var schemaVersion=1;var projectID:String;var policy:RunHistoryPolicy}
    var fileURL:URL {root.appendingPathComponent("run-history-policy/"+projectID+".json")}
    private func parts(id:String?=nil)throws->[String] {
        guard UUID(uuidString:projectID)?.uuidString==projectID else {throw WorkflowArchiveError("Invalid run History project.")}
        if let id {guard UUID(uuidString:id)?.uuidString==id else {throw WorkflowArchiveError("Invalid run History identity.")};return ["run-history-policy",projectID,id+".json"]}
        return ["run-history-policy",projectID+".json"]
    }
    private func decode(_ bytes:Data)throws->RunHistoryPolicy {
        let record=try JSONDecoder().decode(Record.self,from:bytes)
        guard record.schemaVersion==1,record.projectID==projectID else {throw WorkflowArchiveError("Unsupported run History policy.")};return record.policy
    }
    private func encode(_ policy:RunHistoryPolicy)throws->Data {try JSONEncoder().encode(Record(projectID:projectID,policy:policy))}
    func load()throws->RunHistoryPolicy {
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts(),limit:4096) else {return .all}
        return try decode(bytes)
    }
    func save(_ policy:RunHistoryPolicy)throws {try DiagnosticFileIO.writeAtomic(encode(policy),root:root,parts:parts())}
    func capture(id:String,policy:RunHistoryPolicy)throws {
        let path=try parts(id:id)
        if let prior=try DiagnosticFileIO.readIfPresent(root:root,parts:path,limit:4096) {
            guard try decode(prior)==policy else {throw WorkflowArchiveError("An accepted run keeps its original History policy.")};return
        }
        try DiagnosticFileIO.writeImmutable(encode(policy),root:root,parts:path)
    }
    func decision(id:String)throws->RunHistoryPolicy {
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts(id:id),limit:4096) else {return .all}
        return try decode(bytes)
    }
}
