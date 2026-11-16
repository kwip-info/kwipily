import Foundation
import KwipMLCore

/// Private credential REFERENCES only; tokens remain in Keychain and never enter the graph.
struct ModelHubCredentialSelection {
    let root:URL
    let project:String
    private struct Envelope:Codable {var schemaVersion=1;let projectID:String;var names:[String:String]}
    private var parts:[String] {["workflow-hub-credentials",project,"selections.json"]}
    private func values()throws->[String:String] {
        guard UUID(uuidString:project)?.uuidString==project else {throw WorkflowFileLeaseError.unsafePath}
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:100_000) else {return [:]}
        let value=try JSONDecoder().decode(Envelope.self,from:bytes)
        guard value.schemaVersion==1,value.projectID==project,value.names.count<=256,value.names.allSatisfy({!$0.key.isEmpty && $0.key.utf8.count<=128 && !$0.key.contains("\0") && !$0.value.isEmpty && $0.value.utf8.count<=100}) else {throw WorkflowArchiveError("Saved Hub credential references are invalid. They were retained.")}
        return value.names
    }
    func name(for address:WorkflowNodeAddress)throws->String? {guard address.domain == .control else {throw WorkflowFileLeaseError.unsafePath};return try values()[address.localID]}
    func select(_ name:String?,for address:WorkflowNodeAddress)throws {
        guard address.domain == .control,!address.localID.isEmpty,address.localID.utf8.count<=128,!address.localID.contains("\0"),name == nil || (name?.isEmpty == false && name!.utf8.count<=100) else {throw WorkflowFileLeaseError.unsafePath}
        var next=try values();next[address.localID]=name
        guard next.count<=256 else {throw WorkflowArchiveError("Hub credential references exceed their bounded project limit.")}
        try DiagnosticFileIO.writeAtomic(JSONEncoder().encode(Envelope(projectID:project,names:next)),root:root,parts:parts)
    }
}
