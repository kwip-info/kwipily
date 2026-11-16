import Foundation
import KwipMLCore

/// Retained selection metadata only. Never constructs a loader, opens model files,
/// reads credential sidecars, or claims current applicability/content verification.
enum ModelAssetStatusSnapshot {
    struct Item:Sendable {
        let value:[String:JSONValue]
        let components:Set<String>
    }
    struct Snapshot:Sendable {let items:[Item];let sourceStatus:String}
    private struct Entry:Decodable {let address:WorkflowNodeAddress;let reference:ModelAssetReference}
    private struct Envelope:Decodable {let schemaVersion:Int;let projectID:String;let revision:UInt64;let entries:[Entry]}
    static func load(root:URL,projectID:String)async throws->Snapshot {
        try await Task.detached(priority:.utility) {try read(root:root,projectID:projectID)}.value
    }
    static func read(root:URL,projectID:String)throws->Snapshot {
        guard let project=UUID(uuidString:projectID) else {throw MCPSettings.Refusal(code:"asset_invalid_project")}
        let parts=[ModelAssetReference.storageNamespace,project.uuidString,"bindings.json"]
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:1_000_000) else {return .init(items:[],sourceStatus:"noRetainedBindings")}
        guard let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],Set(object.keys)==["schemaVersion","projectID","revision","entries"],let entries=object["entries"] as? [[String:Any]],entries.count<=256 else {throw MCPSettings.Refusal(code:"asset_metadata_unavailable")}
        for entry in entries {
            guard Set(entry.keys)==["address","reference"],let address=entry["address"] as? [String:Any],Set(address.keys)==["domain","localID"],let reference=entry["reference"] as? [String:Any],Set(reference.keys)==ModelAssetReference.fieldNames else {throw MCPSettings.Refusal(code:"asset_metadata_unavailable")}
        }
        let envelope=try JSONDecoder().decode(Envelope.self,from:bytes)
        guard envelope.schemaVersion==1,UUID(uuidString:envelope.projectID)==project,Set(envelope.entries.map(\.address)).count==envelope.entries.count else {throw MCPSettings.Refusal(code:"asset_metadata_unavailable")}
        let items=try envelope.entries.sorted{$0.address.localID < $1.address.localID}.map {entry->Item in
            let ref=entry.reference
            guard entry.address.domain == .control,!entry.address.localID.isEmpty,entry.address.localID.unicodeScalars.count<=128,!entry.address.localID.contains("\0"),UUID(uuidString:ref.projectID)==project,UUID(uuidString:ref.directoryID) != nil else {throw MCPSettings.Refusal(code:"asset_metadata_unavailable")}
            let training=ref.metadata["trainingOperation"] == .bool(true)
            let valid=ref.hasValidPins
            var value:[String:JSONValue]=[
                "address":.object(["domain":.string("control"),"localID":.string(entry.address.localID)]),
                "status":.string(valid ? "retainedPinnedMetadata":"unknownProtectedMetadata"),
                "origin":.string(training ? "trainingSelection":"loadSelection"),
                "currentApplicability":.string("notVerified"),"contentVerification":.string("notPerformed"),
                "protected":.bool(true),"removable":.bool(false)]
            if valid {
                value["identity"] = .string(ref.identity)
                value["fileCount"] = .integer(Int64(ref.files.count))
                // Do not expose arbitrary metadata strings, paths, repository or credentials.
                if let task=ref.metadata["task"]?.stringValue,["causalLanguageModel","typedDecisions","speechTranscription","audioClassification","documentClassification"].contains(task) {value["recordedTask"] = .string(task)}
            }
            return .init(value:value,components:[training ? "workflow.train":"workflow.frozenModel"])
        }
        return .init(items:items,sourceStatus:"retainedMetadataOnly")
    }
}
