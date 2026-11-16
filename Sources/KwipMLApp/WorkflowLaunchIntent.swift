import Foundation
import KwipMLCore

struct WorkflowLaunchIntent:Codable,Equatable {
    enum Phase:String,Codable {case preparation,training}
    let schemaVersion:Int
    let projectID:String
    let groupID:String
    let generation:String
    let runID:String
    let directoryID:String
    let nonce:String
    let ordinal:Int
    let requestIdentity:String
    let ownershipSHA256:String
    let phase:Phase
    let arguments:[String]
    var executionBudget:ResolvedExecutionBudget? = nil
    var remainingSeconds:Double? = nil

    static func decode(_ bytes:Data)throws->Self {
        let fields:Set<String>=["schemaVersion","projectID","groupID","generation","runID","directoryID","nonce","ordinal","requestIdentity","ownershipSHA256","phase","arguments"]
        guard bytes.count <= 256000,let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else {throw WorkflowFileLeaseError.unsafePath}
        let intent = try JSONDecoder().decode(Self.self,from:bytes)
        let expected = intent.schemaVersion == 2 ? fields.union(["executionBudget", "remainingSeconds"]) : fields
        guard Set(object.keys) == expected,
              !(object["executionBudget"] is NSNull), !(object["remainingSeconds"] is NSNull) else {throw WorkflowFileLeaseError.unsafePath}
        try validateBudget(intent)
        if let budget = intent.executionBudget {
            guard let original = object["executionBudget"] as? [String:Any],
                  let normalized = try JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(budget)) as? [String:Any],
                  Set(original.keys) == Set(normalized.keys) else {throw WorkflowFileLeaseError.unsafePath}
        }
        return intent
    }
    private static func validateBudget(_ intent:Self)throws {
        switch (intent.schemaVersion,intent.executionBudget,intent.remainingSeconds) {
        case (1,nil,nil): break
        case let (2,budget?,remaining?):
            guard remaining.isFinite,remaining > 0,remaining <= Double(budget.maximumRunSeconds) else {throw WorkflowFileLeaseError.unsafePath}
        default: throw WorkflowFileLeaseError.unsafePath
        }
    }
    static func validate(_ intent:Self,groupDirectory:URL)throws {
        try validateBudget(intent)
        for id in [intent.projectID,intent.groupID,intent.generation,intent.runID,intent.directoryID,intent.nonce] {
            guard UUID(uuidString:id)?.uuidString == id else {throw WorkflowFileLeaseError.unsafePath}
        }
        guard (1...32).contains(intent.ordinal),WorkflowArchiveCodec.isHash(intent.requestIdentity),WorkflowArchiveCodec.isHash(intent.ownershipSHA256),groupDirectory.lastPathComponent == intent.groupID,groupDirectory.deletingLastPathComponent().lastPathComponent == intent.projectID,groupDirectory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-experiments" else {throw WorkflowFileLeaseError.unsafePath}
        let root=groupDirectory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try WorkflowPhaseArguments.validate(intent.arguments, workspace: root, projectID: intent.projectID,
            directoryID: intent.directoryID, phase: intent.phase)
    }

    static func matchesLedger(_ bytes:Data,intent:Self)throws {
        let record=try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:bytes)
        guard record.schemaVersion == 1,record.groupID == intent.groupID,record.projectID == intent.projectID,!record.stopRequested,(1...32).contains(record.variants.count),record.variants.enumerated().allSatisfy({$0.element.ordinal == $0.offset+1}),record.variants.filter({$0.status == .running}).count == 1,record.variants.indices.contains(intent.ordinal-1) else {throw WorkflowFileLeaseError.unsafePath}
        let variant=record.variants[intent.ordinal-1]
        guard variant.ordinal == intent.ordinal,variant.status == .running,variant.runID == intent.runID,variant.directoryID == intent.directoryID,variant.requestIdentity == intent.requestIdentity else {throw WorkflowFileLeaseError.unsafePath}
    }
}

struct WorkflowLaunchClaim:Codable,Equatable {
    let schemaVersion:Int
    let nonce:String
    let intentSHA256:String
    let state:String
    static func validClaimed(_ bytes:Data,nonce:String,intentSHA256:String)throws->Bool {
        guard let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],Set(object.keys) == ["schemaVersion","nonce","intentSHA256","state"] else {return false}
        let claim=try JSONDecoder().decode(Self.self,from:bytes)
        return claim.schemaVersion == 1 && claim.nonce == nonce && claim.intentSHA256 == intentSHA256 && claim.state == "claimed"
    }
}
