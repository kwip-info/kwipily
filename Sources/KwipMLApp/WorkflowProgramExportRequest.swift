import Foundation
import Darwin
import KwipMLCore

struct WorkflowProgramExportIdentity: Codable, Equatable {
    let runID: String
    let requestIdentity: String
    let modelManifestSHA256: String
}

struct WorkflowProgramExportRequest: Codable, Equatable {
    var schemaVersion: Int = 1
    var kind: String = "programExport"
    let projectID: String
    let exportID: String
    let requestIdentity: String
    let sourceDirectory: String
    let sourceManifestSHA256: String
    let configuration: [String: JSONValue]
    var executionBudget: ResolvedExecutionBudget? = nil
    func validate() throws {
        try WorkflowWorkerLaunchBudget.validatePolicy(schemaVersion:schemaVersion,executionBudget:executionBudget)
        guard kind == "programExport", UUID(uuidString:projectID)?.uuidString == projectID,
              UUID(uuidString:exportID)?.uuidString == exportID,
              WorkflowArchiveCodec.isHash(requestIdentity),WorkflowArchiveCodec.isHash(sourceManifestSHA256),
              sourceDirectory.utf8.count <= 4096,sourceDirectory.hasPrefix("/"),!sourceDirectory.contains("\0"),
              !sourceDirectory.split(separator:"/",omittingEmptySubsequences:false).dropFirst().contains(where:{$0.isEmpty || $0 == "." || $0 == ".."}) else {throw WorkflowFileLeaseError.unsafePath}
        _ = try ProgramExportConfiguration(values:configuration)
        let suffix=Array(sourceDirectory.split(separator:"/").suffix(4)).map(String.init)
        guard suffix.count == 4,["workflow-operations","workflow-program-exports"].contains(suffix[0]),
              Array(suffix.dropFirst()) == [projectID,exportID,"source"],let physical=realpath(sourceDirectory,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(physical)}
        guard String(cString:physical) == sourceDirectory else {throw WorkflowFileLeaseError.unsafePath}
        let fd=try WorkflowEvaluationOwnership.openDirectory(URL(fileURLWithPath:sourceDirectory))
        defer {Darwin.close(fd)}
        _ = try WorkflowEvaluationOwnership.directoryIdentity(fd)
    }
    func encoded() throws -> Data { try validate(); return try WorkflowArchiveCodec.encode(self) }
    func requestSHA256() throws -> String { WorkflowArchiveCodec.hash(try encoded()) }
    static func decode(_ bytes:Data)throws->Self {
        guard bytes.count <= 256_000,let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else {throw WorkflowFileLeaseError.unsafePath}
        let value=try JSONDecoder().decode(Self.self,from:bytes)
        try WorkflowWorkerLaunchBudget.validatePolicyFields(object,required:["schemaVersion","kind","projectID","exportID","requestIdentity","sourceDirectory","sourceManifestSHA256","configuration"],schemaVersion:value.schemaVersion,executionBudget:value.executionBudget)
        try value.validate();return value
    }
}
