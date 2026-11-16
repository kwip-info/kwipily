import Foundation
import KwipMLCore

struct WorkflowEvaluationExportIdentity:Codable,Equatable {
    let evaluationID:String
    let evaluationManifestSHA256:String
    let checkpointSHA256:String
    let stateDigest:String
    let snapshotReceiptSHA256:String
}
struct WorkflowEvaluationExportManifest:Codable {
    let schemaVersion:Int
    var partitionRole:String? = nil
    let kind:String
    let includesTestRows:Bool
    let identities:WorkflowEvaluationExportIdentity
    let files:[WorkflowArchiveFile]
}

/// Native framing/inventory verification of a package produced by the trusted
/// worker. Model semantics and fitted computation are independently checked there.
struct WorkflowEvaluationExportPackage {
    static let requiredLimits:[String:Int]=["model.py":2_000_000,"weights.pt":100_000_000,"input-contract.json":2_000_000,
        "fitted-parameters.json":20_000_000,"saved-summary.json":1_000_000,"portable_evaluation.py":200_000,
        "portable_fitted.py":200_000,"final_evaluation.py":200_000,"execution_device.py":200_000,
        "requirements.txt":10_000,"README.md":100_000]
    /// Reviewed adjacent implementations added by current workers. Older packages
    /// remain valid without these modules; every present module is still inventoried and hashed.
    static let dependencyLimits:[String:Int]=["preprocess_parameters.py":200_000,"dataflow_temporal.py":200_000,"model_output_configuration.py":200_000]
    static let optionalLimits:[String:Int]=["test-input.json":50_000_000,"saved-predictions.jsonl":100_000_000]
    let directory:URL
    let cleanupDirectory:URL
    let identity:WorkflowEvaluationExportIdentity
    let includesTestRows:Bool
    let files:[WorkflowArchiveFile]
    let manifestBytes:Data
    let columns:[ColumnContract]
    let featureColumns:[String]
    let targetColumn:String
    let dtype:String
    let sampleShape:[Int]
    let testCount:Int
    let fittedStateCount:Int
    var storageLifetime: WorkflowExportStorageLifetime? = nil
    func remove() {
        if let storageLifetime { storageLifetime.remove() }
        else { try? FileManager.default.removeItem(at: cleanupDirectory) }
    }
    var totalBytes:Int {files.reduce(manifestBytes.count) {$0+$1.bytes}}
    var fileCount:Int {files.count+1}
    private struct Input:Decodable {
        let schemaVersion:Int;let partitionRole:String?;let kind:String;let columns:[ColumnContract];let featureColumns:[String];let targetColumn:String
        let dtype:String;let sampleShape:[Int];let fittedStateDigests:[String];let appliedPrefix:[String]
        let testEndpoint:DataFlowEndpoint;let testCount:Int;let membershipDigest:String;let selectedDataVersion:String
    }
    static func verify(directory:URL,expected:WorkflowEvaluationExportIdentity,includesTestRows:Bool,manifestSHA256:String,cleanupDirectory:URL?=nil)throws->Self {
        try Task.checkCancellation()
        if let cleanupDirectory {
            guard directory.standardizedFileURL==cleanupDirectory.appendingPathComponent("package").standardizedFileURL else {throw failure("Export cleanup scope differs from its owned staging folder.")}
        }
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["manifest.json"],limit:2_000_000)
        guard WorkflowArchiveCodec.isHash(manifestSHA256),WorkflowArchiveCodec.hash(bytes)==manifestSHA256 else {throw failure("Export manifest differs from the completed worker result.")}
        let raw=try JSONSerialization.jsonObject(with:bytes) as? [String:Any]
        let validation=raw?["schemaVersion"] as? Int == 3
        let object=try keys(bytes,validation ? ["schemaVersion","kind","partitionRole","includesTestRows","identities","files"]:["schemaVersion","kind","includesTestRows","identities","files"])
        guard let identity=object["identities"] as? [String:Any],Set(identity.keys)==["evaluationID","evaluationManifestSHA256","checkpointSHA256","stateDigest","snapshotReceiptSHA256"] else {throw failure("Invalid export identity fields.")}
        let manifest=try JSONDecoder().decode(WorkflowEvaluationExportManifest.self,from:bytes)
        guard (([1,2].contains(manifest.schemaVersion) && manifest.partitionRole == nil) || (manifest.schemaVersion == 3 && manifest.partitionRole == "validation")),manifest.kind=="portableFrozenEvaluation",manifest.includesTestRows==includesTestRows,manifest.identities==expected,
              UUID(uuidString:expected.evaluationID)?.uuidString==expected.evaluationID,
              [expected.evaluationManifestSHA256,expected.checkpointSHA256,expected.stateDigest,expected.snapshotReceiptSHA256].allSatisfy(WorkflowArchiveCodec.isHash) else {throw failure("Export differs from the selected evaluation or privacy option.")}
        let dependencies=dependencyLimits.filter { dependency in manifest.files.contains {$0.path == dependency.key} }
        let limits=requiredLimits.merging(includesTestRows ? optionalLimits:[:]) {a,_ in a}.merging(dependencies) {a,_ in a}
        guard manifest.files.count==limits.count,Set(manifest.files.map(\.path))==Set(limits.keys),
              Set(try FileManager.default.contentsOfDirectory(atPath:directory.path))==Set(limits.keys).union(["manifest.json"]) else {throw failure("Export inventory contains missing, duplicate or unreviewed files.")}
        var total=0
        for file in manifest.files {
            try Task.checkCancellation()
            guard let maximum=limits[file.path],file.bytes>=0,file.bytes<=maximum,file.bytes<=280_000_000-total,WorkflowArchiveCodec.isHash(file.sha256) else {throw failure("Export artifact exceeds its supported bounds.")}
            total+=file.bytes
            let content=try DiagnosticFileIO.read(root:directory,parts:[file.path],limit:file.bytes)
            guard content.count==file.bytes,WorkflowArchiveCodec.hash(content)==file.sha256 else {throw failure("Export artifact changed: \(file.path).")}
        }
        let summaryBytes=try DiagnosticFileIO.read(root:directory,parts:["saved-summary.json"],limit:1_000_000)
        let summary=try JSONSerialization.jsonObject(with:summaryBytes) as? [String:Any]
        try WorkflowEvaluationComputationCompatibility.validate(version:validation ? 2:manifest.schemaVersion,actualDevice:summary?["device"] as? String)
        let inputBytes=try DiagnosticFileIO.read(root:directory,parts:["input-contract.json"],limit:2_000_000)
        let inputKeys:Set<String>=["schemaVersion","kind","columns","featureColumns","targetColumn","dtype","sampleShape","fittedStateDigests","appliedPrefix","testEndpoint","testCount","membershipDigest","selectedDataVersion"]
        _ = try keys(inputBytes,validation ? inputKeys.union(["partitionRole"]):inputKeys)
        let input=try JSONDecoder().decode(Input.self,from:inputBytes)
        guard (validation ? (input.schemaVersion==2 && input.partitionRole=="validation"):(input.schemaVersion==1 && input.partitionRole==nil)),input.kind=="portableEvaluationInput",(1...128).contains(input.columns.count),
              Set(input.columns.map(\.name)).count==input.columns.count,input.columns.allSatisfy({!$0.name.isEmpty && $0.name.utf8.count<=800}),
              ["float32","int64"].contains(input.dtype),(1...7).contains(input.sampleShape.count),input.sampleShape.allSatisfy({(1...128).contains($0)}),
              (1...100000).contains(input.testCount),input.fittedStateDigests.count<=128,input.fittedStateDigests.allSatisfy(WorkflowArchiveCodec.isHash),
              Array(input.fittedStateDigests.prefix(input.appliedPrefix.count))==input.appliedPrefix,
              WorkflowArchiveCodec.isHash(input.membershipDigest),WorkflowArchiveCodec.isHash(input.selectedDataVersion) else {throw failure("Export input contract is invalid.")}
        var width=1
        for dimension in input.sampleShape {guard width<=128/dimension else {throw failure("Export input shape exceeds its bound.")};width*=dimension}
        try TrainingColumnMapping.validate(features:input.featureColumns,target:input.targetColumn)
        guard input.featureColumns.count==width else {throw failure("Export feature mapping differs from its tensor shape.")}
        // Final mapped names can be created by remaining frozen transforms; the
        // selected endpoint's columns are the table schema shown in this review.
        let parameters=try DiagnosticFileIO.read(root:directory,parts:["fitted-parameters.json"],limit:20_000_000)
        guard let states=try JSONSerialization.jsonObject(with:parameters) as? [[String:Any]],states.count==input.fittedStateDigests.count else {throw failure("Export learned parameter count differs from its contract.")}
        return .init(directory:directory,cleanupDirectory:cleanupDirectory ?? directory,identity:expected,includesTestRows:includesTestRows,files:manifest.files,manifestBytes:bytes,
                     columns:input.columns,featureColumns:input.featureColumns,targetColumn:input.targetColumn,dtype:input.dtype,
                     sampleShape:input.sampleShape,testCount:input.testCount,fittedStateCount:states.count)
    }
    func publish(to destination:URL)throws {
        try WorkflowExportPublication.publish(directory:directory,files:files,manifestName:"manifest.json",manifestBytes:manifestBytes,to:destination)
    }
    private static func keys(_ bytes:Data,_ expected:Set<String>)throws->[String:Any] {
        guard let value=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],Set(value.keys)==expected else {throw failure("Unexpected export fields.")};return value
    }
    private static func failure(_ message:String)->WorkflowArchiveError {WorkflowArchiveError(message)}
}
