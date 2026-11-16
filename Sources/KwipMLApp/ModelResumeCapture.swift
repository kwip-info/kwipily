import Foundation
import Darwin
import KwipMLCore

/// Data-only resume grant: exact retained files are captured through descriptors,
/// never a path to an arbitrary Python checkpoint or an executable package.
struct ModelResumeCapture:Sendable {
    let source:URL
    let files:[WorkflowArchiveFile]
    let report:[String:JSONValue]
    let reportBytes:Data
    let manifestBytes:Data
    static func capture(_ record:ModelFineTuningRecord,root:URL,epoch:Int=0,weightsOnly:Bool=false)throws->Self {
        let original=try record.report(root:root)
        let directory=record.directory(root).appendingPathComponent("output/training")
        let prefix=epoch>0 ? "checkpoints/epoch-"+String(format:"%06d",epoch)+"/":"latest/"
        if weightsOnly,epoch>0 {throw WorkflowArchiveError("Weight reuse selects the recorded inference model; epoch selection belongs to exact resume.")}
        let group=weightsOnly ? "model/":prefix
        let relative=record.files.filter{$0.path.hasPrefix("output/training/"+group)}.map{WorkflowArchiveFile(path:String($0.path.dropFirst("output/training/".count)),bytes:$0.bytes,sha256:$0.sha256)}
        guard !relative.isEmpty,relative.count<=128 else{throw WorkflowArchiveError("This run does not retain the requested boundary.")}
        let report:[String:JSONValue]
        if epoch>0 {report=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:try WorkflowArchiveCodec.pathParts(prefix+"training.json"),limit:8_000_000))}else{report=original}
        guard report["baseIdentity"] == .string(record.base.identity),weightsOnly || report["continuation"] == .string("cpuEpochBoundary") else{throw WorkflowArchiveError("Exact resume needs a verified CPU boundary with the matching immutable base.")}
        let reportBytes=try WorkflowArchiveCodec.encode(report)
        let mapped=relative.map{WorkflowArchiveFile(path:(weightsOnly ? "model/":"latest/")+String($0.path.dropFirst(group.count)),bytes:$0.bytes,sha256:$0.sha256)}
        let manifestFiles=(mapped+[.init(path:"training.json",bytes:reportBytes.count,sha256:WorkflowArchiveCodec.hash(reportBytes))]).sorted{$0.path<$1.path}
        let manifest=try WorkflowArchiveCodec.encode(["schemaVersion":JSONValue.integer(1),"files":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(manifestFiles))])
        return .init(source:directory.appendingPathComponent(group),files:relative.map{.init(path:String($0.path.dropFirst(group.count)),bytes:$0.bytes,sha256:$0.sha256)},report:report,reportBytes:reportBytes,manifestBytes:manifest)
    }
    func inputFiles(weightsOnly:Bool)->[WorkflowArchiveFile] {files.map{.init(path:"input/resume/"+(weightsOnly ? "model/":"latest/")+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [.init(path:"input/resume/training.json",bytes:reportBytes.count,sha256:WorkflowArchiveCodec.hash(reportBytes)),.init(path:"input/resume/result-manifest.json",bytes:manifestBytes.count,sha256:WorkflowArchiveCodec.hash(manifestBytes))]}
    func copy(to directory:URL,weightsOnly:Bool)throws {
        let target=directory.appendingPathComponent("input/resume/"+(weightsOnly ? "model":"latest"))
        try FileManager.default.createDirectory(at:target,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let parent=try WorkflowEvaluationOwnership.openDirectory(source);defer{Darwin.close(parent)}
        let destination=try WorkflowEvaluationOwnership.openDirectory(target);defer{Darwin.close(destination)}
        for file in files {guard ModelFineTuningFiles.leaf(file.path),try ModelAssetFiles.stream(file.path,parent:parent,destination:destination,maximum:file.bytes)==file else{throw WorkflowFileLeaseError.unsafePath}}
        try DiagnosticFileIO.writeImmutable(reportBytes,root:directory,parts:["input","resume","training.json"],limit:8_000_000)
        try DiagnosticFileIO.writeImmutable(manifestBytes,root:directory,parts:["input","resume","result-manifest.json"],limit:8_000_000)
    }
}
enum ModelFineTuningFiles {
    static func leaf(_ name:String)->Bool { !name.contains("/") && (ModelHubRequest.allowedFile(name) || ModelTaskArtifactNames.continuationTensor(name) || ["kwip-manifest.json","adapter_config.json","adapter_model.safetensors","kwip-adapter.json","kwip-continuation.json","training.json","result-manifest.json"].contains(name)) }
    static func resumeInput(_ path:String)->Bool {
        let p=path.split(separator:"/").map(String.init)
        return p.count==3 && p[0]=="input" && p[1]=="resume" && ["training.json","result-manifest.json"].contains(p[2]) || p.count==4 && p[0]=="input" && p[1]=="resume" && ["latest","model"].contains(p[2]) && leaf(p[3])
    }
}
