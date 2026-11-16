import Foundation
import Darwin
import KwipMLCore

struct ModelImportedResume:Codable,Identifiable {
    let id:String
    let projectID:String
    let baseIdentity:String
    let files:[WorkflowArchiveFile]
    let completedEpoch:Int
    let importedAt:Date
    func directory(_ root:URL)->URL{root.appendingPathComponent("model-assets/\(projectID)/\(id)")}
    func capture(root:URL,base:ModelAssetReference)throws->ModelResumeCapture {
        guard UUID(uuidString:id)?.uuidString==id,UUID(uuidString:projectID)?.uuidString==projectID,base.projectID==projectID,base.identity==baseIdentity else{throw WorkflowFileLeaseError.unsafePath}
        for file in files{try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory(root),limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}
        return try Self.inspect(directory(root),base:base)
    }
    static func inspect(_ source:URL,base:ModelAssetReference)throws->ModelResumeCapture {
        let manifestBytes=try DiagnosticFileIO.read(root:source,parts:["result-manifest.json"],limit:8_000_000),reportBytes=try DiagnosticFileIO.read(root:source,parts:["training.json"],limit:8_000_000)
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:manifestBytes),report=try JSONDecoder().decode([String:JSONValue].self,from:reportBytes)
        let files=try JSONDecoder().decode([WorkflowArchiveFile].self,from:WorkflowArchiveCodec.encode(manifest["files"] ?? .null))
        guard manifest["schemaVersion"]==1,(1...256).contains(files.count),Set(files.map(\.path)).count==files.count,files.reduce(Int64(0),{$0+Int64($1.bytes)})<=2_010_000_000,report["kind"] == .string("kwipFineTuning"),report["continuation"] == .string("cpuEpochBoundary"),report["baseIdentity"] == .string(base.identity),let epoch=report["completedEpoch"]?.integerValue,(1...10000).contains(epoch) else{throw WorkflowArchiveError("Choose a bounded CPU resume package with the exact connected base.")}
        let nested=files.contains{$0.path.hasPrefix("latest/")}
        let dir=source.appendingPathComponent(nested ? "latest":"")
        for file in files {
            let parts=file.path.split(separator:"/").map(String.init)
            guard file.bytes>0,WorkflowArchiveCodec.isHash(file.sha256),parts.count==1 && ModelFineTuningFiles.leaf(parts[0]) || nested && parts.count==2 && ["latest","base"].contains(parts[0]) && ModelFineTuningFiles.leaf(parts[1]) else{throw WorkflowFileLeaseError.unsafePath}
            try WorkflowPreviewToolContract.verifyFile(file:file,directory:source,limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)
        }
        guard files.first(where:{$0.path=="training.json"}) == WorkflowArchiveFile(path:"training.json",bytes:reportBytes.count,sha256:WorkflowArchiveCodec.hash(reportBytes)) else{throw WorkflowFileLeaseError.unsafePath}
        if nested {
            guard files.first(where:{$0.path=="training.json"}) == WorkflowArchiveFile(path:"training.json",bytes:reportBytes.count,sha256:WorkflowArchiveCodec.hash(reportBytes)) else{throw WorkflowFileLeaseError.unsafePath}
            let packagedBase=files.filter{$0.path.hasPrefix("base/") && $0.path != "base/kwip-manifest.json"}.map{WorkflowArchiveFile(path:String($0.path.dropFirst(5)),bytes:$0.bytes,sha256:$0.sha256)}.sorted{$0.path<$1.path}
            if !packagedBase.isEmpty {guard packagedBase==base.files.filter{$0.path != "kwip-manifest.json"}.sorted(by:{$0.path<$1.path}) else{throw WorkflowArchiveError("Packaged base differs from the selected immutable base.")}}
        }
        let leaves=files.filter{nested ? $0.path.hasPrefix("latest/"):true}.map{WorkflowArchiveFile(path:nested ? String($0.path.dropFirst(7)):$0.path,bytes:$0.bytes,sha256:$0.sha256)}
        guard leaves.contains(where:{$0.path=="kwip-continuation.json"}),leaves.contains(where:{ModelTaskArtifactNames.continuationTensor($0.path)}),leaves.count<=128 else{throw WorkflowArchiveError("Resume package lacks safe optimizer/RNG state.")}
        var mapped=leaves.map{WorkflowArchiveFile(path:"latest/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}
        mapped.append(.init(path:"training.json",bytes:reportBytes.count,sha256:WorkflowArchiveCodec.hash(reportBytes)))
        let generated=try WorkflowArchiveCodec.encode(["schemaVersion":JSONValue.integer(1),"files":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(mapped.sorted{$0.path<$1.path}))])
        return .init(source:dir,files:leaves,report:report,reportBytes:reportBytes,manifestBytes:generated)
    }
    static func importPackage(_ source:URL,base:ModelAssetReference,root:URL,accountingRoot:URL?,limit:Int64?)throws->Self {
        let grant=try inspect(source,base:base),id=UUID().uuidString,target=root.appendingPathComponent("model-assets/\(base.projectID)/\(id)")
        let inputFiles=grant.inputFiles(weightsOnly:false).map{WorkflowArchiveFile(path:String($0.path.dropFirst("input/resume/".count)),bytes:$0.bytes,sha256:$0.sha256)}
        let budget=try limit.map{try WorkflowStorageAdmission(workspace:root,accountingRoot:accountingRoot).reserve(scopes:[.init(namespace:.modelAssets,components:[base.projectID,id])],peakBytes:inputFiles.reduce(Int64(1_000_000),{$0+Int64($1.bytes)}),limitBytes:$0)};defer{budget?.close()}
        try FileManager.default.createDirectory(at:target.appendingPathComponent("latest"),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let ownership=try WorkflowFrozenSnapshotDirectory(directory:target,removeOnDeinit:false)
        do {
            let from=try WorkflowEvaluationOwnership.openDirectory(grant.source);defer{Darwin.close(from)}
            let to=try WorkflowEvaluationOwnership.openDirectory(target.appendingPathComponent("latest"));defer{Darwin.close(to)}
            for file in grant.files{try Task.checkCancellation();guard try ModelAssetFiles.stream(file.path,parent:from,destination:to,maximum:file.bytes)==file else{throw WorkflowFileLeaseError.unsafePath}}
            try DiagnosticFileIO.writeImmutable(grant.reportBytes,root:target,parts:["training.json"],limit:8_000_000)
            try DiagnosticFileIO.writeImmutable(grant.manifestBytes,root:target,parts:["result-manifest.json"],limit:8_000_000)
            let record=Self(id:id,projectID:base.projectID,baseIdentity:base.identity,files:inputFiles,completedEpoch:Int(grant.report["completedEpoch"]?.integerValue ?? 0),importedAt:Date())
            _=try record.capture(root:root,base:base)
            try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:target,parts:["imported-resume.json"],limit:2_000_000)
            return record
        }catch{ownership.remove();throw error}
    }
    static func scan(root:URL,project:String)->[Self] {
        guard UUID(uuidString:project) != nil else{return []};let folder=root.appendingPathComponent("model-assets/"+project)
        return ((try? FileManager.default.contentsOfDirectory(atPath:folder.path)) ?? []).sorted().suffix(1000).compactMap{name in guard UUID(uuidString:name) != nil,let raw=try? DiagnosticFileIO.read(root:folder,parts:[name,"imported-resume.json"],limit:2_000_000),let record=try? JSONDecoder().decode(Self.self,from:raw),record.id==name,record.projectID==project else{return nil};return record}.sorted{$0.importedAt>$1.importedAt}
    }
}
