import Foundation
import Darwin
import CryptoKit
import KwipMLCore

/// Stream fixed reviewed data names through held directory descriptors. No
/// recursive scan, symlinks, source code or whole-weight Data allocations.
enum ModelAssetFiles {
    static func inventory(_ directory:URL)throws->[WorkflowArchiveFile] {
        let parent=try WorkflowEvaluationOwnership.openDirectory(directory);defer {Darwin.close(parent)}
        var names=try FileManager.default.contentsOfDirectory(atPath:directory.path).filter {ModelHubRequest.allowedFile($0) && !$0.contains("/") || $0=="kwip-manifest.json"}.sorted()
        let typed=names.contains("rl_agent_config.json")
        if typed {names += ModelHubRequest.nestedModelFiles.filter{FileManager.default.fileExists(atPath:directory.appendingPathComponent($0).path)};names.sort()}
        if !typed {
        guard (3...129).contains(names.count),names.contains("config.json"),(names.contains("tokenizer.json") || names.contains("preprocessor_config.json")),names.contains(where:{$0.hasSuffix(".safetensors")}) else {throw WorkflowArchiveError("Choose a supported model folder with config.json, safe weights, and its task-matched tokenizer or audio processor.")}
        } else {guard Set(["rl_agent_config.json","encoder/config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","model.safetensors"]).isSubset(of:Set(names)) else{throw WorkflowArchiveError("Choose a complete Laya encoder/tokenizer/decision package.")}}
        var files:[WorkflowArchiveFile]=[],total:Int64=0
        for name in names {
            let result=try stream(name,parent:parent,destination:nil,maximum:name.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)
            total+=Int64(result.bytes);guard total<=2_010_000_000 else {throw WorkflowArchiveError("Model exceeds the 2 GB source admission limit.")};files.append(result)
        }
        if typed {return files}
        let config=try JSONDecoder().decode([String:JSONValue].self,from:WorkflowEvaluationOwnership.readProtocol(directory,name:"config.json",limit:8_000_000))
        guard requiredSourceFiles(config).isSubset(of:Set(names)) else {throw WorkflowArchiveError("Model lacks its task-matched tokenizer or audio processor.")}
        return files
    }
    static func requiredSourceFiles(_ config:[String:JSONValue])->Set<String> {
        if config["encoder"] != nil && config["head_layers"] != nil {return ["rl_agent_config.json","encoder/config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json"]}
        return config["model_type"] == .string("audio-spectrogram-transformer") && config["architectures"] == .array([.string("ASTForAudioClassification")]) ? ["config.json","preprocessor_config.json"]:["config.json","tokenizer.json"]
    }
    static func copy(_ files:[WorkflowArchiveFile],from source:URL,to destination:URL)throws {
        let parent=try WorkflowEvaluationOwnership.openDirectory(source);defer {Darwin.close(parent)}
        let target=try WorkflowEvaluationOwnership.openDirectory(destination);defer {Darwin.close(target)}
        for file in files {
            guard ModelHubRequest.allowedAssetFile(file.path) || file.path=="kwip-manifest.json" else {throw WorkflowFileLeaseError.unsafePath}
            let result=try stream(file.path,parent:parent,destination:target,maximum:file.bytes)
            guard result==file else {throw WorkflowArchiveError("Model file changed while snapshotting: "+file.path)}
        }
    }
    static func stream(_ name:String,parent:Int32,destination:Int32?,maximum:Int,destinationName:String?=nil)throws->WorkflowArchiveFile {
        try Task.checkCancellation()
        if name.contains("/") {
            guard destinationName==nil,ModelHubRequest.nestedModelFiles.contains(name) else{throw WorkflowFileLeaseError.unsafePath}
            let parts=name.split(separator:"/").map(String.init)
            let source=openat(parent,parts[0],O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard source>=0 else{throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(source)}
            var target:Int32?=nil
            if let destination {
                if mkdirat(destination,parts[0],0o700) != 0 && errno != EEXIST {throw WorkflowFileLeaseError.system(errno)}
                let fd=openat(destination,parts[0],O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
                guard fd>=0 else{throw WorkflowFileLeaseError.unsafePath};target=fd
            }
            defer{if let target{Darwin.close(target)}}
            let result=try stream(parts[1],parent:source,destination:target,maximum:maximum)
            return .init(path:name,bytes:result.bytes,sha256:result.sha256)
        }
        guard [name,destinationName ?? name].allSatisfy({!$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("/") && !$0.contains("\0")}) else {throw WorkflowFileLeaseError.unsafePath}
        let fd=openat(parent,name,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)
        guard fd>=0 else {throw WorkflowArchiveError("Model files must be readable regular files.")};defer {Darwin.close(fd)}
        var before=stat();guard fstat(fd,&before)==0,before.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),before.st_size>0,before.st_size<=maximum else {throw WorkflowArchiveError("Model file is unsafe or exceeds its byte budget.")}
        var output:Int32 = -1
        if let destination {output=openat(destination,destinationName ?? name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600);guard output>=0 else {throw WorkflowFileLeaseError.unsafePath}}
        defer {if output>=0 {Darwin.close(output)}}
        var hash=SHA256(),count=0,buffer=[UInt8](repeating:0,count:1_048_576)
        while true {
            try Task.checkCancellation()
            let n=read(fd,&buffer,buffer.count);guard n>=0 else {throw WorkflowFileLeaseError.system(errno)};if n==0 {break}
            count+=n;guard count<=maximum else {throw WorkflowFileLeaseError.unsafePath}
            let data=Data(buffer.prefix(n));hash.update(data:data)
            if output>=0 {try data.withUnsafeBytes {raw in var offset=0;while offset<n {let written=write(output,raw.baseAddress!.advanced(by:offset),n-offset);guard written>0 else {throw WorkflowFileLeaseError.system(errno)};offset+=written}}}
        }
        var after=stat(),named=stat()
        guard fstat(fd,&after)==0,fstatat(parent,name,&named,AT_SYMLINK_NOFOLLOW)==0,WorkflowEvaluationOwnership.identity(before)==WorkflowEvaluationOwnership.identity(named),before.st_size==after.st_size,before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec,count==Int(before.st_size) else {throw WorkflowArchiveError("Model file changed during the operation.")}
        if output>=0,fsync(output) != 0 {throw WorkflowFileLeaseError.system(errno)}
        return .init(path:name,bytes:count,sha256:hash.finalize().map{String(format:"%02x",$0)}.joined())
    }
    static func sourceIdentity(_ files:[WorkflowArchiveFile])throws->String {
        // Python uses alphabetic object keys and compact separators for the same inventory.
        let data=try JSONSerialization.data(withJSONObject:files.map{["path":$0.path,"bytes":$0.bytes,"sha256":$0.sha256] as [String:Any]},options:[.sortedKeys,.withoutEscapingSlashes])
        return WorkflowArchiveCodec.hash(data)
    }
}

struct ModelAssetReference:WorkflowPrivateTrainReference {
    static let storageNamespace="workflow-model-assets"
    static let fieldNames:Set<String>=["projectID","directoryID","metadata","sourceConfiguration"]
    let projectID:String
    let directoryID:String
    let metadata:[String:JSONValue]
    let sourceConfiguration:[String:JSONValue]
    var identity:String {metadata["identity"]?.stringValue ?? ""}
    var files:[WorkflowArchiveFile] {(try? JSONDecoder().decode([WorkflowArchiveFile].self,from:JSONEncoder().encode(metadata["files"]))) ?? []}
    var hasValidPins:Bool {WorkflowArchiveCodec.isHash(identity) && !files.isEmpty && files.count<=128 && files.allSatisfy {WorkflowArchiveCodec.isHash($0.sha256) && (sourceConfiguration["source"] == .string("documentPackage") ? ["state.json","preparation.json"].contains($0.path):ModelHubRequest.allowedAssetFile($0.path))} && (try? LoadModelConfiguration.decode(sourceConfiguration)) != nil}
    static func accepts(_ node:WorkflowNode)->Bool {if case .frozenModel=node.payload{return true};return FineTuningConfiguration.isExternal(node)}
    func directory(root:URL)->URL {root.appendingPathComponent(metadata["trainingOperation"] == .bool(true) ? "workflow-operations/\(projectID)/\(directoryID)/output/training/model":"model-assets/\(projectID)/\(directoryID)")}
    func matches(_ configuration:LoadModelConfiguration)->Bool {
        guard let prior=try? LoadModelConfiguration.decode(sourceConfiguration),configuration.source != .savedRun else {return false}
        return prior.source==configuration.source && (prior.source != .huggingFace || (prior.repository==configuration.repository && prior.revision==configuration.revision && prior.subfolder==configuration.subfolder))
    }
}

enum ModelAdapterFiles {
    static let names:Set<String>=["kwip-training.json","adapter_config.json","adapter_model.safetensors","kwip-adapter.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json","vocab.json","merges.txt","tokenizer.model","README.md","LICENSE","LICENSE.txt","NOTICE","NOTICE.txt"]
    static func inventory(_ directory:URL)throws->[WorkflowArchiveFile] {
        let parent=try WorkflowEvaluationOwnership.openDirectory(directory);defer {Darwin.close(parent)}
        let leaves=try FileManager.default.contentsOfDirectory(atPath:directory.path).filter{names.contains($0)}.sorted()
        guard Set(["adapter_config.json","adapter_model.safetensors","kwip-adapter.json","tokenizer.json"]).isSubset(of:Set(leaves)) else {throw WorkflowArchiveError("Choose a Kwip LoRA export with its tokenizer and exact base provenance.")}
        var result:[WorkflowArchiveFile]=[],total=0
        for name in leaves {let file=try ModelAssetFiles.stream(name,parent:parent,destination:nil,maximum:name.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000);total+=file.bytes;guard total<=2_010_000_000 else {throw WorkflowArchiveError("Adapter exceeds the source byte budget.")};result.append(file)}
        return result
    }
    static func copy(_ files:[WorkflowArchiveFile],from source:URL,to target:URL)throws {
        let parent=try WorkflowEvaluationOwnership.openDirectory(source);defer {Darwin.close(parent)}
        let destination=try WorkflowEvaluationOwnership.openDirectory(target);defer {Darwin.close(destination)}
        for file in files {guard names.contains(file.path),try ModelAssetFiles.stream(file.path,parent:parent,destination:destination,maximum:file.bytes)==file else {throw WorkflowFileLeaseError.unsafePath}}
    }
}
