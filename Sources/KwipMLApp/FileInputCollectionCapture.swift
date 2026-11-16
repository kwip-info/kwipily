import Foundation
import Darwin
import KwipMLCore

struct FileInputCapturedMember:Equatable {
    let capturePath:String
    let url:URL
    let sha256:String
    let byteCount:Int
}

struct FileInputCollectionManifest:Codable {
    struct Member:Codable {
        let relativePath:String
        let capturePath:String?
        let bytes:Int64
        let sha256:String?
        let `extension`:String
        let status:String
        let modifiedAt:String?
        let createdAt:String?
        let error:String?
        enum CodingKeys:String,CodingKey {case relativePath,capturePath,bytes,sha256,`extension`,status,modifiedAt,createdAt,error}
        func encode(to encoder:Encoder)throws {
            var c=encoder.container(keyedBy:CodingKeys.self)
            try c.encode(relativePath,forKey:.relativePath);try c.encode(bytes,forKey:.bytes)
            try c.encode(`extension`,forKey:.extension);try c.encode(status,forKey:.status)
            if let capturePath {try c.encode(capturePath,forKey:.capturePath)} else {try c.encodeNil(forKey:.capturePath)}
            if let sha256 {try c.encode(sha256,forKey:.sha256)} else {try c.encodeNil(forKey:.sha256)}
            try c.encodeIfPresent(modifiedAt,forKey:.modifiedAt);try c.encodeIfPresent(createdAt,forKey:.createdAt);try c.encodeIfPresent(error,forKey:.error)
        }
    }
    let schemaVersion:Int
    let kind:String
    let sourceMode:String
    let readMode:String
    let complete:Bool
    let members:[Member]
    let selection:[String:JSONValue]
    static let selectionKeys:Set<String>=["sourceMode","extensions","includeSubfolders","includeHidden","includePatterns","excludePatterns","symlinks","readMode","unknownFiles","maxFiles","maxDepth","maxFileBytes","maxTotalBytes","errorPolicy"]
    static let maximumMemberBytes=67_108_864
    static let maximumTotalBytes=268_435_456
    static let maximumManifestBytes=2_000_000
    static func decode(_ bytes:Data)throws->Self {
        guard bytes.count<=maximumManifestBytes,let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],
              Set(object.keys)==["schemaVersion","kind","sourceMode","readMode","complete","members","selection"],
              let rows=object["members"] as? [[String:Any]],rows.count<=1000 else {throw WorkflowFileLeaseError.unsafePath}
        let required:Set<String>=["relativePath","capturePath","bytes","sha256","extension","status"]
        for row in rows {guard required.isSubset(of:Set(row.keys)),Set(row.keys).isSubset(of:required.union(["modifiedAt","createdAt","error"])) else {throw WorkflowFileLeaseError.unsafePath}}
        let value=try JSONDecoder().decode(Self.self,from:bytes)
        guard value.schemaVersion==1,value.kind=="kwipFileCollection",["file","directory"].contains(value.sourceMode),["metadata","content"].contains(value.readMode),value.sourceMode != "file" || value.members.count<=1,
              value.members.map(\.relativePath)==value.members.map(\.relativePath).sorted(),Set(value.members.map(\.relativePath)).count==value.members.count else {throw WorkflowFileLeaseError.unsafePath}
        guard Set(value.selection.keys)==selectionKeys,try FileInputConfiguration.validate(value.selection).filter({selectionKeys.contains($0.key)})==value.selection,
              value.selection["sourceMode"]?.stringValue==value.sourceMode,value.selection["readMode"]?.stringValue==value.readMode else {throw WorkflowFileLeaseError.unsafePath}
        var total:Int64=0,captures=Set<String>()
        for row in value.members {
            let parts=try WorkflowArchiveCodec.pathParts(row.relativePath)
            guard !parts.isEmpty,parts.count<=33,parts.allSatisfy({$0.utf8.count<=255}),row.relativePath.utf8.count<=4096,row.bytes>=0,
                  row.extension.utf8.count<=32,row.extension.utf8.allSatisfy({(97...122).contains($0)||(48...57).contains($0)||$0==95||$0==45}),["content","metadata","error"].contains(row.status),value.readMode != "metadata" || row.status != "content" else {throw WorkflowFileLeaseError.unsafePath}
            if row.status=="content" {
                guard let path=row.capturePath,let hash=row.sha256,WorkflowArchiveCodec.isHash(hash),row.bytes<=Int64(maximumMemberBytes),row.error==nil else {throw WorkflowFileLeaseError.unsafePath}
                let parts=try WorkflowArchiveCodec.pathParts(path)
                guard parts.count==2,parts[0]=="members",parts[1].utf8.count<=255,captures.insert(path).inserted else {throw WorkflowFileLeaseError.unsafePath}
                total+=row.bytes;guard total<=Int64(maximumTotalBytes) else {throw WorkflowFileLeaseError.unsafePath}
            } else {
                guard row.capturePath==nil,row.sha256==nil else {throw WorkflowFileLeaseError.unsafePath}
                if row.status=="error" {guard let error=row.error,!error.isEmpty,error.count<=1000,!error.contains("\0") else {throw WorkflowFileLeaseError.unsafePath}}
                else {guard row.error==nil else {throw WorkflowFileLeaseError.unsafePath}}
            }
            for time in [row.modifiedAt,row.createdAt].compactMap({$0}) {guard !time.isEmpty,time.count<=64,!time.contains("\0") else {throw WorkflowFileLeaseError.unsafePath}}
        }
        return value
    }
    func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];let data=try encoder.encode(self);_ = try Self.decode(data);return data}
    func ownedPaths(manifestPath:String,inventory:[String:WorkflowArchiveFile])throws->Set<String> {
        let parts=try WorkflowArchiveCodec.pathParts(manifestPath)
        guard parts.count==4,parts[0]=="input",parts[1]=="file-collections",WorkflowArchiveCodec.isHash(parts[2]),parts[3]=="manifest.json",inventory[manifestPath]?.sha256==parts[2] else {throw WorkflowFileLeaseError.unsafePath}
        let prefix=parts.dropLast().joined(separator:"/");var owned:Set<String>=[manifestPath]
        for member in members where member.status=="content" {
            guard let capture=member.capturePath,let hash=member.sha256 else {throw WorkflowFileLeaseError.unsafePath}
            let name=prefix+"/"+capture
            guard inventory[name]?.sha256==hash,inventory[name]?.bytes==Int(member.bytes) else {throw WorkflowFileLeaseError.unsafePath};owned.insert(name)
        }
        return owned
    }
}

/// Host-only descriptor-relative capture. Original directory authority never crosses the worker boundary.
final class FileInputCollectionCapture {
    let source:WorkflowCapturedSource
    private let lifetime:WorkflowExportStorageLifetime
    private init(source:WorkflowCapturedSource,lifetime:WorkflowExportStorageLifetime) {self.source=source;self.lifetime=lifetime}
    func remove(){lifetime.remove()}
    deinit{remove()}
    private struct Entry {let relative:String;let metadata:stat;let status:String;let error:String?}
    private static func same(_ a:stat,_ b:stat)->Bool {a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_mode==b.st_mode && a.st_size==b.st_size && a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec && a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec}
    private static func timestamp(_ value:timespec)->String {ISO8601DateFormatter().string(from:Date(timeIntervalSince1970:Double(value.tv_sec)+Double(value.tv_nsec)/1_000_000_000))}
    static func capture(path:String,configuration:[String:JSONValue],workspace:URL,accountingRoot:URL?=nil,sourcePermissions:URL,storageLimitBytes:Int64?,previewLimit:Int?=nil,beforeFinalVerification:(()throws->Void)?=nil)throws->FileInputCollectionCapture {
        let config=try FileInputConfiguration.validate(configuration)
        func integer(_ key:String,_ fallback:Int)->Int {config[key]?.integerValue ?? fallback}
        func strings(_ key:String)->[String] {guard case .array(let list)=config[key] else {return []};return list.compactMap(\.stringValue)}
        let mode=config["sourceMode"]?.stringValue ?? "file",readMode=config["readMode"]?.stringValue ?? "metadata"
        let maximumFiles=min(1000,integer("maxFiles",1000)),maximumDepth=min(32,integer("maxDepth",32)),maximumFile=min(67_108_864,integer("maxFileBytes",67_108_864)),maximumTotal=min(268_435_456,integer("maxTotalBytes",268_435_456))
        let recursive=config["includeSubfolders"]?.boolValue ?? false,hidden=config["includeHidden"]?.boolValue ?? false
        let symlinks=config["symlinks"]?.stringValue ?? "error",errors=config["errorPolicy"]?.stringValue ?? "fail",unknown=config["unknownFiles"]?.stringValue ?? "metadata"
        let extensions=Set(strings("extensions")),includes=strings("includePatterns"),excludes=strings("excludePatterns")
        let audioEnabled:Bool
        if case .object(let readers)?=config["readers"],case .object(let audio)?=readers["audio"] {audioEnabled=audio["format"] == .string("pcmWav")} else {audioEnabled=false}
        let known=Set(FileInputConfiguration.allExtensions.filter {audioEnabled || !(FileInputConfiguration.extensionGroups["audio"] ?? []).contains($0)})
        return try SourceAccess(directory:sourcePermissions).withAccess(to:path) { selected in
            try Task.checkCancellation()
            let fd=Darwin.open(selected.path,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC|(mode=="directory" ? O_DIRECTORY:0))
            guard fd>=0 else {throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(fd)}
            var rootInfo=stat();guard fstat(fd,&rootInfo)==0,(rootInfo.st_mode & mode_t(S_IFMT)) == mode_t(mode=="directory" ? S_IFDIR:S_IFREG) else {throw WorkflowFileLeaseError.unsafePath}
            var entries:[Entry]=[],directories:[String:stat]=[:],enumerated=0,total=0,matchedFiles=0
            func matches(_ patterns:[String],_ name:String)->Bool {patterns.contains {pattern in
                func match(_ value:String)->Bool {value.withCString {p in name.withCString {n in fnmatch(p,n,0)==0}}}
                return match(pattern) || (pattern.hasPrefix("**/") && match(String(pattern.dropFirst(3))))
            }}
            func admit(_ name:String,_ info:stat)throws {
                let ext=URL(fileURLWithPath:name).pathExtension.lowercased()
                if mode=="directory" && !extensions.isEmpty && !extensions.contains(ext) {return}
                if !includes.isEmpty && !matches(includes,name) {return};if matches(excludes,name) {return}
                matchedFiles+=1;guard matchedFiles<=maximumFiles else {throw WorkflowArchiveError("File Input exceeds its selected-file count limit.")}
                var status=readMode=="content" ? "content":"metadata",error:String?
                if !known.contains(ext) {
                    if unknown=="skip" {return}
                    if unknown=="error" {
                        if errors=="skip" {return}
                        if errors=="record" {status="error";error="Unsupported File Input extension: "+ext}
                        else {throw WorkflowArchiveError("Unsupported File Input extension: "+ext)}
                    } else {status="metadata"}
                }
                if previewLimit != nil,status=="content",(FileInputConfiguration.extensionGroups["image"]?.contains(ext)==true || FileInputConfiguration.extensionGroups["audio"]?.contains(ext)==true) {status="metadata"}
                if status=="content" && (info.st_size<0 || info.st_size>Int64(maximumFile) || Int64(total)+Int64(info.st_size)>Int64(maximumTotal)) {
                    if errors=="skip" {return}
                    if errors=="record" {status="error";error="File content exceeds the selected capture byte limit."}
                    else {throw WorkflowArchiveError("File Input exceeds its content capture byte limit.")}
                }
                if status=="content" {total+=Int(info.st_size)}
                entries.append(.init(relative:name,metadata:info,status:status,error:error))
            }
            func walk(_ directoryFD:Int32,_ prefix:String,_ depth:Int)throws {
                try Task.checkCancellation();var before=stat();guard fstat(directoryFD,&before)==0 else {throw WorkflowFileLeaseError.unsafePath};directories[prefix]=before
                let copy=dup(directoryFD);guard copy>=0,let iterator=fdopendir(copy) else {if copy>=0{Darwin.close(copy)};throw WorkflowFileLeaseError.unsafePath};defer{closedir(iterator)}
                var names:[String]=[]
                while true {
                    try Task.checkCancellation();errno=0
                    guard let entry=readdir(iterator) else {guard errno==0 else {throw WorkflowFileLeaseError.system(errno)};break}
                    let name=withUnsafePointer(to:&entry.pointee.d_name){$0.withMemoryRebound(to:CChar.self,capacity:Int(MAXNAMLEN)+1){String(cString:$0)}}
                    if name=="." || name==".." {continue};enumerated+=1
                    guard enumerated<=100_000 else {throw WorkflowArchiveError("File Input directory scan exceeds 100000 entries.")}
                    guard !name.contains("/"),!name.contains("\\"),!name.contains("\0"),name.utf8.count<=255 else {throw WorkflowFileLeaseError.unsafePath}
                    if !hidden && name.hasPrefix(".") {continue};names.append(name)
                }
                for name in names.sorted() {
                    if let previewLimit,entries.count>=previewLimit {break}
                    try Task.checkCancellation();var info=stat();guard fstatat(directoryFD,name,&info,AT_SYMLINK_NOFOLLOW)==0 else {throw WorkflowFileLeaseError.unsafePath}
                    let relative=prefix.isEmpty ? name:prefix+"/"+name;guard relative.utf8.count<=4096 else {throw WorkflowFileLeaseError.unsafePath}
                    switch info.st_mode & mode_t(S_IFMT) {
                    case mode_t(S_IFLNK):if symlinks=="error" {throw WorkflowArchiveError("File Input does not follow symbolic links.")}
                    case mode_t(S_IFDIR):
                        if recursive {guard depth<maximumDepth else {throw WorkflowArchiveError("File Input exceeds its directory depth limit.")};let child=openat(directoryFD,name,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);guard child>=0 else {throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(child)};try walk(child,relative,depth+1)}
                    case mode_t(S_IFREG):try admit(relative,info)
                    default:throw WorkflowArchiveError("File Input accepts regular files only; devices, sockets and pipes are not supported.")
                    }
                }
                var after=stat();guard fstat(directoryFD,&after)==0,same(before,after) else {throw WorkflowArchiveError("Directory changed during File Input capture. Refresh the source.")}
            }
            if mode=="directory" {try walk(fd,"",0)} else {try admit(selected.lastPathComponent,rootInfo)}
            entries.sort {$0.relative<$1.relative}
            let complete=previewLimit==nil || mode=="file"
            if let previewLimit {guard (0...8).contains(previewLimit) else {throw WorkflowFileLeaseError.unsafePath};entries=Array(entries.prefix(previewLimit))}
            let payload=entries.filter {$0.status=="content"}.reduce(0){$0+Int($1.metadata.st_size)}
            let manifestAllowance=min(2_000_000,entries.count*5000+4096)
            guard storageLimitBytes==nil || Int64(payload+manifestAllowance)<=storageLimitBytes! else {throw WorkflowArchiveError("File Input capture exceeds the effective storage budget.")}
            let stageID=UUID(),reservation=try storageLimitBytes.map {try WorkflowStorageAdmission(workspace:workspace,accountingRoot:accountingRoot).reserve(scopes:[.init(namespace:.workflowSourceStaging,components:[stageID.uuidString])],peakBytes:Int64(max(1,payload+manifestAllowance)),limitBytes:$0)}
            var transferred=false;defer{if !transferred{reservation?.close()}}
            let rootFD=try WorkflowEvaluationOwnership.openDirectory(workspace);defer{Darwin.close(rootFD)}
            if mkdirat(rootFD,"workflow-source-staging",0o700) != 0,errno != EEXIST {throw WorkflowFileLeaseError.system(errno)}
            let parentFD=openat(rootFD,"workflow-source-staging",O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard parentFD>=0 else {throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(parentFD)}
            guard mkdirat(parentFD,stageID.uuidString,0o700)==0 else {throw WorkflowFileLeaseError.system(errno)}
            let directory=workspace.appendingPathComponent("workflow-source-staging").appendingPathComponent(stageID.uuidString)
            let stageFD=openat(parentFD,stageID.uuidString,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard stageFD>=0 else {throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(stageFD)}
            let stageIdentity=try WorkflowEvaluationOwnership.directoryIdentity(stageFD)
            func verifyStage()throws {let current=try WorkflowEvaluationOwnership.openDirectory(directory);defer{Darwin.close(current)};guard try WorkflowEvaluationOwnership.directoryIdentity(current)==stageIdentity else {throw WorkflowFileLeaseError.unsafePath}}
            let lifetime=try WorkflowExportStorageLifetime(directory:directory,reservation:reservation);transferred=true
            var retained=false;defer{if !retained{lifetime.remove()}}
            func reopen(_ relative:String)->Int32 {
                if mode=="file" {return dup(fd)}
                let parts=relative.split(separator:"/").map(String.init);var current=dup(fd);if current<0{return -1}
                for (index,name) in parts.enumerated() {let next=openat(current,name,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC|(index==parts.count-1 ? 0:O_DIRECTORY));Darwin.close(current);current=next;if current<0{return -1}}
                return current
            }
            var members:[FileInputCollectionManifest.Member]=[],captures:[FileInputCapturedMember]=[]
            for (index,entry) in entries.enumerated() {
                try Task.checkCancellation();let ext=URL(fileURLWithPath:entry.relative).pathExtension.lowercased()
                var status=entry.status,captureError=entry.error,capturePath:String?,hash:String?
                if status=="content" {
                    var copied: URL?
                    do {
                        let file=reopen(entry.relative);guard file>=0 else {throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(file)}
                        var before=stat();guard fstat(file,&before)==0,same(entry.metadata,before),before.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG) else {throw WorkflowFileLeaseError.unsafePath}
                        let suffix=ext.isEmpty ? "data":ext,name=String(format:"%06d",index)+"."+suffix
                        try verifyStage()
                        if mkdirat(stageFD,"members",0o700) != 0,errno != EEXIST {throw WorkflowFileLeaseError.system(errno)}
                        let receipt=try DiagnosticFileIO.copyRegularDescriptor(file,root:directory,parts:["members",name],limit:maximumFile)
                        copied=directory.appendingPathComponent(receipt.path)
                        var after=stat();guard fstat(file,&after)==0,same(before,after),receipt.bytes==Int(before.st_size) else {throw WorkflowArchiveError("Selected file changed during capture.")}
                        let digest=receipt.sha256;capturePath=receipt.path;hash=digest
                        captures.append(.init(capturePath:receipt.path,url:directory.appendingPathComponent(receipt.path),sha256:digest,byteCount:receipt.bytes))
                        copied=nil
                    } catch is CancellationError {if let copied {try? FileManager.default.removeItem(at:copied)};throw CancellationError()}
                    catch {if let copied {try? FileManager.default.removeItem(at:copied)};if errors=="skip"{continue};if errors != "record"{throw error};status="error";captureError="Selected file could not be captured safely."}
                }
                members.append(.init(relativePath:entry.relative,capturePath:capturePath,bytes:Int64(entry.metadata.st_size),sha256:hash,extension:ext,status:status,modifiedAt:timestamp(entry.metadata.st_mtimespec),createdAt:timestamp(entry.metadata.st_birthtimespec),error:captureError))
            }
            try beforeFinalVerification?()
            // Recheck every successful selected file, not only directory entries.
            // Content edits do not update the containing directory's timestamps.
            let successful=Set(members.filter {$0.status != "error"}.map(\.relativePath))
            for entry in entries where successful.contains(entry.relative) {
                try Task.checkCancellation()
                let file=reopen(entry.relative);guard file>=0 else {throw WorkflowFileLeaseError.unsafePath};var actual=stat()
                let valid=fstat(file,&actual)==0 && same(entry.metadata,actual) && actual.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG)
                Darwin.close(file)
                guard valid else {throw WorkflowArchiveError("Selected file changed before collection verification. Refresh the source.")}
            }
            // A renamed/replaced root or changed census invalidates the capture; no partial authority is issued.
            let current=Darwin.open(selected.path,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC|(mode=="directory" ? O_DIRECTORY:0));guard current>=0 else {throw WorkflowFileLeaseError.unsafePath};defer{Darwin.close(current)}
            var currentInfo=stat();guard fstat(current,&currentInfo)==0,same(rootInfo,currentInfo) else {throw WorkflowArchiveError("File Input source changed during capture.")}
            for (relative,expected) in directories {
                let check=relative.isEmpty ? dup(fd):reopen(relative);guard check>=0 else {throw WorkflowFileLeaseError.unsafePath};var actual=stat();let valid=fstat(check,&actual)==0 && same(expected,actual);Darwin.close(check);guard valid else {throw WorkflowArchiveError("Directory changed during capture.")}
            }
            let manifest=FileInputCollectionManifest(schemaVersion:1,kind:"kwipFileCollection",sourceMode:mode,readMode:readMode,complete:complete,members:members,selection:config.filter {FileInputCollectionManifest.selectionKeys.contains($0.key)})
            try verifyStage()
            let data=try manifest.encoded();try DiagnosticFileIO.writeImmutable(data,root:directory,parts:["manifest.json"],limit:2_000_000)
            let source=WorkflowCapturedSource(url:directory.appendingPathComponent("manifest.json"),sha256:WorkflowArchiveCodec.hash(data),byteCount:data.count,collectionMembers:captures,isFileCollection:true)
            retained=true;return .init(source:source,lifetime:lifetime)
        }
    }
}
