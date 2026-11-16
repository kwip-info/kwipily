import Foundation
import Darwin
import KwipMLCore

/// Publishes exactly the reviewed bytes in one fresh directory. Shared by trained
/// model and frozen evaluation exports; neither may merge into an existing export.
enum WorkflowExportPublication {
    static func destination(in parent:URL,folderName:String)throws->URL {
        guard !folderName.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
              folderName != ".",folderName != "..",folderName.utf8.count<=128,
              !folderName.contains("/"),!folderName.unicodeScalars.contains(where:{CharacterSet.controlCharacters.contains($0)}) else {
            throw WorkflowArchiveError("Enter a folder name of up to 128 bytes without path separators or control characters.")
        }
        return parent.appendingPathComponent(folderName,isDirectory:true)
    }
    static func publish(directory:URL,files:[WorkflowArchiveFile],manifestName:String,manifestBytes:Data,to destination:URL)throws {
        try Task.checkCancellation()
        let manager=FileManager.default
        guard !manager.fileExists(atPath:destination.path) else {throw WorkflowArchiveError("Choose a new export folder; existing contents will not be replaced.")}
        let temporary=destination.deletingLastPathComponent().appendingPathComponent(".kwipml-export-"+UUID().uuidString)
        try manager.createDirectory(at:temporary,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        defer {try? manager.removeItem(at:temporary)}
        let manifest=try DiagnosticFileIO.read(root:directory,parts:[manifestName],limit:2_000_000)
        guard manifest==manifestBytes else {throw WorkflowArchiveError("Export manifest changed after preparation.")}
        for file in files {
            try Task.checkCancellation()
            let bytes=try DiagnosticFileIO.read(root:directory,parts:[file.path],limit:file.bytes)
            guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256 else {throw WorkflowArchiveError("Export artifact changed after preparation: \(file.path).")}
            try bytes.write(to:temporary.appendingPathComponent(file.path),options:.withoutOverwriting)
        }
        try manifest.write(to:temporary.appendingPathComponent(manifestName),options:.withoutOverwriting)
        try Task.checkCancellation()
        // Successful rename is the publication commit point.
        try manager.moveItem(at:temporary,to:destination)
    }
}


extension WorkflowExportPublication {
    static func validateExclusiveLeaf(_ name:String)throws {
        guard !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name != ".",name != "..",name.utf8.count<=128,
              !name.contains("/"),!name.contains("\\"),!name.unicodeScalars.contains(where:{CharacterSet.controlCharacters.contains($0)}) else {
            throw WorkflowArchiveError("Choose a single new model filename or folder name of up to 128 bytes.")
        }
    }
    static func requireAbsent(_ name:String,parent:Int32)throws {
        try validateExclusiveLeaf(name)
        var entry=stat()
        if fstatat(parent,name,&entry,AT_SYMLINK_NOFOLLOW)==0 {throw WorkflowArchiveError("Choose a new destination; existing files, folders and symbolic links are never replaced.")}
        guard errno==ENOENT else {throw WorkflowFileLeaseError.system(errno)}
    }
    static func canonicalParentPath(_ url:URL)throws->String {
        guard url.isFileURL,url.host == nil || url.host == "" || url.host == "localhost",url.path.hasPrefix("/"),url.path.utf8.count<=4096,!url.path.contains("\0"),let pointer=realpath(url.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(pointer)}
        let path=String(cString:pointer)
        var expected=url.path.count>1 && url.path.hasSuffix("/") ? String(url.path.dropLast()):url.path
        for alias in ["/var","/tmp","/etc"] where expected==alias || expected.hasPrefix(alias+"/") {expected="/private"+expected;break}
        guard path==expected else {throw WorkflowFileLeaseError.unsafePath}
        return path
    }
    /// Stage exact reviewed package bytes using the same export inventory contract.
    /// Commit is separate so the MainActor can revalidate its source lease immediately before rename.
    static func stageExclusive(_ package:WorkflowTrainedExportPackage,to destination:URL)throws->WorkflowPendingExportPublication {
        try Task.checkCancellation()
        guard package.files.count<=128,Set(package.files.map(\.path)).count==package.files.count,
              !package.files.contains(where:{$0.path=="export-manifest.json"}),package.manifestBytes.count<=2_000_000 else {throw WorkflowArchiveError("Invalid trained export inventory.")}
        let source=try WorkflowFrozenSnapshotDirectory(directory:package.directory,removeOnDeinit:false)
        try source.check()
        let manifest=try DiagnosticFileIO.read(root:package.directory,parts:["export-manifest.json"],limit:2_000_000)
        guard manifest==package.manifestBytes else {throw WorkflowArchiveError("Export manifest changed after preparation.")}
        let pending=try WorkflowPendingExportPublication(destination:destination,directory:true)
        do {
            var total=manifest.count
            for file in package.files {
                try validateExclusiveLeaf(file.path)
                guard (0...100_000_000).contains(file.bytes),WorkflowArchiveCodec.isHash(file.sha256) else {throw WorkflowArchiveError("Export file exceeds its bounded inventory.")}
                total+=file.bytes;guard total<=200_000_000 else {throw WorkflowArchiveError("Model export exceeds its 200 MB publication allowance.")}
                try Task.checkCancellation();try source.check()
                let bytes=try DiagnosticFileIO.read(root:package.directory,parts:[file.path],limit:file.bytes)
                guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256 else {throw WorkflowArchiveError("Export artifact changed after preparation: \(file.path).")}
                try pending.write(name:file.path,bytes:bytes)
            }
            try pending.write(name:"export-manifest.json",bytes:manifest);try source.check();return pending
        } catch {pending.close();throw error}
    }
    static func stageExclusiveCheckpoint(_ bytes:Data,sha256:String,to destination:URL)throws->WorkflowPendingExportPublication {
        guard !bytes.isEmpty,bytes.count<=100_000_000,WorkflowArchiveCodec.isHash(sha256),WorkflowArchiveCodec.hash(bytes)==sha256,
              destination.lastPathComponent.hasSuffix(".pt") else {throw WorkflowArchiveError("Save a verified bounded checkpoint to a new .pt filename.")}
        let pending=try WorkflowPendingExportPublication(destination:destination,directory:false)
        do {try pending.write(name:nil,bytes:bytes);return pending}
        catch {pending.close();throw error}
    }
}

/// Descriptor-anchored, exclusive publication shared by Save model package/checkpoint paths.
/// Only a successful final rename publishes; cleanup never touches the accepted destination.
final class WorkflowPendingExportPublication {
    let destination:URL
    private let parent:URL
    private let leaf:String
    private let temporary=".kwipml-export-"+UUID().uuidString
    private let directory:Bool
    private let parentIdentity:WorkflowFileIdentity
    private var parentFD:Int32
    private var directoryFD:Int32 = -1
    private var temporaryIdentity:WorkflowFileIdentity?
    private var children:[String]=[]
    private var modelGroups:[(name:String,fd:Int32,files:[String])]=[]
    private var committed=false
    private let lock=NSLock()
    init(destination:URL,directory:Bool)throws {
        guard destination.isFileURL else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowExportPublication.validateExclusiveLeaf(destination.lastPathComponent)
        let canonical=try WorkflowExportPublication.canonicalParentPath(destination.deletingLastPathComponent())
        let selectedParent=URL(fileURLWithPath:canonical,isDirectory:true)
        self.parent=selectedParent;self.destination=selectedParent.appendingPathComponent(destination.lastPathComponent,isDirectory:directory)
        self.leaf=destination.lastPathComponent;self.directory=directory
        let fd=try WorkflowEvaluationOwnership.openDirectory(selectedParent)
        do {
            parentIdentity=try WorkflowEvaluationOwnership.directoryIdentity(fd)
            try WorkflowExportPublication.requireAbsent(leaf,parent:fd)
        } catch {Darwin.close(fd);throw error}
        parentFD=fd
        do {
            if directory {
                guard mkdirat(fd,temporary,0o700)==0 else {throw WorkflowFileLeaseError.system(errno)}
                var created=stat()
                guard fstatat(fd,temporary,&created,AT_SYMLINK_NOFOLLOW)==0 else {throw WorkflowFileLeaseError.system(errno)}
                temporaryIdentity=WorkflowEvaluationOwnership.identity(created)
                directoryFD=openat(fd,temporary,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
                guard directoryFD>=0 else {throw WorkflowFileLeaseError.unsafePath}
                temporaryIdentity=try WorkflowEvaluationOwnership.directoryIdentity(directoryFD)
            }
        } catch {close();throw error}
    }
    private func check()throws {
        guard parentFD>=0,!committed else {throw WorkflowFileLeaseError.closed}
        let current=try WorkflowEvaluationOwnership.openDirectory(parent);defer {Darwin.close(current)}
        guard try WorkflowEvaluationOwnership.directoryIdentity(current)==parentIdentity else {throw WorkflowFileLeaseError.unsafePath}
        if let expected=temporaryIdentity {
            var entry=stat()
            guard fstatat(parentFD,temporary,&entry,AT_SYMLINK_NOFOLLOW)==0,WorkflowEvaluationOwnership.identity(entry)==expected,
                  entry.st_mode & mode_t(S_IFMT)==mode_t(directory ? S_IFDIR:S_IFREG) else {throw WorkflowFileLeaseError.unsafePath}
        }
    }
    fileprivate func write(name:String?,bytes:Data)throws {
        lock.lock();defer {lock.unlock()}
        try Task.checkCancellation();try check()
        let fileName:String,fd:Int32
        if directory {
            guard let name,!children.contains(name) else {throw WorkflowFileLeaseError.unsafePath}
            try WorkflowExportPublication.validateExclusiveLeaf(name);fileName=name;fd=directoryFD
        } else {guard name==nil,temporaryIdentity==nil else {throw WorkflowFileLeaseError.unsafePath};fileName=temporary;fd=parentFD}
        let output=openat(fd,fileName,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)
        guard output>=0 else {throw WorkflowFileLeaseError.system(errno)}
        defer {Darwin.close(output)}
        if directory {children.append(fileName)} else {var info=stat();guard fstat(output,&info)==0 else {throw WorkflowFileLeaseError.system(errno)};temporaryIdentity=WorkflowEvaluationOwnership.identity(info)}
        try bytes.withUnsafeBytes {raw in
            var offset=0
            while offset<raw.count {
                try Task.checkCancellation();try check()
                let count=Darwin.write(output,raw.baseAddress!.advanced(by:offset),min(65536,raw.count-offset))
                if count<0,errno==EINTR {continue}
                guard count>0 else {throw WorkflowFileLeaseError.system(errno)};offset+=count
            }
        }
        guard fsync(output)==0 else {throw WorkflowFileLeaseError.system(errno)}
        try check()
    }
    /// Exact inert classifier package inventory from an accepted backend operation.
    func writeClassifierFile(_ file:WorkflowArchiveFile,from source:URL)throws {
        guard ["state.json","preparation.json","document_classification.py","prebuilt_models.py","prebuilt-models.json","inference.py","README.txt","manifest.json"].contains(file.path),file.bytes>0,file.bytes <= (file.path == "preparation.json" ? PrebuiltPreparationAttachment.maximumBytes:32_000_000) else {throw WorkflowFileLeaseError.unsafePath}
        let bytes=try DiagnosticFileIO.read(root:source,parts:[file.path],limit:file.bytes)
        guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256 else {throw WorkflowArchiveError("Classifier package changed before publication.")}
        try write(name:file.path,bytes:bytes)
    }
    func writeModelFile(_ file:WorkflowArchiveFile,from source:URL)throws {
        lock.lock();defer {lock.unlock()};try check()
        if ModelHubRequest.nestedModelFiles.contains(file.path) {
            guard directory,file.bytes>0,file.bytes<=8_000_000 else {throw WorkflowFileLeaseError.unsafePath}
            let parts=file.path.split(separator:"/").map(String.init)
            let index:Int
            if let existing=modelGroups.firstIndex(where:{$0.name==parts[0]}) {index=existing}
            else {
                guard mkdirat(directoryFD,parts[0],0o700)==0 else {throw WorkflowFileLeaseError.system(errno)}
                let fd=openat(directoryFD,parts[0],O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
                guard fd>=0 else {throw WorkflowFileLeaseError.unsafePath}
                modelGroups.append((parts[0],fd,[]));index=modelGroups.count-1
            }
            guard !modelGroups[index].files.contains(parts[1]) else {throw WorkflowFileLeaseError.unsafePath}
            let sourceFD=try WorkflowEvaluationOwnership.openDirectory(source.appendingPathComponent(parts[0]));defer {Darwin.close(sourceFD)}
            modelGroups[index].files.append(parts[1])
            let result=try ModelAssetFiles.stream(parts[1],parent:sourceFD,destination:modelGroups[index].fd,maximum:file.bytes)
            guard result.bytes==file.bytes,result.sha256==file.sha256 else {throw WorkflowArchiveError("Nested model export changed before publication.")}
            guard fsync(modelGroups[index].fd)==0 else {throw WorkflowFileLeaseError.system(errno)}
            try check();return
        }
        guard directory,ModelHubRequest.allowedFile(file.path) || ModelTaskArtifactNames.continuationTensor(file.path) || ["kwip-manifest.json","adapter_config.json","adapter_model.safetensors","kwip-adapter.json","kwip-continuation.json","training-state.safetensors","training.json","result-manifest.json"].contains(file.path),!file.path.contains("/"),file.bytes>0,file.bytes<=2_000_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowExportPublication.validateExclusiveLeaf(file.path)
        let sourceFD=try WorkflowEvaluationOwnership.openDirectory(source);defer {Darwin.close(sourceFD)}
        children.append(file.path)
        guard try ModelAssetFiles.stream(file.path,parent:sourceFD,destination:directoryFD,maximum:file.bytes)==file else {throw WorkflowArchiveError("Model export data changed before publication.")}
        try check()
    }
    func writeResumeMetadata(name:String,bytes:Data)throws {
        guard ["training.json","result-manifest.json"].contains(name),bytes.count<=8_000_000 else{throw WorkflowFileLeaseError.unsafePath}
        try write(name:name,bytes:bytes)
    }
    func writeModelGroup(_ name:String,files:[WorkflowArchiveFile],from source:URL)throws {
        lock.lock();defer{lock.unlock()};try check()
        guard directory,["base","latest"].contains(name),!modelGroups.contains(where:{$0.name==name}),!files.isEmpty,files.count<=128 else{throw WorkflowFileLeaseError.unsafePath}
        guard mkdirat(directoryFD,name,0o700)==0 else{throw WorkflowFileLeaseError.system(errno)}
        let fd=openat(directoryFD,name,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd>=0 else{throw WorkflowFileLeaseError.unsafePath}
        modelGroups.append((name,fd,[]))
        let sourceFD=try WorkflowEvaluationOwnership.openDirectory(source);defer{Darwin.close(sourceFD)}
        for file in files {
            try Task.checkCancellation();guard ModelFineTuningFiles.leaf(file.path),file.bytes>0,file.bytes<=2_000_000_000 else{throw WorkflowFileLeaseError.unsafePath}
            modelGroups[modelGroups.count-1].files.append(file.path)
            guard try ModelAssetFiles.stream(file.path,parent:sourceFD,destination:fd,maximum:file.bytes)==file else{throw WorkflowArchiveError("Resume package file changed while copying.")}
        }
        guard fsync(fd)==0 else{throw WorkflowFileLeaseError.system(errno)}
    }
    func commit()throws {
        lock.lock();defer {lock.unlock()}
        try Task.checkCancellation();try check()
        if directory {guard fsync(directoryFD)==0 else {throw WorkflowFileLeaseError.system(errno)}}
        try WorkflowExportPublication.requireAbsent(leaf,parent:parentFD)
        guard renameatx_np(parentFD,temporary,parentFD,leaf,UInt32(RENAME_EXCL))==0 else {throw WorkflowFileLeaseError.system(errno)}
        committed=true
        // Rename is accepted; late cancellation or a metadata flush cannot undo it.
        _ = fsync(parentFD)
    }
    func close() {
        lock.lock();defer {lock.unlock()}
        guard parentFD>=0 else{return}
        if !committed,let expected=temporaryIdentity {
            var entry=stat()
            if fstatat(parentFD,temporary,&entry,AT_SYMLINK_NOFOLLOW)==0,WorkflowEvaluationOwnership.identity(entry)==expected {
                if directory, directoryFD>=0 {for group in modelGroups {for leaf in group.files{unlinkat(group.fd,leaf,0)};unlinkat(directoryFD,group.name,AT_REMOVEDIR)};for child in children {unlinkat(directoryFD,child,0)};unlinkat(parentFD,temporary,AT_REMOVEDIR)}
                else {unlinkat(parentFD,temporary,0)}
            }
        }
        for group in modelGroups{Darwin.close(group.fd)};modelGroups=[]
        if directoryFD>=0 {Darwin.close(directoryFD);directoryFD = -1}
        Darwin.close(parentFD);parentFD = -1
    }
    deinit {close()}
}

extension WorkflowExportPublication {
    /// Generation reuses exclusive, descriptor-anchored publication with a flat bounded inventory.
    static func stageGeneration(_ files:[String:Data],to destination:URL)throws->WorkflowPendingExportPublication {
        guard !files.isEmpty,files.count<=32,files.values.reduce(0,{$0+$1.count})<=200_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        let pending=try WorkflowPendingExportPublication(destination:destination,directory:true)
        do {for (name,bytes) in files {try validateExclusiveLeaf(name);try pending.write(name:name,bytes:bytes)};return pending}
        catch {pending.close();throw error}
    }
}

extension WorkflowExportPublication {
    static func stageSelfPlay(record:SelfPlayRecord,root:URL,to destination:URL)throws->WorkflowPendingExportPublication {
        _=try record.verify(root:root);let directory=try record.directory(root)
        let pending=try WorkflowPendingExportPublication(destination:destination,directory:true)
        do {
            for name in ["model.pt","model.py","model.json","inference-manifest.json"] {
                try Task.checkCancellation()
                guard let f=record.files.first(where:{$0.path=="artifacts/"+name})else{throw WorkflowFileLeaseError.unsafePath}
                let raw=try DiagnosticFileIO.read(root:directory,parts:["artifacts",name],limit:name=="model.pt" ? 100_000_000:2_000_000)
                guard WorkflowArchiveCodec.hash(raw)==f.sha256 else{throw WorkflowFileLeaseError.unsafePath};try pending.write(name:name,bytes:raw)
            };return pending
        }catch{pending.close();throw error}
    }
}
