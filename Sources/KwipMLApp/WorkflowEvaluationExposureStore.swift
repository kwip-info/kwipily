import Foundation
import Darwin
import KwipMLCore

/// Local disclosure accounting, outside deletable evaluation/project directories.
/// A readable journal establishes only recorded history since tracking began.
final class WorkflowEvaluationExposureStore {
    struct Population:Equatable {
        let checkpointSHA256:String
        let sourceSHA256:String
        let membershipSHA256:String
        fileprivate let rowHashes:[String]
        init(checkpointSHA256:String,sourceSHA256:String,observationIDs:[String])throws {
            guard WorkflowArchiveCodec.isHash(checkpointSHA256),WorkflowArchiveCodec.isHash(sourceSHA256),
                  (1...100000).contains(observationIDs.count),Set(observationIDs).count==observationIDs.count,
                  observationIDs.allSatisfy({!$0.isEmpty && $0.unicodeScalars.count<=200}) else {throw Failure("Invalid bounded exposure population.")}
            self.checkpointSHA256=checkpointSHA256;self.sourceSHA256=sourceSHA256
            let ordered=observationIDs.map {Data($0.utf8)}.sorted {$0.lexicographicallyPrecedes($1)}
            var membership=Data("kwip.membership.v1\n".utf8)
            for value in ordered {membership.append(Data("\(value.count):".utf8));membership.append(value)}
            membershipSHA256=WorkflowArchiveCodec.hash(membership)
            rowHashes=ordered.map {value in
                var bytes=Data(("kwip.exposure.row.v1\n"+sourceSHA256+"\n\(value.count):").utf8);bytes.append(value)
                return WorkflowArchiveCodec.hash(bytes)
            }.sorted()
        }
    }
    struct Event {
        enum Kind:String,Codable {case evaluated,inspectionRecorded}
        let kind:Kind
        let evaluationID:String
        let projectID:String
        let manifestSHA256:String
        let snapshotReceiptSHA256:String
        let population:Population
    }
    struct Report {
        let priorHistoryUnknown:Bool
        let evaluatedCount:Int
        let inspectionCount:Int
        let checkpointInspectionCount:Int
        let sameSourceInspectedOverlap:Int
        /// Completeness refers to recorded sources only, never lifetime certainty.
        let overlapIsComplete:Bool
    }
    enum History {case unknown(String),tracked(Report)}
    private struct Failure:LocalizedError {
        let message:String
        init(_ message:String) {self.message=message}
        var errorDescription:String? {message}
    }
    private struct Origin:Codable {let schemaVersion:Int;let trackerID:String;let priorHistoryUnknown:Bool;let startedAt:Double}
    private struct Entry:Codable {let file:String;let sha256:String}
    private struct Index:Codable {let schemaVersion:Int;let originSHA256:String;var events:[Entry]}
    private struct SavedPopulation:Codable,Equatable {let sourceSHA256:String;let membershipSHA256:String;let rowHashes:[String]}
    private struct SavedEvent:Codable,Equatable {
        let schemaVersion:Int
        let kind:Event.Kind
        let evaluationID:String
        let projectID:String
        let manifestSHA256:String
        let snapshotReceiptSHA256:String
        let checkpointSHA256:String
        let populationSHA256:String
        let recordedAt:Double
    }
    private struct Loaded {let index:Index;let events:[SavedEvent];let populations:[String:SavedPopulation]}
    private let root:URL
    private let namespace="workflow-evaluation-exposure"
    private let maximumBytes=256_000_000
    init(root:URL) {self.root=root}

    func startTracking()throws {
        try access(create:true) {fd,created in
            let names=try names(fd)
            if created && names == ["owner.lock"] {
                let origin=try encode(Origin(schemaVersion:1,trackerID:UUID().uuidString,priorHistoryUnknown:true,startedAt:Date().timeIntervalSince1970))
                try publish(origin,name:"origin.json",fd:fd)
                try publish(encode(Index(schemaVersion:1,originSHA256:WorkflowArchiveCodec.hash(origin),events:[])),name:"index.json",fd:fd)
            } else {_ = try load(fd)}
        }
    }
    func record(_ event:Event)throws {
        try access(create:false) {fd,_ in
            try append(event,loaded:load(fd),fd:fd)
        }
    }
    /// Reads prior exposure and durably appends inspection under one exclusive
    /// lease. A replay returns current history without adding another event.
    func recordInspectionReturningPrior(_ event:Event)throws->Report {
        guard event.kind == .inspectionRecorded else {throw Failure("An inspection transaction requires an inspection event.")}
        return try access(create:false) {fd,_ in
            let loaded=try load(fd)
            let prior=try report(for:event.population,loaded:loaded)
            try append(event,loaded:loaded,fd:fd)
            return prior
        }
    }
    private func append(_ event:Event,loaded:Loaded,fd:Int32)throws {
        guard UUID(uuidString:event.evaluationID)?.uuidString==event.evaluationID,UUID(uuidString:event.projectID)?.uuidString==event.projectID,
              WorkflowArchiveCodec.isHash(event.manifestSHA256),WorkflowArchiveCodec.isHash(event.snapshotReceiptSHA256) else {throw Failure("Invalid exposure event identity.")}
        let population=SavedPopulation(sourceSHA256:event.population.sourceSHA256,membershipSHA256:event.population.membershipSHA256,rowHashes:event.population.rowHashes)
        let populationBytes=try encode(population),populationPin=WorkflowArchiveCodec.hash(populationBytes)
        let proposed=SavedEvent(schemaVersion:1,kind:event.kind,evaluationID:event.evaluationID,projectID:event.projectID,
            manifestSHA256:event.manifestSHA256,snapshotReceiptSHA256:event.snapshotReceiptSHA256,
            checkpointSHA256:event.population.checkpointSHA256,populationSHA256:populationPin,recordedAt:Date().timeIntervalSince1970)
        for prior in loaded.events where prior.evaluationID==event.evaluationID {
            guard prior.projectID==proposed.projectID,prior.manifestSHA256==proposed.manifestSHA256,
                  prior.snapshotReceiptSHA256==proposed.snapshotReceiptSHA256,prior.checkpointSHA256==proposed.checkpointSHA256,
                  prior.populationSHA256==proposed.populationSHA256 else {throw Failure("Evaluation exposure identity conflicts across event kinds.")}
        }
        if let prior=loaded.events.first(where:{$0.evaluationID==event.evaluationID && $0.kind==event.kind}) {
            guard prior.projectID==proposed.projectID,prior.manifestSHA256==proposed.manifestSHA256,
                  prior.snapshotReceiptSHA256==proposed.snapshotReceiptSHA256,prior.checkpointSHA256==proposed.checkpointSHA256,
                  prior.populationSHA256==proposed.populationSHA256 else {throw Failure("Conflicting exposure event; original record preserved.")}
            return
        }
        guard loaded.index.events.count<10000 else {throw Failure("Exposure event limit reached. History remains preserved; disclosure is blocked.")}
        let bytes=try encode(proposed),name="event-"+UUID().uuidString+".json"
        var index=loaded.index;index.events.append(.init(file:name,sha256:WorkflowArchiveCodec.hash(bytes)))
        let indexBytes=try encode(index)
        let existing=try totalBytes(fd)
        guard existing+bytes.count+indexBytes.count+(loaded.populations[populationPin] == nil ? populationBytes.count:0)<=maximumBytes else {throw Failure("Exposure storage limit reached.")}
        if loaded.populations[populationPin] == nil {try publish(populationBytes,name:"population-"+populationPin+".json",fd:fd)}
        try publish(bytes,name:name,fd:fd)
        try replaceIndex(indexBytes,fd:fd)
    }
    func history(for population:Population)->History {
        do {
            return try access(create:false) {fd,_ in
                .tracked(try report(for:population,loaded:load(fd)))
            }
        } catch {return .unknown("Exposure history is unavailable: "+error.localizedDescription)}
    }
    private func report(for population:Population,loaded:Loaded)throws->Report {
        let rows=Set(population.rowHashes)
        var evaluated=0,inspected=0,pair=0,overlap=Set<String>(),complete=true
        var comparedPopulations=Set<String>()
        for event in loaded.events {
            guard let saved=loaded.populations[event.populationSHA256] else {throw Failure("Exposure population is missing.")}
            let exact=saved.sourceSHA256==population.sourceSHA256 && saved.membershipSHA256==population.membershipSHA256
            if event.kind == .evaluated {if exact {evaluated+=1};continue}
            if exact {inspected+=1;if event.checkpointSHA256==population.checkpointSHA256 {pair+=1}}
            guard comparedPopulations.insert(event.populationSHA256).inserted else {continue}
            if saved.sourceSHA256==population.sourceSHA256 {overlap.formUnion(rows.intersection(saved.rowHashes))}
            else {complete=false}
        }
        return .init(priorHistoryUnknown:true,evaluatedCount:evaluated,inspectionCount:inspected,
            checkpointInspectionCount:pair,sameSourceInspectedOverlap:overlap.count,overlapIsComplete:complete)
    }

    private func load(_ fd:Int32)throws->Loaded {
        guard try totalBytes(fd)<=maximumBytes else {throw Failure("Exposure journal exceeds its read budget.")}
        let raw=try read("index.json",fd:fd,limit:4_000_000)
        let index:Index=try decode(raw,keys:["schemaVersion","originSHA256","events"])
        guard let object=try JSONSerialization.jsonObject(with:raw) as? [String:Any],
              let entries=object["events"] as? [[String:Any]],entries.allSatisfy({Set($0.keys)==["file","sha256"]}) else {throw Failure("Unknown exposure index entry fields.")}

        guard index.schemaVersion==1,WorkflowArchiveCodec.isHash(index.originSHA256),index.events.count<=10000,
              Set(index.events.map(\.file)).count==index.events.count else {throw Failure("Invalid exposure journal index.")}
        let originRaw=try read("origin.json",fd:fd,limit:4096)
        guard WorkflowArchiveCodec.hash(originRaw)==index.originSHA256 else {throw Failure("Exposure origin changed.")}
        let origin:Origin=try decode(originRaw,keys:["schemaVersion","trackerID","priorHistoryUnknown","startedAt"])
        guard origin.schemaVersion==1,UUID(uuidString:origin.trackerID) != nil,origin.priorHistoryUnknown,validTime(origin.startedAt) else {throw Failure("Invalid exposure origin.")}
        var events:[SavedEvent]=[],populations:[String:SavedPopulation]=[:],expected:Set<String>=["owner.lock","origin.json","index.json"],unique=Set<String>()
        var evaluationIdentities:[String:SavedEvent]=[:]
        for entry in index.events {
            guard entry.file.hasPrefix("event-"),entry.file.hasSuffix(".json"),
                  UUID(uuidString:String(entry.file.dropFirst(6).dropLast(5))) != nil,WorkflowArchiveCodec.isHash(entry.sha256) else {throw Failure("Invalid exposure entry path.")}
            let data=try read(entry.file,fd:fd,limit:8192)
            guard WorkflowArchiveCodec.hash(data)==entry.sha256 else {throw Failure("Exposure event changed.")}
            let event:SavedEvent=try decode(data,keys:["schemaVersion","kind","evaluationID","projectID","manifestSHA256","snapshotReceiptSHA256","checkpointSHA256","populationSHA256","recordedAt"])
            guard event.schemaVersion==1,UUID(uuidString:event.evaluationID)?.uuidString==event.evaluationID,UUID(uuidString:event.projectID)?.uuidString==event.projectID,
                  [event.manifestSHA256,event.snapshotReceiptSHA256,event.checkpointSHA256,event.populationSHA256].allSatisfy(WorkflowArchiveCodec.isHash),
                  validTime(event.recordedAt),unique.insert(event.evaluationID+":"+event.kind.rawValue).inserted else {throw Failure("Invalid or conflicting exposure event.")}
            expected.insert(entry.file)
            let populationName="population-"+event.populationSHA256+".json";expected.insert(populationName)
            if populations[event.populationSHA256] == nil {
                let bytes=try read(populationName,fd:fd,limit:8_000_000)
                guard WorkflowArchiveCodec.hash(bytes)==event.populationSHA256 else {throw Failure("Exposure membership changed.")}
                let value:SavedPopulation=try decode(bytes,keys:["sourceSHA256","membershipSHA256","rowHashes"])
                guard WorkflowArchiveCodec.isHash(value.sourceSHA256),WorkflowArchiveCodec.isHash(value.membershipSHA256),
                      (1...100000).contains(value.rowHashes.count),value.rowHashes==value.rowHashes.sorted(),
                      Set(value.rowHashes).count==value.rowHashes.count,value.rowHashes.allSatisfy(WorkflowArchiveCodec.isHash) else {throw Failure("Invalid exposure population.")}
                populations[event.populationSHA256]=value
            }
            if let prior=evaluationIdentities[event.evaluationID] {
                guard prior.projectID==event.projectID,prior.manifestSHA256==event.manifestSHA256,
                      prior.snapshotReceiptSHA256==event.snapshotReceiptSHA256,prior.checkpointSHA256==event.checkpointSHA256,
                      prior.populationSHA256==event.populationSHA256 else {throw Failure("Conflicting evaluation history across event kinds.")}
            }
            evaluationIdentities[event.evaluationID]=event
            events.append(event)
        }
        guard try names(fd)==expected else {throw Failure("Exposure journal is incomplete or contains an interrupted publication. Restore it before disclosure.")}
        return .init(index:index,events:events,populations:populations)
    }
    private func validTime(_ value:Double)->Bool {value.isFinite && (0...253402300799).contains(value)}
    private func encode<T:Encodable>(_ value:T)throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];return try encoder.encode(value)}
    private func decode<T:Decodable>(_ data:Data,keys:Set<String>)throws->T {
        guard let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],Set(object.keys)==keys else {throw Failure("Unknown or missing exposure record fields.")}
        return try JSONDecoder().decode(T.self,from:data)
    }
    private func access<T>(create:Bool,_ action:(Int32,Bool)throws->T)throws->T {
        guard let resolved=realpath(root.path,nil) else {throw Failure("Exposure storage root is unavailable.")}
        defer {free(resolved)}
        let canonical=String(cString:resolved),parent=Darwin.open(canonical,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)
        guard parent>=0 else {throw Failure("Exposure storage root is unreadable.")};defer {Darwin.close(parent)}
        var created=false
        if create {
            if mkdirat(parent,namespace,0o700)==0 {created=true;guard fsync(parent)==0 else {throw Failure("Exposure directory could not be flushed.")}}
            else if errno != EEXIST {throw Failure("Exposure storage could not be created.")}
        }
        let path=URL(fileURLWithPath:canonical).appendingPathComponent(namespace)
        let lease=try WorkflowFileLease(directory:path,name:"owner.lock",mode:.exclusive,create:created)
        defer {lease.close()}
        let fd=openat(parent,namespace,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd>=0 else {throw Failure("Exposure directory disappeared.")};defer {Darwin.close(fd)}
        var info=stat();guard fstat(fd,&info)==0,
              WorkflowFileIdentity(device:UInt64(UInt32(bitPattern:info.st_dev)),inode:UInt64(info.st_ino))==lease.parentIdentity else {throw Failure("Exposure directory identity changed.")}
        try lease.checkIdentity()
        let value=try action(fd,created)
        try lease.checkIdentity()
        return value
    }
    private func names(_ fd:Int32)throws->Set<String> {
        let copy=dup(fd);guard copy>=0 else {throw Failure("Cannot enumerate exposure records.")}
        guard let directory=fdopendir(copy) else {Darwin.close(copy);throw Failure("Cannot enumerate exposure records.")}
        defer {closedir(directory)}
        // dup shares the directory cursor; always rewind before a fresh inventory.
        rewinddir(directory)
        var values=Set<String>()
        while true {
            errno=0
            guard let entry=readdir(directory) else {
                guard errno==0 else {throw Failure("Exposure directory enumeration failed.")}
                break
            }
            let name=withUnsafePointer(to:&entry.pointee.d_name) {$0.withMemoryRebound(to:CChar.self,capacity:256) {String(cString:$0)}}
            if name=="." || name==".." {continue}
            values.insert(name)
            guard values.count<=20003 else {throw Failure("Exposure file count exceeds its bound.")}
        }
        return values
    }
    private func totalBytes(_ fd:Int32)throws->Int {
        var total=0
        for name in try names(fd) {
            var info=stat()
            guard fstatat(fd,name,&info,AT_SYMLINK_NOFOLLOW)==0,(info.st_mode&S_IFMT)==S_IFREG,
                  info.st_nlink==1,info.st_uid==geteuid(),info.st_mode&0o022==0,info.st_size>=0,
                  info.st_size<=Int64(maximumBytes-total) else {throw Failure("Exposure inventory is unsafe or exceeds its byte budget.")}
            total+=Int(info.st_size)
        }
        return total
    }
    private func read(_ name:String,fd:Int32,limit:Int)throws->Data {
        guard !name.contains("/"),name != ".",name != "..",!name.contains("\0") else {throw Failure("Invalid exposure filename.")}
        let file=openat(fd,name,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)
        guard file>=0 else {throw Failure("Required exposure evidence is missing.")};defer {Darwin.close(file)}
        var before=stat()
        guard fstat(file,&before)==0,(before.st_mode&S_IFMT)==S_IFREG,before.st_nlink==1,
              before.st_uid==geteuid(),before.st_mode&0o022==0,before.st_size>=0,before.st_size<=Int64(limit) else {throw Failure("Invalid exposure file bounds or permissions.")}
        var data=Data(),buffer=[UInt8](repeating:0,count:65536)
        while true {
            let count=Darwin.read(file,&buffer,min(buffer.count,limit-data.count+1))
            if count<0 {if errno==EINTR {continue};throw Failure("Exposure read failed.")}
            if count==0 {break}
            data.append(contentsOf:buffer.prefix(count))
            guard data.count<=limit else {throw Failure("Exposure file exceeds its bound.")}
        }
        var after=stat(),named=stat()
        guard fstat(file,&after)==0,fstatat(fd,name,&named,AT_SYMLINK_NOFOLLOW)==0,
              before.st_dev==after.st_dev,before.st_ino==after.st_ino,before.st_size==after.st_size,
              before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,
              named.st_dev==after.st_dev,named.st_ino==after.st_ino,named.st_size==after.st_size,
              data.count==Int(before.st_size) else {throw Failure("Exposure evidence changed while reading.")}
        return data
    }
    private func publish(_ data:Data,name:String,fd:Int32)throws {
        guard data.count<=8_000_000 else {throw Failure("Exposure record exceeds its bound.")}
        let file=openat(fd,name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)
        guard file>=0 else {throw Failure("Exposure record already exists or cannot be created.")};defer {Darwin.close(file)}
        try data.withUnsafeBytes {bytes in
            var offset=0
            while offset<bytes.count {
                let count=Darwin.write(file,bytes.baseAddress!.advanced(by:offset),bytes.count-offset)
                if count<0 && errno==EINTR {continue}
                guard count>0 else {throw Failure("Exposure publication was interrupted.")};offset+=count
            }
        }
        guard fsync(file)==0,fsync(fd)==0 else {throw Failure("Exposure record could not be durably flushed.")}
    }
    private func replaceIndex(_ data:Data,fd:Int32)throws {
        let temporary="index-"+UUID().uuidString+".tmp"
        try publish(data,name:temporary,fd:fd)
        guard renameat(fd,temporary,fd,"index.json")==0,fsync(fd)==0 else {throw Failure("Exposure journal index publication was interrupted.")}
    }
}
