import Foundation
import Darwin
import KwipMLCore

/// Native issuance of one captured, same-project, canonical directory capability.
/// Runtime accounting already includes all workflow-operations descendants.
enum SelfPlayCampaignGrant {
    static func enabled(_ values:[String:JSONValue])->Bool {if case .object(let c)=values["campaign"] {return c["enabled"] == .bool(true)};return false}
    static func captured(_ record:SelfPlayRecord,root:URL)throws->[String:JSONValue]? {
        let directory=try record.directory(root)
        guard FileManager.default.fileExists(atPath:directory.appendingPathComponent("input/campaign-grant.json").path)else{return nil}
        let request=try JSONDecoder().decode(WorkflowToolRequest.self,from:DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:262144))
        guard let entry=request.inputFiles.first(where:{$0.path=="input/campaign-grant.json"})else{throw WorkflowFileLeaseError.unsafePath}
        let data=try DiagnosticFileIO.read(root:directory,parts:["input","campaign-grant.json"],limit:2000000)
        guard WorkflowArchiveCodec.hash(data)==entry.sha256 else{throw WorkflowFileLeaseError.unsafePath}
        let grant=try JSONDecoder().decode([String:JSONValue].self,from:data)
        guard let id=grant["campaignID"]?.stringValue else{throw WorkflowFileLeaseError.unsafePath}
        let path=try self.directory(root:root,project:record.document.projectID,id:id);var value=stat()
        guard lstat(path.path,&value)==0,grant["device"]?.integerValue==Int(value.st_dev),grant["inode"]?.integerValue==Int(value.st_ino),(value.st_mode & S_IFMT)==S_IFDIR,value.st_uid==getuid(),value.st_mode & 0o077==0,try SelfPlayRecord.canonicalRoot(path).path==path.path else{throw WorkflowFileLeaseError.unsafePath}
        return grant
    }
    static func directory(root:URL,project:String,id:String)throws->URL {
        guard UUID(uuidString:project)?.uuidString==project,UUID(uuidString:id)?.uuidString==id else{throw WorkflowFileLeaseError.unsafePath}
        return try SelfPlayRecord.canonicalRoot(root).appendingPathComponent("workflow-operations/\(project)/\(id)/campaign")
    }
    static func issue(root:URL,project:String,id:String,action:String)throws->[String:JSONValue] {
        let path=try directory(root:root,project:project,id:id)
        if action=="start" || action=="fork" {try FileManager.default.createDirectory(at:path,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])}
        var value=stat();guard lstat(path.path,&value)==0,(value.st_mode & S_IFMT)==S_IFDIR,value.st_uid==getuid(),value.st_mode & 0o077==0,try SelfPlayRecord.canonicalRoot(path).path==path.path else{throw WorkflowFileLeaseError.unsafePath}
        let index=try? DiagnosticFileIO.read(root:path,parts:["campaign.json"],limit:2000000)
        return ["campaignID":.string(id),"device":.integer(Int64(value.st_dev)),"inode":.integer(Int64(value.st_ino)),"action":.string(action),"expectedIndexSHA256":index.map{.string(WorkflowArchiveCodec.hash($0))} ?? .null,"parent":.null]
    }
    static func index(_ record:SelfPlayRecord,root:URL)throws->[String:JSONValue]? {
        guard let grant=try captured(record,root:root),let id=grant["campaignID"]?.stringValue else{return nil}
        let path=try directory(root:root,project:record.document.projectID,id:id)
        return try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:path,parts:["campaign.json"],limit:2000000))
    }
}

extension SelfPlayCampaignGrant {
    struct Boundary:Identifiable {let id:String;let pinned:Bool}
    static func boundaries(_ record:SelfPlayRecord,root:URL)throws->[Boundary] {
        guard let grant=try captured(record,root:root),let id=grant["campaignID"]?.stringValue else{return []}
        let base=try directory(root:root,project:record.document.projectID,id:id).appendingPathComponent("replay/checkpoints")
        guard FileManager.default.fileExists(atPath:base.path)else{return []}
        let values=try FileManager.default.contentsOfDirectory(at:base,includingPropertiesForKeys:[])
        guard values.count<=32 else{throw WorkflowArchiveError("Checkpoint inventory exceeds its admitted bound.")}
        return try values.filter{$0.lastPathComponent.range(of:"^iteration-[0-9]{2,8}$",options:.regularExpression) != nil}.sorted{Int($0.lastPathComponent.dropFirst(10))!>Int($1.lastPathComponent.dropFirst(10))!}.map {url in
            _=try DiagnosticFileIO.read(root:url,parts:["manifest.json"],limit:2000000)
            return .init(id:url.lastPathComponent,pinned:(try? DiagnosticFileIO.read(root:url,parts:["PINNED"],limit:64)) != nil)
        }
    }
    static func pin(_ record:SelfPlayRecord,root:URL,boundary:String,value:Bool)throws {
        guard let grant=try captured(record,root:root),let id=grant["campaignID"]?.stringValue,boundary.range(of:"^iteration-[0-9]{2,8}$",options:.regularExpression) != nil else{throw WorkflowFileLeaseError.unsafePath}
        let base=try directory(root:root,project:record.document.projectID,id:id)
        let fd=open(base.appendingPathComponent("campaign.lock").path,O_RDWR|O_NOFOLLOW|O_NONBLOCK)
        guard fd>=0 else{throw WorkflowFileLeaseError.unsafePath};defer{close(fd)}
        guard flock(fd,LOCK_EX|LOCK_NB)==0 else{throw WorkflowArchiveError("Pause the campaign before changing checkpoint retention.")}
        let entries=try boundaries(record,root:root);guard entries.contains(where:{$0.id==boundary})else{throw WorkflowFileLeaseError.unsafePath}
        let checkpoint=base.appendingPathComponent("replay/checkpoints/"+boundary)
        if value {
            if entries.contains(where:{$0.id==boundary && $0.pinned}){return}
            guard entries.filter(\.pinned).count<17 else{throw WorkflowArchiveError("Keep at most17 explicit pins plus the active boundary.")}
            try DiagnosticFileIO.writeImmutable(Data("Owner pin\n".utf8),root:checkpoint,parts:["PINNED"],limit:64)
        }else {
            _=try DiagnosticFileIO.read(root:checkpoint,parts:["PINNED"],limit:64)
            try FileManager.default.removeItem(at:checkpoint.appendingPathComponent("PINNED"))
        }
    }
}
