import Foundation
import Darwin

/// Private app-local policy, separate from the 4-KiB general appearance settings.
/// Directory descriptors anchor every read and atomic replacement; symlinks and
/// writable/shared files fail closed. Bearer secrets belong in Keychain, not here.
public struct MCPSettingsStore {
    public let root:URL
    public init(root:URL){self.root=root}
    private func directory()throws->Int32 {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let parent=open(root.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard parent>=0 else{throw MCPSettings.Refusal(code:"invalid_settings")};defer{close(parent)}
        if mkdirat(parent,"mcp",0o700)<0 && errno != EEXIST{throw MCPSettings.Refusal(code:"invalid_settings")}
        let fd=openat(parent,"mcp",O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard fd>=0 else{throw MCPSettings.Refusal(code:"invalid_settings")}
        var s=stat();guard fstat(fd,&s)==0,s.st_uid==getuid(),s.st_mode & 0o077 == 0 else{close(fd);throw MCPSettings.Refusal(code:"invalid_settings")}
        return fd
    }
    public func load()throws->MCPSettings {
        guard let data=try loadBytes("settings.json",limit:262144)else{return .factory}
        return try MCPSettings.decode(data)
    }
    public func save(_ settings:MCPSettings)throws {try saveBytes(settings.encoded(),name:"settings.json",limit:262144)}
    public func loadRequests()throws->Data? {try loadBytes("requests.json",limit:262144)}
    public func saveRequests(_ data:Data)throws {try saveBytes(data,name:"requests.json",limit:262144)}
    public func loadAudit()throws->Data? {try loadBytes("activity.json",limit:1_048_576)}
    public func saveAudit(_ data:Data)throws {try saveBytes(data,name:"activity.json",limit:1_048_576)}
    private func loadBytes(_ name:String,limit:Int)throws->Data? {
        let dir=try directory();defer{close(dir)}
        let fd=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK)
        if fd<0 && errno==ENOENT{return nil}
        guard fd>=0 else{throw MCPSettings.Refusal(code:"invalid_settings")};defer{close(fd)}
        var s=stat();guard fstat(fd,&s)==0,s.st_mode & S_IFMT == S_IFREG,s.st_uid==getuid(),s.st_nlink==1,s.st_mode & 0o077 == 0,s.st_size<=limit else{throw MCPSettings.Refusal(code:"invalid_settings")}
        var bytes=[UInt8](repeating:0,count:limit+1),count=0
        while count<bytes.count {
            let n=bytes.withUnsafeMutableBytes{Darwin.read(fd,$0.baseAddress!.advanced(by:count),$0.count-count)}
            if n<0 && errno==EINTR{continue};guard n>=0 else{throw MCPSettings.Refusal(code:"invalid_settings")};if n==0{break};count+=n
        }
        guard count<=limit else{throw MCPSettings.Refusal(code:"invalid_settings")}
        return Data(bytes.prefix(count))
    }
    private func saveBytes(_ data:Data,name:String,limit:Int)throws {
        guard data.count<=limit else{throw MCPSettings.Refusal(code:"invalid_settings")}
        let dir=try directory();defer{close(dir)}
        var prior=stat()
        if fstatat(dir,name,&prior,AT_SYMLINK_NOFOLLOW)==0 {
            guard prior.st_mode & S_IFMT == S_IFREG,prior.st_uid==getuid(),prior.st_nlink==1,prior.st_mode & 0o077 == 0 else{throw MCPSettings.Refusal(code:"invalid_settings")}
        }else if errno != ENOENT{throw MCPSettings.Refusal(code:"invalid_settings")}
        let temp=".mcp-"+UUID().uuidString
        let fd=openat(dir,temp,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0o600)
        guard fd>=0 else{throw MCPSettings.Refusal(code:"invalid_settings")}
        defer{close(fd);unlinkat(dir,temp,0)}
        var count=0
        while count<data.count {
            let n=data.withUnsafeBytes{Darwin.write(fd,$0.baseAddress!.advanced(by:count),$0.count-count)}
            if n<0 && errno==EINTR{continue};guard n>0 else{throw MCPSettings.Refusal(code:"invalid_settings")};count+=n
        }
        guard fsync(fd)==0,renameat(dir,temp,dir,name)==0 else{throw MCPSettings.Refusal(code:"invalid_settings")}
        _=fsync(dir)
    }
}
