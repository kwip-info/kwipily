import Foundation
import Darwin
import KwipMLCore

struct WorkflowProgramExportPackage {
    static let limits = ["program.pt2":100_000_000,"inference.py":200_000,"README.md":100_000,"requirements.txt":10_000]
    let directory: URL
    let cleanupDirectory: URL
    let identity: WorkflowProgramExportIdentity
    let sourceManifestSHA256: String
    let configuration: ProgramExportConfiguration
    let torchVersion: String
    let inputDtype: String
    let inputSampleShape: [Int]
    let outputSampleShape: [Int]
    let files: [WorkflowArchiveFile]
    let manifestBytes: Data
    var storageLifetime: WorkflowExportStorageLifetime? = nil
    var totalBytes: Int { files.reduce(manifestBytes.count) { $0 + $1.bytes } }
    var fileCount: Int { files.count + 1 }
    func remove() {
        if let storageLifetime { storageLifetime.remove() }
        else { try? FileManager.default.removeItem(at:cleanupDirectory) }
    }
    static func verify(directory:URL,expected:WorkflowProgramExportIdentity,configuration:ProgramExportConfiguration,
                       sourceManifestSHA256:String,manifestSHA256:String,expectedTorchVersion:String,cleanupDirectory:URL?=nil)throws->Self {
        try Task.checkCancellation()
        if let cleanupDirectory {
            guard directory.standardizedFileURL == cleanupDirectory.appendingPathComponent("package").standardizedFileURL else {throw failure("Program cleanup scope differs from its retained operation.")}
        }
        let descriptor=try WorkflowEvaluationOwnership.openDirectory(directory)
        defer {Darwin.close(descriptor)}
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["manifest.json"],limit:2_000_000)
        guard WorkflowArchiveCodec.isHash(manifestSHA256),WorkflowArchiveCodec.hash(bytes) == manifestSHA256 else {throw failure("Program manifest differs from the completed worker pin.")}
        let decoded=try JSONDecoder().decode(JSONValue.self,from:bytes)
        let value=try object(decoded,keys:["schemaVersion","kind","runtime","configuration","input","output","sourceIdentity","shapeConstraints","rangeConstraints","state","files"])
        guard value["schemaVersion"] == .integer(1),value["kind"] == .string("kwipmlExportedProgram"),value["configuration"] == .object(configuration.values) else {throw failure("Unsupported program manifest or changed configuration.")}
        let runtime=try object(value["runtime"],keys:["torch","device"])
        guard !expectedTorchVersion.isEmpty,expectedTorchVersion.utf8.count <= 80,runtime["torch"] == .string(expectedTorchVersion),runtime["device"] == .string(configuration.device) else {throw failure("Captured Torch version or device differs from the selected runtime.")}
        let source=try object(value["sourceIdentity"],keys:["runID","requestIdentity","modelManifestSHA256","exportManifestSHA256"])
        guard UUID(uuidString:expected.runID)?.uuidString == expected.runID,
              [expected.requestIdentity,expected.modelManifestSHA256,sourceManifestSHA256].allSatisfy(WorkflowArchiveCodec.isHash),
              source == ["runID":.string(expected.runID),"requestIdentity":.string(expected.requestIdentity),"modelManifestSHA256":.string(expected.modelManifestSHA256),"exportManifestSHA256":.string(sourceManifestSHA256)] else {throw failure("Program source differs from the selected saved model.")}
        let input=try tensor(value["input"],maximumBatch:configuration.maximumBatchSize)
        let output=try tensor(value["output"],maximumBatch:configuration.maximumBatchSize)
        guard output.0 == "float32" else {throw failure("Program output must retain a real float32 tensor contract.")}
        let shape=try object(value["shapeConstraints"],keys:["mode","leadingAxis","minimum","maximum","example"])
        guard shape == ["mode":.string(configuration.batchMode),"leadingAxis":.string("batch"),"minimum":.integer(Int64(configuration.minimumBatchSize)),"maximum":.integer(Int64(configuration.maximumBatchSize)),"example":.integer(Int64(configuration.exampleBatchSize))],case .array(let ranges)=value["rangeConstraints"] else {throw failure("Program batch constraints differ from the configuration.")}
        if configuration.batchMode == "static" {
            guard ranges.isEmpty else {throw failure("Static programs cannot report symbolic axes.")}
        } else {
            guard ranges.count == 1 else {throw failure("Bounded programs require one leading symbolic axis.")}
            let range=try object(ranges[0],keys:["symbol","minimum","maximum"])
            guard let symbol=range["symbol"]?.stringValue,!symbol.isEmpty,symbol.utf8.count <= 128,
                  range["minimum"] == .integer(Int64(configuration.minimumBatchSize)),range["maximum"] == .integer(Int64(configuration.maximumBatchSize)) else {throw failure("Native symbolic bounds differ from the requested interval.")}
        }
        guard case .array(let states)=value["state"],states.count <= 4096 else {throw failure("Program state inventory exceeds its bound.")}
        var names=Set<String>(),totals:[String:Int64]=[:]
        for raw in states {
            let state=try object(raw,keys:["name","kind","dtype","shape"])
            guard let name=state["name"]?.stringValue,!name.isEmpty,name.utf8.count <= 512,names.insert(name).inserted,
                  let kind=state["kind"]?.stringValue,["parameter","buffer","constant"].contains(kind),
                  let dtype=state["dtype"]?.stringValue,["float32","int64","bool"].contains(dtype),case .array(let axes)=state["shape"],axes.count <= 8 else {throw failure("Invalid program state metadata.")}
            var count:Int64=1
            for axis in axes {
                guard case .integer(let n)=axis,(0...2_000_000).contains(n),n == 0 || count <= 2_000_000/n else {throw failure("Program state shape exceeds the element budget.")}
                count *= n
            }
            let total=(totals[kind] ?? 0)+count
            guard total <= 2_000_000 else {throw failure("Program state exceeds its element budget.")}
            totals[kind]=total
        }
        guard case .array(let entries)=value["files"],entries.count == limits.count else {throw failure("Program file inventory differs from its supported package.")}
        var files:[WorkflowArchiveFile]=[],seen=Set<String>(),total=bytes.count
        for raw in entries {
            let row=try object(raw,keys:["path","bytes","sha256"])
            guard let name=row["path"]?.stringValue,let ceiling=limits[name],seen.insert(name).inserted,
                  case .integer(let size)=row["bytes"],size > 0,size <= Int64(ceiling),
                  let pin=row["sha256"]?.stringValue,WorkflowArchiveCodec.isHash(pin) else {throw failure("Unreviewed or oversized program file.")}
            total += Int(size)
            guard total <= 102_310_000 else {throw failure("Program package exceeds its total byte bound.")}
            let content=try DiagnosticFileIO.read(root:directory,parts:[name],limit:Int(size))
            guard content.count == Int(size),WorkflowArchiveCodec.hash(content) == pin else {throw failure("Program artifact changed: "+name)}
            files.append(.init(path:name,bytes:Int(size),sha256:pin))
        }
        guard seen == Set(limits.keys),Set(try FileManager.default.contentsOfDirectory(atPath:directory.path)) == seen.union(["manifest.json"]) else {throw failure("Program package contains missing or unreviewed files.")}
        return .init(directory:directory,cleanupDirectory:cleanupDirectory ?? directory,identity:expected,sourceManifestSHA256:sourceManifestSHA256,configuration:configuration,torchVersion:expectedTorchVersion,inputDtype:input.0,inputSampleShape:input.1,outputSampleShape:output.1,files:files,manifestBytes:bytes)
    }
    func publish(to destination:URL)throws {
        try WorkflowExportPublication.publish(directory:directory,files:files,manifestName:"manifest.json",manifestBytes:manifestBytes,to:destination)
    }
    private static func object(_ value:JSONValue?,keys:Set<String>)throws->[String:JSONValue] {
        guard case .object(let value)=value,Set(value.keys) == keys else {throw failure("Unexpected program contract fields.")}
        return value
    }
    private static func tensor(_ value:JSONValue?,maximumBatch:Int)throws->(String,[Int]) {
        let spec=try object(value,keys:["dtype","sampleShape"])
        guard let dtype=spec["dtype"]?.stringValue,["float32","int64"].contains(dtype),case .array(let axes)=spec["sampleShape"],(1...7).contains(axes.count) else {throw failure("Program tensor contract is invalid.")}
        var shape:[Int]=[],count=maximumBatch
        for axis in axes {
            guard case .integer(let n)=axis,(1...4096).contains(n),count <= 2_000_000/Int(n) else {throw failure("Program tensor exceeds the maximum-batch allocation bound.")}
            count *= Int(n);shape.append(Int(n))
        }
        return (dtype,shape)
    }
    private static func failure(_ message:String)->WorkflowArchiveError {WorkflowArchiveError(message)}
}
