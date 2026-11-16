import Foundation
import KwipMLCore

/// Private shared source bytes for one experiment group. No live authoring state,
/// grants or original paths are serialized. The future group record must pin manifestSHA256.
struct WorkflowExperimentInputs:Equatable {
    let directory:URL
    let manifestSHA256:String
    let sources:[WorkflowNodeAddress:WorkflowCapturedSource]
    private struct Manifest:Codable {
        let schemaVersion:Int
        let groupID:String
        let sources:[WorkflowArchiveSource]
    }
    static func capture(addresses:[WorkflowNodeAddress],parent:URL,
                        snapshot:(WorkflowNodeAddress)throws->WorkflowCapturedSource)throws->Self {
        try validateAddresses(addresses)
        let captures = try Dictionary(uniqueKeysWithValues: addresses.map { ($0, try snapshot($0)) })
        return try captureSources(addresses: addresses, parent: parent, sources: captures)
    }
    static func captureSources(addresses: [WorkflowNodeAddress], parent: URL,
                               sources: [WorkflowNodeAddress: WorkflowCapturedSource]) throws -> Self {
        try validateAddresses(addresses)
        let directory = parent.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        do { return try captureSources(addresses: addresses, directory: directory, sources: sources) }
        catch { try? FileManager.default.removeItem(at: directory); throw error }
    }
    static func captureSources(addresses: [WorkflowNodeAddress], directory: URL,
                               sources: [WorkflowNodeAddress: WorkflowCapturedSource]) throws -> Self {
        try validateAddresses(addresses)
        guard Set(addresses) == Set(sources.keys), UUID(uuidString: directory.lastPathComponent)?.uuidString == directory.lastPathComponent else {
            throw DataFlowContractError("Experiment input closure differs from the selected sources.")
        }
        var total = 0
        let aggregateLimit = WorkflowCapturedSourceArchive.aggregateLimit(hasCollections: sources.values.contains { $0.isFileCollection })
        for source in sources.values {
            let count = try WorkflowCapturedSourceArchive.declaredByteCount(source)
            guard count <= aggregateLimit - total else { throw DataFlowContractError("Experiment source closure exceeds its byte budget.") }
            total += count
        }
        let entries = try addresses.sorted(by: { $0.localID < $1.localID }).enumerated().map { index, address in
            try WorkflowCapturedSourceArchive.publish(sources[address]!, address: address, index: index, root: directory, prefix: "")
        }
        let manifest = Manifest(schemaVersion: 1, groupID: directory.lastPathComponent, sources: entries)
        let bytes = try WorkflowArchiveCodec.encode(manifest)
        try DiagnosticFileIO.writeImmutable(bytes, root: directory, parts: ["inputs.json"], limit: 2_000_000)
        return try open(directory: directory, expectedManifestSHA256: WorkflowPhaseArtifacts.hash(bytes))
    }

    /// The provider owns source permission lifetime through the complete byte read.
    static func captureBytes(addresses:[WorkflowNodeAddress],parent:URL, sourceURLs:[WorkflowNodeAddress:URL] = [:],
                             read:(WorkflowNodeAddress)throws->Data)throws->Self {
        try validateAddresses(addresses)
        let directory=parent.appendingPathComponent(UUID().uuidString,isDirectory:true)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        do {return try captureBytes(addresses:addresses,directory:directory,sourceURLs:sourceURLs,read:read)}
        catch {
            // Only this allocating overload owns cleanup of its newly-created directory.
            try? FileManager.default.removeItem(at:directory)
            throw error
        }
    }
    /// The caller retains directory ownership through completion. Partial bytes and
    /// ownership files survive all failures/cancellation; this overload never deletes.
    static func captureBytes(addresses:[WorkflowNodeAddress],directory:URL, sourceURLs:[WorkflowNodeAddress:URL] = [:],
                             read:(WorkflowNodeAddress)throws->Data)throws->Self {
        try validateAddresses(addresses)
        let groupID=directory.lastPathComponent
        guard UUID(uuidString:groupID)?.uuidString == groupID else {throw DataFlowContractError("Use the existing canonical UUID experiment directory.")}
        var isDirectory:ObjCBool=false
        guard FileManager.default.fileExists(atPath:directory.path,isDirectory:&isDirectory),isDirectory.boolValue else {throw DataFlowContractError("The caller must create and own the experiment directory before capture.")}
        var entries:[WorkflowArchiveSource]=[],total=0
        for (index,address) in addresses.sorted(by:{$0.localID<$1.localID}).enumerated() {
            try Task.checkCancellation()
            let bytes=try read(address)
            try Task.checkCancellation()
            guard bytes.count<=10_000_000,bytes.count<=256_000_000-total else {
                throw DataFlowContractError("Experiment sources exceed the byte budget.")
            }
            total+=bytes.count
            let path=WorkflowSourceFilename.archive(index:index,source:sourceURLs[address] ?? URL(fileURLWithPath:"source.csv"))
            try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:[path],limit:10_000_000)
            entries.append(.init(address:address,path:path,bytes:bytes.count,sha256:WorkflowPhaseArtifacts.hash(bytes)))
        }
        let manifest=Manifest(schemaVersion:1,groupID:groupID,sources:entries)
        let bytes=try WorkflowArchiveCodec.encode(manifest)
        try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:["inputs.json"],limit:2_000_000)
        let result=try open(directory:directory,expectedManifestSHA256:WorkflowPhaseArtifacts.hash(bytes))
        try Task.checkCancellation()
        return result
    }
    private static func validateAddresses(_ addresses:[WorkflowNodeAddress])throws {
        try Task.checkCancellation()
        guard (1...128).contains(addresses.count),Set(addresses).count == addresses.count,addresses.allSatisfy({$0.domain == .data}) else {
            throw DataFlowContractError("Experiment sources require distinct data addresses within the source limit.")
        }
    }
    static func open(directory:URL,expectedManifestSHA256:String)throws->Self {
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["inputs.json"],limit:2_000_000)
        guard WorkflowPhaseArtifacts.hash(bytes) == expectedManifestSHA256 else {
            throw DataFlowContractError("The experiment input manifest differs from its recorded identity.")
        }
        let manifest=try JSONDecoder().decode(Manifest.self,from:bytes)
        guard manifest.schemaVersion == 1,UUID(uuidString:manifest.groupID) != nil,
              manifest.groupID == directory.lastPathComponent,(1...128).contains(manifest.sources.count),
              Set(manifest.sources.map(\.address)).count == manifest.sources.count else {
            throw DataFlowContractError("Invalid experiment input manifest.")
        }
        var sources:[WorkflowNodeAddress:WorkflowCapturedSource]=[:],total=0
        var capturedPaths = Set<String>()
        let aggregateLimit = WorkflowCapturedSourceArchive.aggregateLimit(hasCollections: manifest.sources.contains { $0.collectionMembers != nil })
        for (index,entry) in manifest.sources.enumerated() {
            guard entry.address.domain == .data else { throw DataFlowContractError("Invalid experiment source address.") }
            let paths = try WorkflowCapturedSourceArchive.validate(entry, prefix: "")
            guard paths.isDisjoint(with: capturedPaths) else { throw DataFlowContractError("Duplicate experiment source closure paths.") }
            capturedPaths.formUnion(paths)
            // Historical scalar names retain their index contract; collection names do too.
            let expected = entry.collectionMembers == nil ? WorkflowSourceFilename.validArchive(entry.path, index: index) : entry.path == "source-\(index).collection/\(entry.sha256)/manifest.json"
            guard expected else { throw DataFlowContractError("Invalid experiment source path.") }
            let captured = try WorkflowCapturedSourceArchive.open(entry, root: directory, prefix: "")
            let count = try WorkflowCapturedSourceArchive.declaredByteCount(captured)
            guard count <= aggregateLimit - total else { throw DataFlowContractError("Experiment sources exceed the byte budget.") }
            total += count
            sources[entry.address] = captured
        }
        return .init(directory:directory,manifestSHA256:expectedManifestSHA256,sources:sources)
    }
    func verify()throws {
        guard try Self.open(directory:directory,expectedManifestSHA256:manifestSHA256) == self else {
            throw DataFlowContractError("Experiment input identities changed.")
        }
    }
}
