import Foundation
import Combine
import KwipMLCore

/// Private metadata selection, absent from portable workflow configuration. This
/// binder does not verify an archive or promise a cache hit; selection and worker
/// services must independently validate the pinned evidence before reuse.
@MainActor final class WorkflowPrivateTrainBindings<Reference:WorkflowPrivateTrainReference>:ObservableObject {
    let owner:WorkflowAuthoringStore
    @Published private(set) var revision:UInt64
    @Published private(set) var bindings:[WorkflowNodeAddress:Reference]
    @Published private(set) var error:String?
    private let ephemeral:Bool
    private let directory:URL
    private let parts:[String]
    private static var maximumBytes:Int {1_000_000}
    private static var maximumBindings:Int {256}
    private struct Entry:Codable {
        let address:WorkflowNodeAddress
        let reference:Reference
    }
    private struct Envelope:Codable {
        let schemaVersion:Int
        let projectID:String
        let revision:UInt64
        let entries:[Entry]
    }
    init(owner:WorkflowAuthoringStore,directory:URL,ephemeral:Bool=false)throws {
        guard let id=UUID(uuidString:owner.projectID),owner.isCurrent(owner.lease()) else {
            throw Failure("Private component reference selection requires an active workflow with a UUID identity.")
        }
        self.owner=owner;self.directory=directory;self.ephemeral=ephemeral
        parts=[Reference.storageNamespace,id.uuidString,"bindings.json"]
        if let bytes=try DiagnosticFileIO.readIfPresent(root:directory,parts:parts,limit:Self.maximumBytes) {
            let value=try Self.decode(bytes,projectID:owner.projectID)
            revision=value.revision
            bindings=Dictionary(uniqueKeysWithValues:value.entries.map {($0.address,$0.reference)})
        } else {revision=0;bindings=[:]}
    }
    /// Cached lookup only. Inactive references are retained for Undo, but are not
    /// exposed as an applicable component selection until that exact node exists again.
    func reference(for train:WorkflowNodeAddress)->Reference? {
        guard owner.isCurrent(owner.lease()),acceptsNode(train) else {return nil}
        return bindings[train]
    }
    @discardableResult func setReference(_ reference:Reference?,for train:WorkflowNodeAddress,
                                         lease:WorkflowEditorLease,expectedRevision:UInt64)->Bool {
        do {
            guard owner.isCurrent(lease),!owner.isBusy,expectedRevision == revision else {
                throw Failure("This component reference selection belongs to an earlier workflow or private binding revision, or execution is active. Select the reference again.")
            }
            guard acceptsNode(train) else {throw Failure("Choose an existing control component that accepts this reference.")}
            if let reference {try Self.validate(address:train,reference:reference,projectID:owner.projectID)}
            if bindings[train] == reference {error=nil;return true}
            guard revision < UInt64.max else {throw Failure("The private component reference selection revision limit has been reached.")}
            var next=bindings;next[train]=reference
            guard next.count<=Self.maximumBindings else {throw Failure("At most 256 private component reference selections, including inactive references, can be retained.")}
            let envelope=Envelope(schemaVersion:1,projectID:owner.projectID,revision:revision+1,
                entries:next.sorted {$0.key.localID<$1.key.localID}.map {Entry(address:$0.key,reference:$0.value)})
            let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
            let data=try encoder.encode(envelope)
            guard data.count<=Self.maximumBytes else {throw Failure("Private component reference selections exceed the 1 MB sidecar limit.")}
            if !ephemeral {try DiagnosticFileIO.writeAtomic(data,root:directory,parts:parts)}
            bindings=next;revision=envelope.revision;error=nil;return true
        } catch {self.error=error.localizedDescription;return false}
    }
    private func acceptsNode(_ address:WorkflowNodeAddress)->Bool {
        address.domain == .control && owner.document.nodes.contains {
            guard $0.address == address else {return false};return Reference.accepts($0)
        }
    }
    private static func validate(address:WorkflowNodeAddress,reference:Reference,projectID:String)throws {
        guard address.domain == .control,!address.localID.isEmpty,!address.localID.contains("\0"),address.localID.unicodeScalars.count<=128,
              let project=UUID(uuidString:reference.projectID),project == UUID(uuidString:projectID),
              UUID(uuidString:reference.directoryID) != nil,
              reference.hasValidPins else {
            throw Failure("A private component reference selection requires a bounded control-node identity, this project's UUID, a run-directory UUID and a lowercase SHA-256 manifest pin.")
        }
    }
    private static func decode(_ data:Data,projectID:String)throws->Envelope {
        guard data.count<=maximumBytes,
              let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],
              Set(object.keys)==["schemaVersion","projectID","revision","entries"],
              let entries=object["entries"] as? [[String:Any]],entries.count<=maximumBindings else {
            throw Failure("Private component reference selections are malformed, unsupported or oversized. Saved references were preserved.")
        }
        for entry in entries {
            guard Set(entry.keys)==["address","reference"],
                  let address=entry["address"] as? [String:Any],Set(address.keys)==["domain","localID"],
                  let reference=entry["reference"] as? [String:Any],
                  Set(reference.keys)==Reference.fieldNames else {
                throw Failure("Private component reference selection contains unknown or missing fields.")
            }
        }
        let value=try JSONDecoder().decode(Envelope.self,from:data)
        guard value.schemaVersion == 1,UUID(uuidString:value.projectID) != nil,
              UUID(uuidString:value.projectID)==UUID(uuidString:projectID),
              Set(value.entries.map(\.address)).count == value.entries.count else {
            throw Failure("Private component reference selection schema, project identity or duplicate addresses are invalid.")
        }
        for entry in value.entries {try validate(address:entry.address,reference:entry.reference,projectID:projectID)}
        return value
    }
    private struct Failure:LocalizedError {
        let message:String
        init(_ message:String) {self.message=message}
        var errorDescription:String? {message}
    }
}

/// Serialization and mutation rules are shared; each reference owns its pin contract.
protocol WorkflowPrivateTrainReference:Codable,Equatable {
    static var storageNamespace:String {get}
    static func accepts(_ node:WorkflowNode)->Bool
    static var fieldNames:Set<String> {get}
    var projectID:String {get}
    var directoryID:String {get}
    var hasValidPins:Bool {get}
}
extension WorkflowPrivateTrainReference {
    static func accepts(_ node:WorkflowNode)->Bool {
        guard node.address.domain == .control,case .train=node.payload else {return false}
        return true
    }
}
extension WorkflowArchiveCacheReference:WorkflowPrivateTrainReference {
    static var storageNamespace:String {"workflow-cache"}
    static var fieldNames:Set<String> {["projectID","directoryID","preparationManifestSHA256"]}
    var hasValidPins:Bool {WorkflowArchiveCodec.isHash(preparationManifestSHA256)}
}
typealias WorkflowPrivateCacheBindings = WorkflowPrivateTrainBindings<WorkflowArchiveCacheReference>

/// Local archive identity only. Selecting a reference does not verify its artifacts;
/// request capture must revalidate every pin before staging or executing a parent.
struct WorkflowCheckpointReference:WorkflowPrivateTrainReference {
    let projectID:String
    let directoryID:String
    let runID:String
    let requestIdentity:String
    let modelManifestSHA256:String
    static var storageNamespace:String {"workflow-checkpoint"}
    static var fieldNames:Set<String> {["projectID","directoryID","runID","requestIdentity","modelManifestSHA256"]}
    var hasValidPins:Bool {
        UUID(uuidString:runID) != nil && WorkflowArchiveCodec.isHash(requestIdentity)
            && WorkflowArchiveCodec.isHash(modelManifestSHA256)
    }
}
typealias WorkflowPrivateCheckpointBindings = WorkflowPrivateTrainBindings<WorkflowCheckpointReference>
