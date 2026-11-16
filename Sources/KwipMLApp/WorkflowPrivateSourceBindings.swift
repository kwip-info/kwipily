import Foundation
import Combine
import KwipMLCore

/// Private local references, deliberately absent from WorkflowDocument and portable exports.
/// The owner controls graph identity; this sidecar has its own revision for asynchronous source selection.
@MainActor final class WorkflowPrivateSourceBindings: ObservableObject {
    let owner: WorkflowAuthoringStore
    @Published private(set) var revision: UInt64
    @Published private(set) var bindings: [WorkflowNodeAddress: String]
    @Published private(set) var error: String?
    private let ephemeral:Bool
    private let directory: URL
    private let parts: [String]
    private static let maximumBytes = 1_000_000
    private static let maximumBindings = 128
    private struct Entry: Codable {
        let address: WorkflowNodeAddress
        let path: String
    }
    private struct Envelope: Codable {
        let schemaVersion: Int
        let projectID: String
        let revision: UInt64
        let entries: [Entry]
    }

    init(owner: WorkflowAuthoringStore, directory: URL, ephemeral:Bool=false) throws {
        guard let uuid = UUID(uuidString: owner.projectID), owner.isCurrent(owner.lease()) else {
            throw Failure("Private sources require an active workflow with a UUID identity.")
        }
        self.owner = owner; self.directory = directory;self.ephemeral=ephemeral
        parts = ["workflow-sources", uuid.uuidString, "bindings.json"]
        if let data = try DiagnosticFileIO.readIfPresent(root: directory, parts: parts, limit: Self.maximumBytes) {
            let envelope = try Self.decode(data, projectID: owner.projectID)
            revision = envelope.revision
            bindings = Dictionary(uniqueKeysWithValues: envelope.entries.map { ($0.address, $0.path) })
        } else {
            revision = 0; bindings = [:]
        }
    }

    /// Cached local lookup only; does not touch source files, resolve bookmarks, or claim readable access.
    func path(for address: WorkflowNodeAddress) -> String? {
        guard owner.isCurrent(owner.lease()), isSource(address) else { return nil }
        return bindings[address]
    }

    /// Presence only: rendering must never reopen user files, resolve grants or connect databases.
    var configurationIssues:[WorkflowModelValidationIssue] {
        guard owner.isCurrent(owner.lease()) else {return []}
        return owner.document.nodes.compactMap {node in
            guard isSource(node.address),path(for:node.address)==nil,case .dataOperator(let id,let config)=node.payload else {return nil}
            let message:String
            if id==DataFlowSourceOperators.fileInput {message="Choose a file or directory for File Input."}
            else if id==DataFlowSourceOperators.tableInput,(try? TableInputConfiguration.format(of:config)) == .api {message="Fetch an API snapshot in Table Input before running this flow."}
            else if id==DataFlowSourceOperators.tableInput,(try? TableInputConfiguration.format(of:config)) == .database {message="Connect and capture the selected database table in Table Input."}
            else if id==DataFlowSourceOperators.capturedTable {message="Choose the captured table for this source."}
            else {message="Choose a local file for Table Input."}
            return .init(address:node.address,message:message)
        }
    }

    @discardableResult func setPath(_ path: String?, for address: WorkflowNodeAddress,
                                    lease: WorkflowEditorLease, expectedRevision: UInt64) -> Bool {
        updatePath(path, for: address, lease: lease, expectedRevision: expectedRevision, permitsExecutionOwner: false)
    }

    /// Only a joined source capture may commit while its exact workspace busy
    /// reservation remains held. Ordinary pickers/edits retain their idle guard.
    @discardableResult func commitCapturedPath(_ path: String, for address: WorkflowNodeAddress,
                                              lease: WorkflowEditorLease, expectedRevision: UInt64,
                                              workspace: WorkspaceStore, reservationID: UUID) -> Bool {
        guard workspace.activeProjectID == owner.projectID, workspace.isPreparingData,
              workspace.dataPreparationReservationID == reservationID, owner.isBusy else {
            error = "The source capture no longer owns the active workspace reservation."
            return false
        }
        return updatePath(path, for: address, lease: lease, expectedRevision: expectedRevision, permitsExecutionOwner: true)
    }

    private func updatePath(_ path: String?, for address: WorkflowNodeAddress,
                            lease: WorkflowEditorLease, expectedRevision: UInt64, permitsExecutionOwner: Bool) -> Bool {
        do {
            guard owner.isCurrent(lease), permitsExecutionOwner || !owner.isBusy, expectedRevision == revision else {
                throw Failure("This source selection belongs to an earlier workflow or binding revision, or execution is active. Choose the source again.")
            }
            guard isSource(address) else { throw Failure("Choose an existing data-domain source.") }
            if let path { try Self.validate(address: address, path: path) }
            if bindings[address] == path { error = nil; return true }
            guard revision < UInt64.max else { throw Failure("The private source revision limit has been reached.") }
            var next = bindings
            next[address] = path
            guard next.count <= Self.maximumBindings else { throw Failure("At most 128 private source bindings, including inactive references, can be retained.") }
            let envelope = Envelope(schemaVersion: 1, projectID: owner.projectID, revision: revision + 1,
                entries: next.sorted { $0.key.localID < $1.key.localID }.map { Entry(address: $0.key, path: $0.value) })
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(envelope)
            guard data.count <= Self.maximumBytes else { throw Failure("Private source sidecar exceeds 1 MB.") }
            if !ephemeral {try DiagnosticFileIO.writeAtomic(data, root: directory, parts: parts)}
            // No graph edits or history mutations: private binding authority is independent.
            bindings = next; revision = envelope.revision; error = nil
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    private func isSource(_ address: WorkflowNodeAddress) -> Bool {
        address.domain == .data && owner.document.nodes.contains {
            return $0.address == address && DataFlowSourceOperators.isSource($0)
        }
    }
    private static func validate(address: WorkflowNodeAddress, path: String) throws {
        guard address.domain == .data, !address.localID.isEmpty, address.localID.unicodeScalars.count <= 128,
              path.hasPrefix("/"), path.utf8.count <= 4096, !path.contains("\0") else {
            throw Failure("Private sources require a bounded data-node identity and an absolute local path without NUL characters.")
        }
    }
    private static func decode(_ data: Data, projectID: String) throws -> Envelope {
        guard data.count <= maximumBytes,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == ["schemaVersion", "projectID", "revision", "entries"],
              let entries = object["entries"] as? [[String: Any]], entries.count <= maximumBindings else {
            throw Failure("Private source sidecar is malformed, unsupported, or oversized. Its saved references were preserved.")
        }
        for entry in entries {
            guard Set(entry.keys) == ["address", "path"], let address = entry["address"] as? [String: Any],
                  Set(address.keys) == ["domain", "localID"] else { throw Failure("Private source entry contains unsupported fields.") }
        }
        let value = try JSONDecoder().decode(Envelope.self, from: data)
        guard value.schemaVersion == 1, UUID(uuidString: value.projectID) != nil,
              UUID(uuidString: value.projectID) == UUID(uuidString: projectID),
              Set(value.entries.map(\.address)).count == value.entries.count else {
            throw Failure("Private source identity, schema version, or duplicate entries are invalid.")
        }
        for entry in value.entries { try validate(address: entry.address, path: entry.path) }
        return value
    }
    private struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
