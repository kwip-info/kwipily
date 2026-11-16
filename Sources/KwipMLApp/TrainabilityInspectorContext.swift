import Foundation
import Combine
import KwipMLCore

/// Structured override draft; callers supply actual expanded native owners and command authority.
@MainActor final class TrainabilityInspectorContext: ObservableObject {
    let owners: [InitializationOwnerChoice]
    let origin: String
    @Published private(set) var draftConfiguration: [String: JSONValue]
    @Published private(set) var rejection: String?
    private(set) var expectedConfiguration: [String: JSONValue]
    private var baseline: [String: JSONValue]
    private let current: () -> Bool
    private let busy: () -> Bool
    private let validateContext: ([String: JSONValue]) throws -> Void
    private let commit: ([String: JSONValue], [String: JSONValue]) -> Bool
    private let pending: (Bool) -> Void
    init(configuration: [String: JSONValue], owners: [InitializationOwnerChoice], origin: String,
         isCurrent: @escaping () -> Bool, isBusy: @escaping () -> Bool,
         validate: @escaping ([String: JSONValue]) throws -> Void = { _ in },
         apply: @escaping ([String: JSONValue], [String: JSONValue]) -> Bool,
         pending: @escaping (Bool) -> Void) {
        self.owners = owners; self.origin = origin
        draftConfiguration = configuration; expectedConfiguration = configuration; baseline = configuration
        current = isCurrent; busy = isBusy; validateContext = validate; commit = apply; self.pending = pending
    }
    var rows: [JSONValue] { if case .array(let rows) = draftConfiguration["owners"] { return rows }; return [] }
    var isBusy: Bool { !current() || busy() }
    var hasChanges: Bool { draftConfiguration != baseline }
    var validationMessage: String? {
        do {
            let config = try TrainabilityConfiguration(values: draftConfiguration)
            for row in config.owners {
                guard case .object(let entry) = row,
                      owners.contains(where: { $0.node == entry["node"]?.stringValue && $0.parameter == entry["parameter"]?.stringValue }) else {
                    return "A selected learned array is no longer available. Choose its replacement or remove that override."
                }
            }
            try validateContext(config.values)
            return nil
        } catch { return error.localizedDescription }
    }
    func validatedConfiguration() throws -> [String: JSONValue] {
        if let message = validationMessage { throw GraphConfigurationError(message) }
        return try TrainabilityConfiguration(values: draftConfiguration).values
    }
    var canApply: Bool { !isBusy && hasChanges && validationMessage == nil }
    private func choice(_ id: String) -> InitializationOwnerChoice? { owners.first { $0.id == id } }
    private func identity(_ row: JSONValue, matches owner: InitializationOwnerChoice) -> Bool {
        guard case .object(let entry) = row else { return false }
        return entry["node"] == .string(owner.node) && entry["parameter"] == .string(owner.parameter)
    }
    private func replace(_ rows: [JSONValue]) -> Bool {
        guard !isBusy, (try? TrainabilityConfiguration(values: ["owners": .array(rows)])) != nil else { return false }
        draftConfiguration = ["owners": .array(rows)]; rejection = nil; pending(hasChanges); return true
    }
    @discardableResult func addOwner(_ id: String) -> Bool {
        guard let owner = choice(id), rows.count < TrainabilityConfiguration.maximumOwners,
              !rows.contains(where: { identity($0, matches: owner) }) else { return false }
        return replace(rows + [.object(["node": .string(owner.node), "parameter": .string(owner.parameter), "mode": .string("inherit")])])
    }
    @discardableResult func setOwner(_ id: String, at index: Int) -> Bool {
        guard let owner = choice(id), rows.indices.contains(index),
              !rows.enumerated().contains(where: { $0.offset != index && identity($0.element, matches: owner) }),
              case .object(var row) = rows[index] else { return false }
        row["node"] = .string(owner.node); row["parameter"] = .string(owner.parameter)
        var next = rows; next[index] = .object(row); return replace(next)
    }
    @discardableResult func setMode(_ mode: String, at index: Int) -> Bool {
        guard ["inherit", "trainable", "frozen"].contains(mode), rows.indices.contains(index),
              case .object(var row) = rows[index] else { return false }
        row["mode"] = .string(mode); var next = rows; next[index] = .object(row); return replace(next)
    }
    @discardableResult func removeOwner(at index: Int) -> Bool {
        guard rows.indices.contains(index) else { return false }
        var next = rows; next.remove(at: index); return replace(next)
    }
    @discardableResult func apply() -> Bool {
        guard !isBusy, validationMessage == nil else { return false }
        do {
            let canonical = try TrainabilityConfiguration(values: draftConfiguration).values
            guard hasChanges else { pending(false); return true }
            guard commit(expectedConfiguration, canonical) else {
                rejection = "The draft was not applied. Its source may have changed; your edits are retained."; return false
            }
            expectedConfiguration = canonical; baseline = canonical; draftConfiguration = canonical
            rejection = nil; pending(false); return true
        } catch { rejection = error.localizedDescription; return false }
    }
    @discardableResult func discard() -> Bool {
        guard !isBusy else { return false }
        draftConfiguration = baseline; rejection = nil; pending(false); return true
    }
    func releasePendingEdits() { pending(false) }
    /// No numeric text exists in this choice-only editor. Persist typed rows, including unresolved owners.
    func serializedDraft() -> Data? {
        guard !isBusy, let data = try? JSONEncoder().encode(JSONValue.object(["schemaVersion": 1, "configuration": .object(draftConfiguration)])), data.count <= 131072 else { return nil }
        return data
    }
    @discardableResult func restorePending(_ data: Data) -> Bool {
        guard !isBusy, data.count <= 131072,
              let value = try? JSONDecoder().decode(JSONValue.self, from: data), case .object(let envelope) = value,
              Set(envelope.keys) == ["schemaVersion", "configuration"], envelope["schemaVersion"] == .integer(1),
              case .object(let configuration) = envelope["configuration"], Set(configuration.keys) == ["owners"],
              (try? TrainabilityConfiguration(values: configuration)) != nil else { return false }
        draftConfiguration = configuration; rejection = nil; pending(hasChanges); return true
    }
}
