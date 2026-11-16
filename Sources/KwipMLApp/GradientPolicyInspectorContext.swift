import Foundation
import Combine
import KwipMLCore

/// Captured graph endpoint choices; the caller owns resolution and freshness.
struct GradientEndpointChoice: Identifiable, Equatable {
    let node: String
    let port: String
    let name: String
    let dtype: ContractDType
    let shape: [Int]
    let isModelInput: Bool
    var id: String { "\(node.utf8.count):\(node)\(port.utf8.count):\(port)" }
    var title: String { name + " · " + port + " · " + dtype.rawValue + " [" + shape.map(String.init).joined(separator: " × ") + "]" }
}

@MainActor final class GradientPolicyInspectorContext: ObservableObject {
    static let maximumSnapshotBytes = 131_072
    let endpoints: [GradientEndpointChoice]
    let owners: [InitializationOwnerChoice]
    let origin: String
    @Published private(set) var draftConfiguration: [String: JSONValue]
    @Published private(set) var rawNumericDrafts: [String: String] = [:]
    @Published private(set) var rejection: String?
    private(set) var expectedConfiguration: [String: JSONValue]
    private(set) var resetGeneration = 0
    private var baseline: [String: JSONValue]
    private let current: () -> Bool
    private let busy: () -> Bool
    private let validateContext: ([String: JSONValue]) throws -> Void
    private let commit: ([String: JSONValue], [String: JSONValue]) -> Bool
    private let pending: (Bool) -> Void
    private var dormantUpstream: [String: JSONValue] = [:]
    init(configuration: [String: JSONValue], endpoints: [GradientEndpointChoice], owners: [InitializationOwnerChoice], origin: String,
         isCurrent: @escaping () -> Bool, isBusy: @escaping () -> Bool,
         validate: @escaping ([String: JSONValue]) throws -> Void = { _ in },
         apply: @escaping ([String: JSONValue], [String: JSONValue]) -> Bool, pending: @escaping (Bool) -> Void) {
        self.endpoints = endpoints; self.owners = owners; self.origin = origin
        draftConfiguration = configuration; expectedConfiguration = configuration; baseline = configuration
        current = isCurrent; busy = isBusy; validateContext = validate; commit = apply; self.pending = pending
    }
    var requests: [[String: JSONValue]] {
        guard case .array(let rows) = draftConfiguration["requests"] else { return [] }
        return rows.compactMap { if case .object(let value) = $0 { return value }; return nil }
    }
    var isBusy: Bool { !current() || busy() }
    var hasChanges: Bool { draftConfiguration != baseline }
    var hasUncommittedNumbers: Bool { !rawNumericDrafts.isEmpty }
    var canApply: Bool { !isBusy && hasChanges && !hasUncommittedNumbers && validationMessage == nil }
    func validatedConfiguration() throws -> [String: JSONValue] {
        guard !hasUncommittedNumbers else { throw GraphConfigurationError("Apply or revert pending numeric values first.") }
        return try canonical().values
    }
    var validationMessage: String? { do { _ = try canonical(); return nil } catch { return error.localizedDescription } }
    private func report() { pending(hasChanges || hasUncommittedNumbers) }
    private func outputChoice(_ row: [String: JSONValue]) -> GradientEndpointChoice? {
        guard case .object(let selected) = row["output"] else { return nil }
        return endpoints.first { $0.node == selected["node"]?.stringValue && $0.port == selected["port"]?.stringValue }
    }
    private func canonical() throws -> GradientPolicyConfiguration {
        let configuration = try GradientPolicyConfiguration(values: draftConfiguration)
        for row in requests {
            guard let output = outputChoice(row), output.dtype == .float32 else {
                throw GraphConfigurationError("Choose a current float32 result endpoint from the model connected to Train.")
            }
            if row["reduction"] == .string("upstream") {
                guard case .object(let upstream) = row["upstream"], case .array(let shape) = upstream["shape"],
                      let rows = row["row_count"]?.integerValue,
                      shape == ([JSONValue.integer(Int64(rows))] + output.shape.map { .integer(Int64($0)) }) else {
                    throw GraphConfigurationError("Upstream shape must match selected rows followed by the exact result dimensions.")
                }
            }
            guard case .array(let targets) = row["targets"] else { throw GraphConfigurationError("Choose a derivative target.") }
            for raw in targets {
                guard case .object(let target) = raw else { throw GraphConfigurationError("Choose a typed derivative target.") }
                if target["kind"] == .string("input") {
                    guard let input = endpoints.first(where: { $0.isModelInput && $0.node == target["node"]?.stringValue && $0.port == target["port"]?.stringValue }) else {
                        throw GraphConfigurationError("A selected input is unavailable. Choose a current model input or remove the target.")
                    }
                    guard input.dtype == .float32 || (input.dtype == .int64 && row["unavailable"] == .string("reportUnavailable")) else {
                        throw GraphConfigurationError("Integer inputs have no derivative values. Choose Report unavailable or select a float32 model input.")
                    }
                } else {
                    guard owners.contains(where: { $0.node == target["node"]?.stringValue && $0.parameter == target["parameter"]?.stringValue }) else {
                        throw GraphConfigurationError("A learned array is unavailable. Choose a current model parameter or remove the target.")
                    }
                }
            }
        }
        try validateContext(configuration.values)
        return configuration
    }
    private func changed() { rejection = nil; report() }
    private func setRows(_ rows: [[String: JSONValue]]) { draftConfiguration["requests"] = .array(rows.map { .object($0) }); changed() }
    @discardableResult private func replace(_ index: Int, _ key: String, _ value: JSONValue) -> Bool {
        guard !isBusy, requests.indices.contains(index) else { return false }
        var rows = requests; rows[index][key] = value; setRows(rows); return true
    }
    @discardableResult func setAccumulationSteps(_ value: Int) -> Bool {
        guard !isBusy, (1...32).contains(value) else { return false }
        draftConfiguration["accumulation_steps"] = .integer(Int64(value)); changed(); return true
    }
    @discardableResult func setGradientScale(_ value: String) -> Bool {
        guard !isBusy, ["mean", "sum"].contains(value) else { return false }
        draftConfiguration["gradient_scale"] = .string(value); changed(); return true
    }
    @discardableResult func addRequest() -> Bool {
        guard !isBusy, !hasUncommittedNumbers, requests.count < 32,
              let output = endpoints.first(where: { $0.dtype == .float32 && !$0.isModelInput }) ?? endpoints.first(where: { $0.dtype == .float32 }) else { return false }
        let target: JSONValue
        if let input = endpoints.first(where: { $0.isModelInput && $0.dtype == .float32 }) {
            target = .object(["kind": .string("input"), "node": .string(input.node), "port": .string(input.port)])
        } else if let owner = owners.first {
            target = .object(["kind": .string("parameter"), "node": .string(owner.node), "parameter": .string(owner.parameter)])
        } else { return false }
        var number = 1
        while requests.contains(where: { $0["id"] == .string("request-" + String(number)) }) { number += 1 }
        var rows = requests
        rows.append(["id": .string("request-" + String(number)), "output": .object(["node": .string(output.node), "port": .string(output.port)]), "partition": .string("validation"), "row_start": 0, "row_count": 1, "reduction": .string("sum"), "upstream": .null, "targets": .array([target]), "unavailable": .string("error")])
        setRows(rows); return true
    }
    @discardableResult func removeRequest(at index: Int) -> Bool {
        guard !isBusy, !hasUncommittedNumbers, requests.indices.contains(index) else { return false }
        var rows = requests; rows.remove(at: index); setRows(rows); resetGeneration += 1; return true
    }
    @discardableResult func moveRequest(from: Int, to: Int) -> Bool {
        guard !isBusy, !hasUncommittedNumbers, requests.indices.contains(from), requests.indices.contains(to) else { return false }
        var rows = requests; let row = rows.remove(at: from); rows.insert(row, at: to); setRows(rows); resetGeneration += 1; return true
    }
    @discardableResult func setRequestID(_ value: String, at index: Int) -> Bool {
        // Retain unfinished names while editing; strict ASCII/nonempty validation belongs to Apply.
        guard value.utf8.count <= 64, !value.contains("\0") else { return false }
        return replace(index, "id", .string(value))
    }
    @discardableResult func setOutput(_ id: String, at index: Int) -> Bool {
        guard let choice = endpoints.first(where: { $0.id == id && $0.dtype == .float32 }) else { return false }
        return replace(index, "output", .object(["node": .string(choice.node), "port": .string(choice.port)]))
    }
    @discardableResult func setPartition(_ value: String, at index: Int) -> Bool { ["training", "validation"].contains(value) && replace(index, "partition", .string(value)) }
    @discardableResult func setRowStart(_ value: Int64, at index: Int) -> Bool { value >= 0 && replace(index, "row_start", .integer(value)) }
    @discardableResult func setRowCount(_ value: Int, at index: Int) -> Bool { (1...32).contains(value) && replace(index, "row_count", .integer(Int64(value))) }
    @discardableResult func setUnavailable(_ value: String, at index: Int) -> Bool { ["error", "reportUnavailable"].contains(value) && replace(index, "unavailable", .string(value)) }
    @discardableResult func setReduction(_ value: String, at index: Int) -> Bool {
        guard !isBusy, !hasUncommittedNumbers, requests.indices.contains(index), ["sum", "mean", "upstream"].contains(value) else { return false }
        var rows = requests; let id = rows[index]["id"]?.stringValue ?? String(index)
        if rows[index]["reduction"] == .string("upstream"), let previous = rows[index]["upstream"] { dormantUpstream[id] = previous }
        if value == "upstream" {
            if let previous = dormantUpstream[id] { rows[index]["upstream"] = previous }
            else {
                guard let output = outputChoice(rows[index]), let count = rows[index]["row_count"]?.integerValue, (1...32).contains(count) else { return false }
                let shape = [Int(count)] + output.shape
                guard let elements = Self.elements(shape) else { return false }
                rows[index]["upstream"] = .object(["dtype": .string("float32"), "shape": .array(shape.map { .integer(Int64($0)) }), "values": .array(Array(repeating: .number(0), count: elements))])
            }
        } else { rows[index]["upstream"] = .null }
        rows[index]["reduction"] = .string(value); setRows(rows); return true
    }
    @discardableResult private func addTarget(_ value: JSONValue, at index: Int) -> Bool {
        guard requests.indices.contains(index), case .array(var targets) = requests[index]["targets"], targets.count < 32, !targets.contains(value) else { return false }
        targets.append(value); return replace(index, "targets", .array(targets))
    }
    @discardableResult func addInputTarget(_ id: String, at index: Int) -> Bool {
        guard let choice = endpoints.first(where: { $0.id == id && $0.isModelInput && [.float32, .int64].contains($0.dtype) }) else { return false }
        return addTarget(.object(["kind": .string("input"), "node": .string(choice.node), "port": .string(choice.port)]), at: index)
    }
    @discardableResult func addParameterTarget(_ id: String, at index: Int) -> Bool {
        guard let choice = owners.first(where: { $0.id == id }) else { return false }
        return addTarget(.object(["kind": .string("parameter"), "node": .string(choice.node), "parameter": .string(choice.parameter)]), at: index)
    }
    @discardableResult func removeTarget(at target: Int, request index: Int) -> Bool {
        guard requests.indices.contains(index), case .array(var targets) = requests[index]["targets"], targets.indices.contains(target) else { return false }
        targets.remove(at: target); return replace(index, "targets", .array(targets))
    }
    private static func elements(_ shape: [Int]) -> Int? {
        guard (1...8).contains(shape.count) else { return nil }
        var count = 1
        for dimension in shape { guard (1...4096).contains(dimension), count <= 65_536 / dimension else { return nil }; count *= dimension }
        return count
    }
    @discardableResult func setUpstream(shape: [Int], values: [Double], at index: Int) -> Bool {
        guard requests.indices.contains(index), requests[index]["reduction"] == .string("upstream"), Self.elements(shape) == values.count,
              values.allSatisfy({ $0.isFinite && Float($0).isFinite && ($0 == 0 || Float($0) != 0) }) else { return false }
        return replace(index, "upstream", .object(["dtype": .string("float32"), "shape": .array(shape.map { .integer(Int64($0)) }), "values": .array(values.map { .number($0) })]))
    }
    func numericOrigin(_ key: String) -> String { origin + ":" + String(resetGeneration) + ":" + key }
    private func numericKey(_ key: String, configuration: [String: JSONValue]? = nil) -> Bool {
        if key == "accumulation_steps" { return true }
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 3, parts[0] == "request", let index = Int(parts[1]), case .array(let rows) = (configuration ?? draftConfiguration)["requests"], rows.indices.contains(index) else { return false }
        if parts.count == 3 { return ["row_start", "row_count"].contains(parts[2]) }
        guard parts.count == 4, parts[2] == "upstream", ["shape", "values"].contains(parts[3]), case .object(let row) = rows[index] else { return false }
        return row["reduction"] == .string("upstream")
    }
    @discardableResult func setRawNumericDraft(_ key: String, text: String, origin: String) -> Bool {
        guard !isBusy, origin == numericOrigin(key), numericKey(key), text.utf8.count <= (key.hasSuffix("upstream.values") ? 32_768 : 256) else { return false }
        rawNumericDrafts[key] = text; changed(); return true
    }
    @discardableResult func commitNumericDraft(_ key: String) -> Bool {
        guard !isBusy, numericKey(key), let text = rawNumericDrafts[key] else { return false }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var accepted = false
        if key == "accumulation_steps" {
            if let number = Int(trimmed), String(number) == trimmed { accepted = setAccumulationSteps(number) }
        } else {
            let parts = key.split(separator: ".").map(String.init); let index = Int(parts[1])!
            if parts.count == 3 {
                if let number = Int64(trimmed), String(number) == trimmed {
                    accepted = parts[2] == "row_start" ? setRowStart(number, at: index) : (Int(exactly: number).map { setRowCount($0, at: index) } ?? false)
                }
            } else if case .object(let upstream) = requests[index]["upstream"], case .array(let oldShape) = upstream["shape"], case .array(let oldValues) = upstream["values"] {
                let prefix = "request.\(index).upstream."
                let shapeKey = prefix + "shape", valuesKey = prefix + "values"
                func tokens(_ text: String) -> [String] { text.split { $0.isWhitespace || $0 == "," || $0 == "×" }.map(String.init) }
                let shape: [Int]?
                if let rawShape = rawNumericDrafts[shapeKey] {
                    let parts = tokens(rawShape), parsed = tokens(rawShape).compactMap(Int.init)
                    shape = !parts.isEmpty && parsed.count == parts.count ? parsed : nil
                } else { shape = oldShape.compactMap { $0.integerValue.flatMap { Int(exactly: $0) } } }
                let numbers: [Double]?
                if let rawValues = rawNumericDrafts[valuesKey] {
                    let parts = tokens(rawValues), parsed = tokens(rawValues).compactMap(Double.init)
                    numbers = !parts.isEmpty && parsed.count == parts.count ? parsed : nil
                } else { numbers = oldValues.compactMap(\.doubleValue) }
                if let shape, let numbers, let output = outputChoice(requests[index]),
                   let rows = requests[index]["row_count"]?.integerValue,
                   shape == [Int(rows)] + output.shape {
                    accepted = setUpstream(shape: shape, values: numbers, at: index)
                }
                if accepted { rawNumericDrafts.removeValue(forKey: shapeKey); rawNumericDrafts.removeValue(forKey: valuesKey) }
            }
        }
        if accepted { rawNumericDrafts.removeValue(forKey: key); rejection = nil; report() }
        else { rejection = "Enter complete bounded numbers, then use the value. The unfinished text remains available for repair." }
        return accepted
    }
    /// Editor snapshots may contain known incomplete edits; they are never executable recipes.
    private static func structuralDraft(_ configuration: [String: JSONValue]) -> [String: JSONValue]? {
        guard Set(configuration.keys) == ["accumulation_steps", "gradient_scale", "requests"],
              case .integer(let steps) = configuration["accumulation_steps"], (1...32).contains(steps),
              let scale = configuration["gradient_scale"]?.stringValue, ["mean", "sum"].contains(scale),
              case .array(let requests) = configuration["requests"], requests.count <= 32,
              let bytes = try? JSONEncoder().encode(configuration), bytes.count <= GradientPolicyConfiguration.maximumRecipeBytes else { return nil }
        func name(_ value: JSONValue?) -> Bool { guard let text = value?.stringValue else { return false }; return !text.isEmpty && !text.contains("\0") }
        var normalized: [JSONValue] = [], totalUpstream = 0, identities = Set<String>()
        for raw in requests {
            guard case .object(var row) = raw,
                  Set(row.keys) == ["id", "output", "partition", "row_start", "row_count", "reduction", "upstream", "targets", "unavailable"],
                  let id = row["id"]?.stringValue, id.utf8.count <= 64, !id.contains("\0"),
                  case .object(let output) = row["output"], Set(output.keys) == ["node", "port"], name(output["node"]), name(output["port"]),
                  let partition = row["partition"]?.stringValue, ["training", "validation"].contains(partition),
                  case .integer(let start) = row["row_start"], start >= 0,
                  case .integer(let count) = row["row_count"], (1...32).contains(count), !start.addingReportingOverflow(count).overflow,
                  let reduction = row["reduction"]?.stringValue, ["sum", "mean", "upstream"].contains(reduction),
                  let unavailable = row["unavailable"]?.stringValue, ["error", "reportUnavailable"].contains(unavailable),
                  case .array(let targets) = row["targets"], targets.count <= 32 else { return nil }
            for rawTarget in targets {
                guard case .object(let target) = rawTarget, let kind = target["kind"]?.stringValue, ["input", "parameter"].contains(kind) else { return nil }
                let field = kind == "input" ? "port" : "parameter"
                guard Set(target.keys) == ["kind", "node", field], name(target["node"]), name(target[field]) else { return nil }
                let parts = [kind, target["node"]!.stringValue!, target[field]!.stringValue!]
                identities.insert(parts.map { "\($0.utf8.count):\($0)" }.joined())
                guard identities.count <= 32 else { return nil }
            }
            if reduction == "upstream" {
                guard case .object(var upstream) = row["upstream"], Set(upstream.keys) == ["dtype", "shape", "values"], upstream["dtype"] == .string("float32"),
                      case .array(let shapeValues) = upstream["shape"], (1...8).contains(shapeValues.count),
                      case .array(let values) = upstream["values"], values.count <= 65_536 else { return nil }
                var shape: [Int] = []
                for value in shapeValues { guard case .integer(let dimension) = value, (1...4096).contains(dimension), let size = Int(exactly: dimension) else { return nil }; shape.append(size) }
                guard let elements = elements(shape), elements == values.count, totalUpstream <= 65_536 - elements else { return nil }
                totalUpstream += elements
                var numbers: [JSONValue] = []
                for value in values {
                    guard let number = value.doubleValue, number.isFinite, Float(number).isFinite, number == 0 || Float(number) != 0 else { return nil }
                    numbers.append(.number(number))
                }
                upstream["values"] = .array(numbers); row["upstream"] = .object(upstream)
            } else if row["upstream"] != .null { return nil }
            normalized.append(.object(row))
        }
        var result = configuration; result["requests"] = .array(normalized); return result
    }
    func snapshotData() throws -> Data {
        guard let configuration = Self.structuralDraft(draftConfiguration) else { throw GraphConfigurationError("The retained gradient draft has unknown fields or exceeds its structural bounds.") }
        let snapshot: JSONValue = .object(["schemaVersion": 1, "configuration": .object(configuration), "expectedConfiguration": .object(expectedConfiguration), "rawNumericDrafts": .object(rawNumericDrafts.mapValues { .string($0) })])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(snapshot)
        guard data.count <= Self.maximumSnapshotBytes else { throw GraphConfigurationError("Gradient editor draft exceeds 131,072 bytes.") }
        return data
    }
    @discardableResult func restorePending(_ data: Data) -> Bool {
        guard !isBusy, data.count <= Self.maximumSnapshotBytes,
              let value = try? JSONDecoder().decode(JSONValue.self, from: data), case .object(let fields) = value,
              Set(fields.keys) == ["schemaVersion", "configuration", "expectedConfiguration", "rawNumericDrafts"], fields["schemaVersion"] == .integer(1),
              case .object(let configuration) = fields["configuration"], case .object(let expected) = fields["expectedConfiguration"],
              let restored = Self.structuralDraft(configuration), let original = try? GradientPolicyConfiguration(values: expected),
              original.values == expectedConfiguration, case .object(let raw) = fields["rawNumericDrafts"], raw.count <= 129 else { return false }
        var retained: [String: String] = [:]
        for (key, value) in raw {
            guard numericKey(key, configuration: restored), let text = value.stringValue, text.utf8.count <= (key.hasSuffix("upstream.values") ? 32_768 : 256) else { return false }
            retained[key] = text
        }
        draftConfiguration = restored; rawNumericDrafts = retained; dormantUpstream.removeAll(); resetGeneration += 1; rejection = nil; report(); return true
    }
    @discardableResult func apply() -> Bool {
        guard !isBusy, !hasUncommittedNumbers else { return false }
        do {
            let updated = try canonical().values
            guard hasChanges else { return true }
            guard commit(expectedConfiguration, updated) else { rejection = "The source changed. Reopen this editor before applying the draft."; return false }
            baseline = updated; expectedConfiguration = updated; draftConfiguration = updated
            dormantUpstream.removeAll(); resetGeneration += 1; rejection = nil; report(); return true
        } catch { rejection = error.localizedDescription; return false }
    }
    @discardableResult func discard() -> Bool {
        guard !isBusy else { return false }
        draftConfiguration = baseline; rawNumericDrafts.removeAll(); dormantUpstream.removeAll(); resetGeneration += 1; rejection = nil; report(); return true
    }
    func releasePendingEdits() { pending(false) }
}
